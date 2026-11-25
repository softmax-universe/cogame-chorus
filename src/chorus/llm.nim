## Claude-backed decision making for Chorus. Each seat's policy is just a
## prompt: the game server composes the seat's view (its voice, the whole
## grid, the chord plan, the live score, its own counterfactual credit, the
## other seats' messages, its notes) plus that seat's prompt and asks Claude
## which bar it writes.
##
## Decisions within a turn are simultaneous by rule, so all four requests go
## out as ONE parallel batch (curly.makeRequests); invalid replies are
## retried once as a smaller batch carrying an explicit hint, and anything
## still failing ends the episode with an explicit failure while preserving
## the submitted raw data (no implicit scripted fallback).
##
## Credentials, in order of preference:
##   COWORLD_LLM_ENDPOINT            - hosted sidecar
##   Bedrock bearer token            - local play
##   ANTHROPIC_API_KEY                - the key itself
##   ANTHROPIC_API_KEY_URI            - a URI holding the key
## Explicitly scripted seats (PLAYER_SCRIPTED=arpeggio|pedal) run their
## selected baseline deterministically without calling an LLM.

import
  std/[json, options, os, strutils, times, unicode],
  bitworld/runtime,
  curly,
  sim,
  training,
  prompts

export prompts

const
  AnthropicUrl = "https://api.anthropic.com/v1/messages"
  AnthropicVersion = "2023-06-01"
  BedrockAnthropicVersion = "bedrock-2023-05-31"
  ## Any error text that could reach the replay is cut at this many runes.
  MaxErrorLen* = 200

type
  ScriptKind* = enum
    skNone = "none"
    skArpeggio = "arpeggio"
    skPedal = "pedal"

  FailureKind* = enum
    fkNone = "none"
    fkAction = "action_failure"
    fkTransport = "transport_failure"
    fkRegistration = "registration_failure"

  Decision* = object
    target*: int
    steps*: seq[int]
    say*: string
    notes*: string      ## "" when the reply carried none
    scripted*: bool     ## true ONLY when the seat explicitly selected a
                        ## scripted baseline (`arpeggio` or `pedal`). The
                        ## replay's `bar.scripted` flag is this field.
    failed*: bool       ## true when an LLM seat failed to produce a legal
                        ## action (no credentials, transport error, or invalid
                        ## reply after retry). Never falls back to scripted.
    failureKind*: FailureKind
    failureReason*: string
    rawReply*: string   ## last raw reply/body preserved on failure

  LlmTransport = enum
    ltNone, ltSidecar, ltBedrock, ltAnthropic

  LlmClient* = ref object
    curl: Curly
    transport: LlmTransport
    apiKey: string          ## anthropic transport
    sidecarEndpoint: string
    bedrockEndpoint: string ## local Bedrock transport
    bedrockModels: seq[string]  ## candidates, tried in order on denial
    bedrockModel: int           ## index into bedrockModels
    bedrockToken: string
    model: string         ## direct-Anthropic transport only; Bedrock
                          ## picks from bedrockModels instead
    maxOutputTokens: int
    timeoutSeconds: int
    nativeAttempts*: seq[seq[DecisionAttempt]]
    disabled*: bool   ## true once credentials are known-unavailable

proc parseScriptKind*(text: string): ScriptKind =
  ## Canonical PLAYER_SCRIPTED values: "" / "none" for LLM-driven seats,
  ## "arpeggio" for the arpeggio baseline, "pedal" for the pedal baseline.
  let cleaned = text.strip().toLowerAscii()
  case cleaned
  of "", "none": skNone
  of "arpeggio": skArpeggio
  of "pedal": skPedal
  else:
    raise newException(ChorusError,
      "unknown PLAYER_SCRIPTED value: " & text &
        " (expected arpeggio, pedal, or empty)")

proc resolveApiKey(): string =
  result = getEnv("ANTHROPIC_API_KEY").strip()
  if result.len > 0:
    return
  let uri = getEnv("ANTHROPIC_API_KEY_URI").strip()
  if uri.len == 0:
    return ""
  try:
    result = readCogameUri(uri, "ANTHROPIC_API_KEY_URI").strip()
  except CatchableError as error:
    echo "chorus llm: failed to fetch ANTHROPIC_API_KEY_URI: ", error.msg
    result = ""

proc bedrockModelIds(): seq[string] =
  ## Bedrock inference-profile candidates, tried in order. BEDROCK_MODEL
  ## pins a single id; without it, fall through this list — model access is
  ## a per-account Marketplace subscription, so an id that works in one
  ## account 403s in another. The config "model" field is NOT consulted
  ## here: it applies to the direct-Anthropic transport only, and the
  ## haiku-first ordering below is a shared-capacity decision that trumps
  ## per-game preference.
  let pinned = getEnv("BEDROCK_MODEL").strip()
  if pinned.len > 0:
    return @[pinned]
  ## Haiku leads: hosted Bedrock capacity is shared account-wide and the
  ## sonnet profiles run out of daily tokens first.
  @[
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  ]

proc tryNextBedrockModel(client: LlmClient, why: string): bool =
  if client.transport != ltBedrock or
      client.bedrockModel + 1 >= client.bedrockModels.len:
    return false
  client.bedrockModel.inc
  echo "chorus llm: ", client.bedrockModels[client.bedrockModel - 1],
    " unusable (", why, "); falling back to ",
    client.bedrockModels[client.bedrockModel]
  true

proc bedrockUrl(client: LlmClient): string =
  client.bedrockEndpoint & "/model/" &
    client.bedrockModels[client.bedrockModel] & "/invoke"

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    model: config.model,
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.llmTimeoutSeconds
  )
  let sidecarEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  if sidecarEndpoint.len > 0:
    result.transport = ltSidecar
    result.sidecarEndpoint = sidecarEndpoint.strip(chars = {'/'}, leading = false)
    result.model = getEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
    result.curl = newCurly()
    return
  let bedrockEndpoint = getEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME").strip()
  let bedrockToken = getEnv("AWS_BEARER_TOKEN_BEDROCK").strip()
  if bedrockEndpoint.len > 0 or bedrockToken.len > 0:
    let region = getEnv("AWS_REGION",
      getEnv("AWS_DEFAULT_REGION", "us-west-2"))
    let endpoint =
      if bedrockEndpoint.len > 0: bedrockEndpoint
      else: "https://bedrock-runtime." & region & ".amazonaws.com"
    result.transport = ltBedrock
    result.bedrockEndpoint = endpoint.strip(chars = {'/'}, leading = false)
    result.bedrockModels = bedrockModelIds()
    result.bedrockToken = bedrockToken
    result.curl = newCurly()
    echo "chorus llm: bedrock transport, model ",
      result.bedrockModels[result.bedrockModel],
      ", url ", result.bedrockUrl
    return
  result.apiKey = resolveApiKey()
  if result.apiKey.len > 0:
    result.transport = ltAnthropic
    result.curl = newCurly()
    echo "chorus llm: anthropic transport, model ", result.model
  else:
    result.transport = ltNone
    result.disabled = true
    echo "chorus llm: no LLM credentials configured"

# ---- Scripted baselines -----------------------------------------------------

proc clampToken(token: int): int =
  max(0, min(MaxToken, token))

proc emptySteps(): seq[int] =
  result = newSeq[int](Steps)
  for index in 0 ..< Steps:
    result[index] = Rest

proc arpeggioBar*(sim: Sim, voice, bar: int): seq[int] =
  ## The strong scripted baseline: chord tones spread across the voices,
  ## rotated a step each bar so the piece never settles on the repetition floor.
  result = emptySteps()
  let root = sim.chords[bar]
  let tones = [clampToken(root), clampToken(root + 2), clampToken(root + 4)]
  let rot = bar mod 3
  var onsets: seq[int]
  var tokens: seq[int]
  case voice
  of 0:
    onsets = @[0, 8]
    tokens =
      if bar mod 2 == 0: @[tones[0], tones[0]]
      else: @[tones[0], tones[2]]
  of 1:
    onsets = @[0, 4, 8, 12]
    tokens = @[tones[0], tones[1], tones[2], tones[1]]
  of 2:
    onsets = @[2, 6, 10, 14]
    tokens = @[tones[1], tones[2], tones[0], tones[2]]
  else:
    onsets = @[0, 3, 6, 10, 12]
    tokens = @[tones[2], clampToken(tones[0] + 7), tones[1], tones[2],
      tones[1]]
  for index, step in onsets:
    result[step] = tokens[(index + rot) mod tokens.len]

proc pedalBar*(sim: Sim, voice, bar: int): seq[int] =
  ## The weak, honest filler: a root pedal on the downbeat and a fifth
  ## halfway through the odd bars. Too thin to sit in the density band, so
  ## a table of four pedals scores well below a table of four arpeggios.
  result = emptySteps()
  let root = sim.chords[bar]
  result[0] = clampToken(root)
  if bar mod 2 == 1:
    result[8] = clampToken(root + 4)

proc scriptedAction*(sim: Sim, seat: int, kind: ScriptKind): Decision =
  ## Rule-based baseline for `seat`. Always legal; never talks or notes;
  ## always writes this turn's new bar.
  let voice = sim.voiceOf[seat]
  result.target = sim.turn
  result.scripted = true
  result.steps =
    case kind
    of skPedal: pedalBar(sim, voice, sim.turn)
    else: arpeggioBar(sim, voice, sim.turn)

# ---- Prompt building --------------------------------------------------------

proc seatName(sim: Sim, seat: int): string =
  sim.names[seat]

proc voiceLabel(sim: Sim, voice: int): string =
  ## "Gizmo (ALTO)".
  sim.seatName(sim.seatOf[voice]) & " (" & VoiceNames[voice].toUpperAscii() &
    ")"

proc one(value: float): string =
  formatFloat(value, ffDecimal, 1)

proc two(value: float): string =
  formatFloat(value, ffDecimal, 2)

proc signed(value: float): string =
  (if value >= 0.0: "+" else: "") & one(value)

proc chordDegrees(root: int): string =
  $root & ", " & $(root + 2) & ", " & $(root + 4)

proc chordPlanLine*(sim: Sim): string =
  var parts: seq[string]
  for bar, degree in sim.chords:
    parts.add("bar " & $bar & " " & chordName(degree))
  parts.join(" | ")

proc tokenCell(token: int): string =
  let text = if token < 0: "." else: $token
  align(text, 3)

proc gridBlock(sim: Sim, seat: int): string =
  ## Every bar written so far, all four voices, column-aligned oldest first.
  let ownVoice = sim.voiceOf[seat]
  var lines: seq[string]
  for bar in 0 .. sim.turn:
    var head = "bar " & $bar & "  " & chordName(sim.chords[bar])
    if bar == sim.turn:
      var hold = ""
      for step in 0 ..< Steps:
        hold.add(tokenCell(sim.grid[ownVoice][bar][step]))
      head.add("   (this turn — your voice currently holds:" & hold & ")")
    lines.add(head)
    for voice in 0 ..< Voices:
      var row = "  " & align(VoiceNames[voice].toUpperAscii() & "(" &
        sim.seatName(sim.seatOf[voice]) & ")", 22) & " "
      for step in 0 ..< Steps:
        row.add(tokenCell(sim.grid[voice][bar][step]))
      if voice == ownVoice:
        row.add("   <- YOU")
      lines.add(row)
  lines.join("\n")

proc systemPrompt*(sim: Sim, seat: int): string =
  let me = sim.seatName(seat)
  let voice = sim.voiceOf[seat]
  let voiceUp = VoiceNames[voice].toUpperAscii()
  var others: seq[string]
  for other in 0 ..< Voices:
    if other != voice:
      others.add(sim.voiceLabel(other))
  let piece = "THE PIECE: " & $sim.config.bars & " bars of " & $Steps &
    " steps, key " & sim.keyName() & ", " & $sim.bpm &
    " BPM. Every step is one note or a rest, and a note lasts exactly one " &
    "step.\n\n"
  let notation = "NOTATION: a bar is " & $Steps & " tokens. -1 is a rest. 0.." &
    $MaxToken & " are scale degrees: 0 is the tonic, 6 the seventh, 7 the " &
    "tonic an octave up, 13 the seventh two octaves up. Your voice sounds " &
    "in the " & voiceUp & " register (MIDI " & $sim.baseMidi(voice) & ".." &
    $(sim.baseMidi(voice) + 23) & ").\n\n"
  let chords = "THE CHORD PLAN is fixed, shared and public: " &
    sim.chordPlanLine() & ". The chord tones of a chord rooted on degree r " &
    "are r, r+2 and r+4.\n\n"
  if sim.config.selectedPromptVariant != pvCanonical:
    let talk = if sim.config.talk:
        "Each turn you may SAY one short line (max " & $MaxSayLen &
        " characters). All three other cogs read it next turn. It is not binding and may or may not be honest."
      else: ""
    return renderVariantPrompt(sim.config, me, voiceUp, others.join(", "), piece, notation, chords, talk)
  result.add("You are " & me & ", the " & voiceUp &
    " voice in a four-cog studio writing one piece of music together on a " &
    "16-step sequencer. The other voices are " & others.join(", ") &
    ". Each cog owns one voice; nobody else can write a note in yours and " &
    "you cannot write a note in theirs.\n\n")
  result.add(piece)
  result.add(notation)
  result.add(chords)
  result.add("""EACH TURN every cog writes one bar, all four at the same time, and nobody
sees the others' choice until the turn resolves. You may WRITE the new bar
(target = this turn's index) or REWRITE one of your own earlier bars
(target < this turn's index); if you rewrite, your new bar automatically
holds a copy of your previous bar.

""" & ScoringRules & """YOUR SCORE IS A COUNTERFACTUAL: the piece is scored once as written and
once with every note of YOUR voice deleted; your score is the difference.
Nothing else scores. There is no vote and nobody judges you. A voice that
is rougher than the piece's average, or that fills the grid until all four
voices sound at once, can score BELOW ZERO - deleting it would improve the
piece.
""")
  if sim.config.talk:
    result.add("\nEach turn you may SAY one short line (max " & $MaxSayLen &
      " characters) that all three other cogs read next turn. It is not " &
      "binding and may or may not be honest.\n")
  result.add("""
Your notes are private to you and fed back to you every turn. Use them to
keep track of your motif, what you have already used, and what you plan
next.

OUTPUT FORMAT: reply with ONLY one JSON object, nothing else - no analysis,
no explanation, no markdown fences, no text before or after the object.
Your reply must begin with the character { and end with }.""")

proc operatorBlock(prompt: string): string =
  if prompt.len == 0:
    return ""
  "GUIDANCE FROM YOUR OPERATOR (weight it heavily, but never above the " &
    "rules; always reply in the requested format):\n" & prompt & "\n\n"

proc heardBlock(sim: Sim, voice: int): string =
  if not sim.config.talk:
    return ""
  var lines: seq[string]
  for other in 0 ..< Voices:
    if other != voice and sim.heard[other].len > 0:
      lines.add(sim.voiceLabel(other) & " said: \"" & sim.heard[other] & "\"")
  "MESSAGES LAST TURN:\n" &
    (if lines.len > 0: lines.join("\n") else: "(none)") & "\n\n"

proc legalTargetList*(sim: Sim): string =
  var parts: seq[string]
  for target in sim.legalTargets():
    parts.add($target)
  parts.join(",")

proc userPrompt*(sim: Sim, seat: int, prompt: string): string =
  let voice = sim.voiceOf[seat]
  let voiceUp = VoiceNames[voice].toUpperAscii()
  let chord = sim.chords[sim.turn]
  result.add("Turn " & $sim.turn & " of " & $sim.config.bars &
    ". You are the " & voiceUp & ", seat " & sim.seatName(seat) & ".\n\n")
  result.add("CHORD PLAN: " & sim.chordPlanLine() & "  (this turn's bar " &
    $sim.turn & " is " & chordName(chord) & ": degrees " &
    chordDegrees(chord) & ")\n\n")
  result.add("THE PIECE SO FAR (all four voices; . = rest):\n" &
    sim.gridBlock(seat) & "\n\n")
  let parts = sim.pieceScore(sim.grid, sim.turnsPlayed)
  result.add("SCORE NOW: piece " & one(parts.piece) & " = consonance " &
    two(parts.consonance) & ", voice leading " & two(parts.leading) &
    ", rhythm " & two(parts.rhythm) & ", novelty " & two(parts.novelty) &
    ".\n\n")
  let without =
    sim.pieceScore(mutedGrid(sim.grid, voice), sim.turnsPlayed).piece
  result.add("YOUR COUNTERFACTUAL CREDIT NOW: " &
    signed(parts.piece - without) & " (the piece without your voice scores " &
    one(without) & ").\n\n")
  result.add(sim.heardBlock(voice))
  result.add("YOUR NOTES FROM EARLIER TURNS:\n" &
    (if sim.notes[seat].len > 0: sim.notes[seat] else: "(none)") & "\n\n")
  result.add(operatorBlock(prompt))
  result.add("LEGAL TARGETS THIS TURN: " & sim.legalTargetList() &
    "  (bar " & $sim.turn & " is the new bar).\n\n")
  result.add("Reply with one JSON object. Example for the new bar: {\"target\": " &
    $sim.turn & ", \"steps\": [0,-1,-1,4, -1,-1,2,-1, 0,-1,-1,4, -1,-1,-1,-1]}. " &
    "Choose a legal target and write exactly " & $Steps &
    " integers in one flat array (four groups of four). Use -1 for a rest or 0.." &
    $MaxToken & " for a note. Only target and steps are required. " &
    (if sim.config.talk: "Optional say: one line up to " & $MaxSayLen &
      " characters. " else: "") &
    "Omit notes to keep your private notes, or replace them with a string up to " &
    $MaxNotesLen & " characters.")

proc retryHint*(sim: Sim, reason: string): string =
  "\nYour previous reply was invalid: " & reason &
    ". Respond with ONLY the requested JSON object, with \"steps\" exactly " &
    $Steps & " whole numbers each -1 or 0.." & $MaxToken &
    " and \"target\" one of " & sim.legalTargetList() & "."

# ---- Anthropic / Bedrock transport ------------------------------------------

proc stripOuterJsonFence(text: string): string =
  let trimmed = text.strip()
  if not trimmed.startsWith("```"):
    return trimmed
  let firstNl = trimmed.find('\n')
  if firstNl < 0:
    return trimmed
  let header = trimmed[3 ..< firstNl].strip().toLowerAscii()
  if header.len > 0 and header != "json":
    return trimmed
  if not trimmed.endsWith("```"):
    return trimmed
  let innerEnd = trimmed.len - 3
  if innerEnd <= firstNl:
    return trimmed
  trimmed[firstNl + 1 ..< innerEnd].strip()

proc extractJsonObject*(text: string): JsonNode =
  ## Parses a single JSON object from a model response, accepting only
  ## optional outer whitespace and a standard outer Markdown JSON fence.
  let candidate = stripOuterJsonFence(text)
  if candidate.len == 0 or candidate[0] != '{' or candidate[^1] != '}':
    var head = text.strip()
    if head.len > 160:
      head = head[0 ..< 160] & "..."
    raise newException(ChorusError, "response must be a single JSON object: " &
      head.replace("\n", " "))
  let parsed =
    try:
      parseJson(candidate)
    except CatchableError as error:
      raise newException(ChorusError, "invalid JSON object: " & error.msg)
  if parsed.kind != JObject:
    raise newException(ChorusError, "response must be a JSON object")
  parsed

proc requestFor(client: LlmClient, system, user: string, slot: int):
    tuple[url: string, headers: HttpHeaders, body: string] =
  var body = %*{
    "max_tokens": client.maxOutputTokens,
    "system": system,
    "messages": [{"role": "user", "content": user}]
  }
  var headers: HttpHeaders
  headers["content-type"] = "application/json"
  if client.transport == ltSidecar and slot >= 0:
    headers["X-Coworld-Player-Slot"] = $slot
  if client.transport == ltBedrock:
    body["anthropic_version"] = %BedrockAnthropicVersion
    if client.bedrockToken.len > 0:
      headers["authorization"] = "Bearer " & client.bedrockToken
    result.url = client.bedrockUrl()
  elif client.transport == ltSidecar:
    body["model"] = %client.model
    headers["anthropic-version"] = AnthropicVersion
    result.url = client.sidecarEndpoint & "/v1/messages"
  else:
    body["model"] = %client.model
    ## Only the Claude 5 / Opus tiers accept an effort setting; Haiku 4.5
    ## rejects the whole request with a 400 if it is present.
    if "haiku" notin client.model and "4-5" notin client.model:
      body["output_config"] = %*{"effort": "low"}
    headers["x-api-key"] = client.apiKey
    headers["anthropic-version"] = AnthropicVersion
    result.url = AnthropicUrl
  result.headers = headers
  result.body = $body

proc textOf(client: LlmClient, response: Response, error, url: string):
    string =
  ## The text of one batched reply, or a typed error describing why there is
  ## none. Transport/auth/throttle/HTTP errors raise ChorusTransportError;
  ## model refusals and truncated/malformed replies raise ChorusError.
  if error.len > 0:
    raise newException(ChorusTransportError, "llm transport: " & error)
  if response.code == 401 or response.code == 403:
    let detail = response.body[0 .. min(response.body.high, 400)]
    if "Model access is denied" in response.body and
        client.tryNextBedrockModel("no model access"):
      raise newException(ChorusTransportError,
        "bedrock model access denied: " & detail)
    client.disabled = true
    raise newException(ChorusTransportError,
      "llm auth failed (" & $response.code & ") at " & url & ": " & detail)
  if response.code == 429:
    let detail = response.body[0 .. min(response.body.high, 300)]
    discard client.tryNextBedrockModel("throttled")
    raise newException(ChorusTransportError, "llm throttled (429): " & detail)
  if response.code < 200 or response.code >= 300:
    raise newException(ChorusTransportError, "anthropic error " & $response.code &
      ": " & response.body[0 .. min(response.body.high, 300)])
  let payload = parseJson(response.body)
  if payload{"stop_reason"}.getStr() == "refusal":
    raise newException(ChorusError, "anthropic refusal")
  for contentBlock in payload["content"]:
    if contentBlock{"type"}.getStr() == "text":
      result.add(contentBlock{"text"}.getStr())
  if payload{"stop_reason"}.getStr() == "max_tokens" and '{' notin result:
    raise newException(ChorusError, "reply cut off at max_tokens before " &
      "any JSON: " & result[0 .. min(result.high, 160)].replace("\n", " "))

proc cleanText*(text: string, limit: int): string =
  ## Text over the cap is cut at a RUNE boundary with the cut marked. A byte
  ## slice through a multi-byte character would leave invalid UTF-8 in the
  ## replay and break a strict JSON parser.
  result = text.strip()
  if result.runeLen <= limit:
    return
  result = result.runeSubStr(0, limit - 1) & "…"

proc parseSteps*(node: JsonNode): seq[int] =
  ## `steps` must be a JSON array of exactly 16 integers, each -1 (rest) or
  ## 0..13. No string splitting, rest spelling aliases, float rounding, or
  ## null coercion.
  if node.isNil or node.kind == JNull:
    raise newException(ChorusError, "no steps in response")
  if node.kind != JArray:
    raise newException(ChorusError, "steps must be a JSON array of integers")
  for item in node:
    if item.kind != JInt:
      raise newException(ChorusError, "every step must be an integer, got: " & $item)
    result.add(int(item.getInt()))
  if result.len != Steps:
    raise newException(ChorusError,
      "steps must be exactly " & $Steps & " values, got " & $result.len)
  for value in result:
    if value != Rest and (value < 0 or value > MaxToken):
      raise newException(ChorusError,
        "every step is -1 (rest) or 0.." & $MaxToken & ", got " & $value)

proc parseDecision*(payload: JsonNode, turn: int): Decision =
  ## Parses a canonical Chorus action: required integer `target` in 0..turn,
  ## required 16-integer `steps` array, and optional string `say` / `notes`
  ## stored by the simulation within its published rune limits.
  if payload.isNil or payload.kind != JObject:
    raise newException(ChorusError, "decision must be a JSON object")
  if not payload.hasKey("target") or payload["target"].kind != JInt:
    raise newException(ChorusError, "target must be an integer in 0.." & $turn)
  let target = int(payload["target"].getInt())
  if target < 0 or target > turn:
    raise newException(ChorusError,
      "target must be 0.." & $turn & ": " & $target)
  result.target = target
  result.steps = parseSteps(payload{"steps"})
  if payload.hasKey("say"):
    if payload["say"].kind != JString:
      raise newException(ChorusError, "say must be a string")
    let say = payload["say"].getStr()
    if '\n' in say or '\r' in say:
      raise newException(ChorusError, "say must be a single line")
    result.say = say
  if payload.hasKey("notes"):
    if payload["notes"].kind != JString:
      raise newException(ChorusError, "notes must be a string")
    result.notes = payload["notes"].getStr()

proc decideAll*(
  client: LlmClient,
  sim: Sim,
  seats: seq[int],
  prompts: seq[string],
  scripted: seq[ScriptKind],
  registered: seq[bool] = @[]
): seq[Decision] =
  ## One decision per seat in `seats`, in order. Explicitly scripted seats run
  ## their selected baseline; unregistered seats fail explicitly without
  ## issuing any model request; LLM seats that lack credentials or fail both
  ## attempts return `failed = true` with preserved failure reason and raw
  ## reply (never falling back to a scripted baseline).
  let capture = getEnv(CogameSaveTrajectoryUriEnv).len > 0
  if capture: client.nativeAttempts = newSeq[seq[DecisionAttempt]](Seats)
  result = newSeq[Decision](seats.len)
  var open: seq[int]     ## indexes into `seats` still undecided
  var reasons = newSeq[string](seats.len)
  var failureKinds = newSeq[FailureKind](seats.len)
  var lastRawReplies = newSeq[string](seats.len)
  for index, seat in seats:
    if registered.len > 0 and (seat < 0 or seat >= registered.len or not registered[seat]):
      let reason = "seat " & $seat & " has not completed role registration"
      echo "chorus llm: ", reason
      result[index] = Decision(
        failed: true,
        failureKind: fkRegistration,
        failureReason: reason
      )
      continue
    let kind = scripted[seat]
    if kind != skNone:
      result[index] = scriptedAction(sim, seat, kind)
    elif client.disabled:
      let reason = "LLM credentials unavailable for non-scripted seat " & $seat
      echo "chorus llm: ", reason
      result[index] = Decision(
        failed: true,
        failureKind: fkTransport,
        failureReason: reason
      )
    else:
      open.add(index)
  for attempt in 0 .. 1:
    if open.len == 0 or client.disabled:
      break
    var batch: RequestBatch
    var pendingEvidence: seq[DecisionAttempt]
    for index in open:
      let seat = seats[index]
      var user = sim.userPrompt(seat, prompts[seat])
      if attempt > 0 and failureKinds[index] == fkAction:
        user.add(sim.retryHint(reasons[index]))
      let request = client.requestFor(systemPrompt(sim, seat), user, seat)
      if capture:
        pendingEvidence.add(nativeAttempt(sim.config.players[seat].name, sim.turn,
          seat, attempt, request.body))
      batch.post(request.url, request.headers, request.body, $index)
    let started = epochTime()
    let responses = client.curl.makeRequests(batch, client.timeoutSeconds)
    let latencyMs = int((epochTime() - started) * 1000.0)
    var stillOpen: seq[int]
    for position, index in open:
      let seat = seats[index]
      var evidence: DecisionAttempt
      if capture: evidence = pendingEvidence[position]
      lastRawReplies[index] =
        if responses[position].response.body.len > 0:
          responses[position].response.body
        else:
          responses[position].error
      try:
        if capture:
          evidence.receiveNativeResponse(responses[position].response,
            responses[position].error, latencyMs)
        let text = client.textOf(responses[position].response,
          responses[position].error, batch[position].url)
        lastRawReplies[index] = text
        let payload = extractJsonObject(text)
        var decision = parseDecision(payload, sim.turn)
        if not sim.config.talk and decision.say.len > 0:
          raise newException(ChorusError, "say is disabled when talk is off")
        ## Reject illegal replies here so the retry carries the hint.
        var probe = sim
        probe.applyBar(seat, decision.target, decision.steps, decision.say,
          decision.notes, false)
        if capture:
          evidence.parsedAction = %*{"target": decision.target, "steps": decision.steps,
            "say": (if sim.config.talk: decision.say else: ""), "notes": decision.notes}
          evidence.accepted = true
        result[index] = decision
      except ChorusTransportError as error:
        if capture: evidence.rejectionReason = some(error.msg)
        failureKinds[index] = fkTransport
        reasons[index] = cleanText(error.msg, MaxErrorLen)
        echo "chorus llm: seat ", seat, " attempt ", attempt, " transport failed"
        stillOpen.add(index)
      except CatchableError as error:
        if capture: evidence.rejectionReason = some(error.msg)
        failureKinds[index] = fkAction
        reasons[index] = cleanText(error.msg, MaxErrorLen)
        echo "chorus llm: seat ", seat, " attempt ", attempt, " action rejected"
        stillOpen.add(index)
      if capture: client.nativeAttempts[seat].add(evidence)
    open = stillOpen
  for index in open:
    let seat = seats[index]
    let reason =
      if reasons[index].len > 0: reasons[index]
      else: "LLM client disabled before decision completed"
    let kind =
      if failureKinds[index] != fkNone: failureKinds[index]
      else: fkTransport
    echo "chorus llm: seat ", seat, " failed without fallback (", $kind, ")"
    result[index] = Decision(
      failed: true,
      failureKind: kind,
      failureReason: reason,
      rawReply: lastRawReplies[index]
    )
