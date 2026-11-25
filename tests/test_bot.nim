## The scripted baselines must play whole episodes without ever proposing an
## illegal bar — they are both the no-credentials fallback (offline
## certification) and fieldable policies, so this is the completion path. The
## arpeggio baseline must also actually make music, or it is no partner worth
## beating; the pedal baseline must be the weaker of the two.

import std/[json, monotimes, os, strutils, times, unicode, unittest]
import chorus/[llm, sim]

proc fixture(seed: int, bars = 8): GameConfig =
  result = defaultGameConfig()
  result.seed = seed
  result.bars = bars
  result.sampled = true
  for index in 0 ..< Seats:
    result.players.add(PlayerConfig(name: "P" & $(index + 1)))
    result.tokens.add("t" & $index)

proc playScripted(config: GameConfig, kind: ScriptKind,
    strict = false): Sim =
  result = initSim(config)
  while not result.done:
    let turn = result.turn
    for seat in result.pendingSeats():
      let decision = scriptedAction(result, seat, kind)
      if strict:
        ## Every scripted bar must be legal AS IS: applyBar raises on
        ## anything else and would fail this test anyway, but assert the
        ## bounds explicitly so a regression names itself.
        check decision.target == turn
        check decision.steps.len == Steps
        var bounded = true
        for step in decision.steps:
          if step != Rest and (step < 0 or step > MaxToken):
            bounded = false
        check bounded
        check decision.say.len == 0
        check decision.notes.len == 0
      result.applyBar(seat, decision.target, decision.steps, decision.say,
        decision.notes, true)

suite "scripted baselines":
  test "both baselines play full episodes legally and fast, in every voice":
    for seed in [1, 5, 42, 1234]:
      for kind in [skArpeggio, skPedal]:
        let config = fixture(seed)
        let started = getMonoTime()
        let sim = playScripted(config, kind, strict = true)
        let elapsed = (getMonoTime() - started).inMilliseconds
        check sim.done
        check sim.reason == "complete"
        check sim.turnsPlayed == config.bars
        var bars = 0
        for event in sim.events:
          if event.kind == evBar:
            inc bars
            check not event.edit
            check event.steps.len == Steps
            check event.say.len == 0
        check bars == config.bars * Seats
        check sim.events.len == 5 * config.bars + 3
        ## Every voice is covered: seat->voice is a seeded permutation, so
        ## four seats of one baseline is one baseline in all four voices.
        var voices = 0
        for voice in 0 ..< Voices:
          if sim.onsetsOf(voice, sim.turnsPlayed) > 0:
            inc voices
        check voices == Voices
        check elapsed < 2000

  test "arpeggio sits in its quality band and pedal is the weaker filler":
    var total = 0.0
    var lowest = 200.0
    var highest = -1.0
    var pedalLower = 0
    let trials = 200
    for seed in 0 ..< trials:
      let arpeggio = playScripted(fixture(seed), skArpeggio)
      let pedal = playScripted(fixture(seed), skPedal)
      let a = arpeggio.pieceScore(arpeggio.grid, arpeggio.turnsPlayed).piece
      let p = pedal.pieceScore(pedal.grid, pedal.turnsPlayed).piece
      total += a
      lowest = min(lowest, a)
      highest = max(highest, a)
      if p < a:
        inc pedalLower
    echo "arpeggio piece score over ", trials, " seeds: mean ",
      formatFloat(total / trials.float, ffDecimal, 2), ", min ",
      formatFloat(lowest, ffDecimal, 2), ", max ",
      formatFloat(highest, ffDecimal, 2),
      "; pedal scored lower on ", pedalLower, "/", trials
    check lowest >= 40.0
    check highest <= 92.0
    check pedalLower.float / trials.float >= 0.9

  test "decideAll fails non-scripted seats with no credentials instead of falling back":
    for key in ["COWORLD_LLM_ENDPOINT", "COWORLD_LLM_MODEL",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "AWS_BEARER_TOKEN_BEDROCK",
        "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_URI"]:
      delEnv(key)
    let config = fixture(3, bars = 6)
    let client = newLlmClient(config)
    check client.disabled
    var sim = initSim(config)
    let seats = sim.pendingSeats()
    let decisions = client.decideAll(sim, seats,
      @["write a countermelody", "", "", ""],
      @[skNone, skArpeggio, skPedal, skNone])
    check decisions.len == Seats
    check decisions[0].failed
    check not decisions[0].scripted
    check "unavailable" in decisions[0].failureReason
    check decisions[3].failed
    check not decisions[3].scripted
    ## Explicitly scripted seats still run their selected baseline.
    check not decisions[1].failed
    check decisions[1].scripted
    check decisions[1].steps == scriptedAction(sim, 1, skArpeggio).steps
    check not decisions[2].failed
    check decisions[2].scripted
    check decisions[2].steps == scriptedAction(sim, 2, skPedal).steps

  test "a seat that fails both attempts returns explicit failure without fallback":
    ## Point an enabled client at a closed port so both attempts fail on
    ## transport, fast and without a network. No scripted fallback is synthesized.
    for key in ["COWORLD_LLM_ENDPOINT", "COWORLD_LLM_MODEL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_URI"]:
      delEnv(key)
    putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://127.0.0.1:9")
    putEnv("AWS_BEARER_TOKEN_BEDROCK", "unusable-in-tests")
    var config = fixture(7, bars = 6)
    config.llmTimeoutSeconds = 5
    let client = newLlmClient(config)
    delEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
    delEnv("AWS_BEARER_TOKEN_BEDROCK")
    check not client.disabled
    var sim = initSim(config)
    let seats = sim.pendingSeats()
    let decisions = client.decideAll(sim, seats, @["", "", "", ""],
      @[skNone, skNone, skNone, skNone])
    check decisions.len == Seats
    for index, seat in seats:
      check decisions[index].failed
      check not decisions[index].scripted
      check decisions[index].failureReason.len > 0
      check decisions[index].rawReply.len > 0

  test "model replies accept canonical JSON and outer markdown fence, and reject aliases/shims":
    let array16 = "[0,-1,-1,4,-1,-1,2,-1,0,-1,-1,4,-1,-1,-1,-1]"
    let want = @[0, -1, -1, 4, -1, -1, 2, -1, 0, -1, -1, 4, -1, -1, -1, -1]
    check parseDecision(parseJson("""{"target": 2, "steps": """ & array16 &
      "}"), 3).steps == want
    ## A parsed reply is the model's own bar, never a baseline.
    check not parseDecision(parseJson("""{"target": 2, "steps": """ &
      array16 & "}"), 3).scripted
    check parseDecision(parseJson("""{"target": 2, "steps": """ & array16 &
      "}"), 5).target == 2
    ## Standard outer Markdown JSON fence is supported.
    let fencedJson = "```json\n{\"target\": 0, \"steps\": " & array16 & "}\n```"
    check parseDecision(extractJsonObject(fencedJson), 0).steps == want
    let fencedPlain = "```\n{\"target\": 0, \"steps\": " & array16 & "}\n```"
    check parseDecision(extractJsonObject(fencedPlain), 0).steps == want
    ## Leading or trailing prose outside the JSON object is rejected.
    expect ChorusError:
      discard extractJsonObject("Sure! {\"target\": 0, \"steps\": " &
        array16 & "} — I kept the tonic on 0 and 8.")
    expect ChorusError:
      discard extractJsonObject("{\"target\": 0, \"steps\": " & array16 & "} trailing")
    ## String-form steps, rest spellings, float rounding, string numbers, and null are rejected.
    expect ChorusError:
      discard parseDecision(parseJson(
        """{"target": 0, "steps": "0 . r 4 R rest - -1 0 . . 4 . . . ."}"""), 0)
    expect ChorusError:
      discard parseDecision(parseJson(
        """{"target": 0, "steps": "0,-1,-1,4,-1,-1,2,-1,0,-1,-1,4,-1,-1,-1,-1"}"""), 0)
    expect ChorusError:
      discard parseDecision(parseJson(
        """{"target": 0, "steps": [0.0,-1,-1,3.6,-1,-1,"2",-1,0,-1,-1,4,-1,-1,-1,-1]}"""), 0)
    expect ChorusError:
      discard parseDecision(parseJson(
        """{"target": 0, "steps": [0,null,-1,4,-1,-1,2,-1,0,-1,-1,4,-1,-1,-1,-1]}"""), 0)
    ## Missing, null, float, or string target is rejected (no implicit default).
    expect ChorusError:
      discard parseDecision(parseJson("""{"steps": """ & array16 & "}"), 5)
    expect ChorusError:
      discard parseDecision(parseJson("""{"target": null, "steps": """ & array16 & "}"), 5)
    expect ChorusError:
      discard parseDecision(parseJson("""{"target": 2.0, "steps": """ & array16 & "}"), 5)
    expect ChorusError:
      discard parseDecision(parseJson("""{"target": "2", "steps": """ & array16 & "}"), 5)
    ## Out-of-bounds lengths, tokens, targets, and missing steps are rejected.
    expect ChorusError:
      discard parseDecision(parseJson(
        """{"target": 0, "steps": [0,-1,-1,4,-1,-1,2,-1,0,-1,-1,4,-1,-1,-1]}"""), 0)
    expect ChorusError:
      discard parseDecision(parseJson(
        """{"target": 0, "steps": [0,-1,-1,4,-1,-1,2,-1,0,-1,-1,4,-1,-1,-1,-1,-1]}"""), 0)
    expect ChorusError:
      discard parseDecision(parseJson(
        """{"target": 0, "steps": [14,-1,-1,4,-1,-1,2,-1,0,-1,-1,4,-1,-1,-1,-1]}"""), 0)
    expect ChorusError:
      discard parseDecision(parseJson(
        """{"target": 0, "steps": [-2,-1,-1,4,-1,-1,2,-1,0,-1,-1,4,-1,-1,-1,-1]}"""), 0)
    expect ChorusError:
      discard parseDecision(parseJson("""{"target": 4, "steps": """ &
        array16 & "}"), 3)
    expect ChorusError:
      discard parseDecision(parseJson("""{"target": 0}"""), 0)
    ## Overlong say/notes and multiline say are rejected instead of silently truncated.
    var longSay = ""
    for index in 0 ..< MaxSayLen + 1:
      longSay.add("音")
    var longNotes = ""
    for index in 0 ..< MaxNotesLen + 1:
      longNotes.add("音")
    expect ChorusError:
      discard parseDecision(parseJson("{\"target\": 0, \"steps\": " & array16 &
        ", \"say\": \"" & longSay & "\"}"), 0)
    expect ChorusError:
      discard parseDecision(parseJson("{\"target\": 0, \"steps\": " & array16 &
        ", \"notes\": \"" & longNotes & "\"}"), 0)
    expect ChorusError:
      discard parseDecision(parseJson("{\"target\": 0, \"steps\": " & array16 &
        ", \"say\": \"line1\\nline2\"}"), 0)
    ## Exact-limit UTF-8 say/notes are accepted.
    var exactSay = ""
    for index in 0 ..< MaxSayLen:
      exactSay.add("音")
    var exactNotes = ""
    for index in 0 ..< MaxNotesLen:
      exactNotes.add("音")
    let accepted = parseDecision(parseJson("{\"target\": 0, \"steps\": " & array16 &
      ", \"say\": \"" & exactSay & "\", \"notes\": \"" & exactNotes & "\"}"), 0)
    check accepted.say.runeLen == MaxSayLen
    check accepted.notes.runeLen == MaxNotesLen
    ## Canonical PLAYER_SCRIPTED values only; legacy aliases rejected.
    check parseScriptKind("arpeggio") == skArpeggio
    check parseScriptKind("pedal") == skPedal
    check parseScriptKind("") == skNone
    check parseScriptKind("none") == skNone
    for alias in ["1", "true", "yes", "drone", "unknown"]:
      expect ChorusError:
        discard parseScriptKind(alias)
