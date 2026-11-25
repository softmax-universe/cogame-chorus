"""Exercise complete native server/player games and preserve all test artifacts."""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Provider(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fail_turn = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        body = json.loads(raw)
        user = body["messages"][0]["content"]
        turn = int(re.search(r"Turn (\d+) of", user)[1])
        slot = self.headers["X-Coworld-Player-Slot"]
        assert "PRIVATE_OPERATOR_" + slot in user
        for other in range(4):
            if str(other) != slot:
                assert "PRIVATE_OPERATOR_" + str(other) not in user
        retry = "Your previous reply was invalid" in user
        if self.fail_turn is not None and turn == self.fail_turn:
            text = "invalid fixture reply"
        elif turn == 0 and not retry:
            text = "invalid fixture reply"
        elif turn == 2 and not retry:
            # Overlong notes are rejected on attempt 0 (no silent truncation).
            text = json.dumps(
                {
                    "target": turn,
                    "steps": [0] + [-1] * 15,
                    "say": "",
                    "notes": "x" * 900,
                }
            )
        else:
            payload = json.dumps(
                {
                    "target": turn,
                    "steps": [0] + [-1] * 15,
                    "say": "",
                    "notes": "PRIVATE_CAPTURE_" + slot,
                }
            )
            # Exercise standard outer Markdown JSON fence on turn 0 retry.
            text = f"```json\n{payload}\n```" if turn == 0 and retry else payload
        response = json.dumps(
            {
                "model": body["model"],
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": text}],
                "usage": None if turn == 3 else {"input_tokens": 100, "output_tokens": 20},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("X-Softmax-Llm-Call-Id", str(uuid.uuid5(uuid.NAMESPACE_URL, raw.decode())))
        self.end_headers()
        self.wfile.write(response)


def run(game, player, output, variant, capture, fail_turn=None):
    suffix = f"fail{fail_turn}" if fail_turn is not None else "ok"
    root = output / f"{variant['id']}-capture{capture}-{suffix}"
    root.mkdir(mode=0o700)
    config = dict(variant["game_config"])
    seats = config["num_agents"]
    config.update(
        seed=42,
        turnDelayMs=0,
        minTurnSpacingMs=0,
        player_connect_timeout_seconds=5,
        episodeTimeoutSeconds=120,
        tokens=[f"fixture-{i}" for i in range(seats)],
    )
    (root / "config.json").write_text(json.dumps(config))
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    env = dict(os.environ)
    for key in (
        "COWORLD_LLM_ENDPOINT",
        "COWORLD_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY_URI",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        env.pop(key, None)
    episode = f"native-capture-fixture-{variant['id']}-{suffix}"
    env.update(
        COGAME_HOST="127.0.0.1",
        COGAME_PORT=str(port),
        COGAME_CONFIG_URI=(root / "config.json").as_uri(),
        COGAME_RESULTS_URI=(root / "results.json").as_uri(),
        COGAME_SAVE_REPLAY_URI=(root / "replay.json").as_uri(),
        COGAME_PLAYER_FAILURE_URI=(root / "player_failure.json").as_uri(),
        COGAME_SAVE_TRAJECTORY_URI=(root / "trajectory.jsonl").as_uri(),
        COWORLD_EPISODE_ID=episode,
        COWORLD_GAME_VERSION="identity-fixture",
        COWORLD_SOURCE_REVISION="a" * 40,
    )
    if not capture:
        env.pop("COGAME_SAVE_TRAJECTORY_URI")

    class ScopedProvider(Provider):
        pass

    ScopedProvider.fail_turn = fail_turn
    provider = ThreadingHTTPServer(("127.0.0.1", 0), ScopedProvider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    env["COWORLD_LLM_ENDPOINT"] = f"http://127.0.0.1:{provider.server_port}"
    env["COWORLD_LLM_MODEL"] = "native-capture-fixture"
    children = []
    streams = []
    try:
        log = (root / "game.log").open("wb")
        streams.append(log)
        server = subprocess.Popen([str(game)], env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(server)
        for _ in range(100):
            assert server.poll() is None, "game exited before readiness; see retained game.log"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                    assert response.status == 200
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.05)
        else:
            raise TimeoutError("native game did not become ready")
        for seat in range(seats):
            log = (root / f"player-{seat}.log").open("wb")
            streams.append(log)
            player_env = dict(env, PLAYER_PROMPT=f"PRIVATE_OPERATOR_{seat}", COWORLD_TIMEOUT_SECONDS="120")
            player_env["COWORLD_PLAYER_WS_URL"] = f"ws://127.0.0.1:{port}/player?slot={seat}&token=fixture-{seat}"
            children.append(subprocess.Popen([str(player)], env=player_env, stdout=log, stderr=subprocess.STDOUT))
        assert server.wait(timeout=120) == 0
        for child in children[1:]:
            assert child.wait(timeout=5) == 0
        results = json.loads((root / "results.json").read_text())
        replay = json.loads((root / "replay.json").read_text())
        if fail_turn is not None:
            assert results["reason"] == "action_failure"
            assert results["bars"] == fail_turn
            failure = json.loads((root / "player_failure.json").read_text())
            assert failure["reason"] == "action_failure"
            assert failure["turn"] == fail_turn
            assert len(failure["failures"]) == seats
            for entry in failure["failures"]:
                assert entry["raw_reply"] == "invalid fixture reply"
            if capture:
                trace = root / "trajectory.jsonl"
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                assert trace.stat().st_mode & 0o077 == 0
                assert rows[-1]["status"] == "failed"
                assert rows[-1]["outcome"] == results
                failed_rows = [r for r in rows[:-1] if r["decision_index"] // seats == fail_turn]
                assert len(failed_rows) == seats
                for row in failed_rows:
                    assert row["action_status"] == "rejected"
                    assert row["selected_attempt_id"] is None
                    assert row["executed_action"] is None
                    assert row["fallback_origin"] is None
                    assert len(row["attempts"]) == 2
                    for attempt in row["attempts"]:
                        assert not attempt["accepted"]
                        assert attempt["response"] == "invalid fixture reply"
        else:
            assert results["reason"] == "complete"
            assert results["bars"] == config["bars"]
            assert not (root / "player_failure.json").exists()
            if capture:
                trace = root / "trajectory.jsonl"
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                assert trace.stat().st_mode & 0o077 == 0
                assert len(rows) == config["bars"] * seats + 1
                assert rows[-1]["status"] == "completed"
                assert rows[-1]["outcome"] == results
                for row in rows[:-1]:
                    turn = row["decision_index"] // seats
                    attempts = row["attempts"]
                    if turn in (0, 2):
                        assert len(attempts) == 2 and not attempts[0]["accepted"]
                        assert attempts[1]["accepted"]
                        assert row["selected_attempt_id"] == attempts[1]["attempt_id"]
                        assert row["action_status"] == "accepted"
                    else:
                        assert row["action_status"] == "accepted"
                    if turn == 3:
                        assert attempts[0]["input_tokens"] is None
                        assert attempts[0]["output_tokens"] is None
                    for attempt in attempts:
                        assert attempt["prompt"][0]["content"] == attempt["request"]["system"]
                        assert attempt["platform_call_id"] is not None
                        assert attempt["behavior_logprobs"] is None
        return results, replay
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        for stream in streams:
            stream.close()


def main():
    game, player, destination = map(lambda value: Path(value).resolve(), sys.argv[1:])
    destination.mkdir(mode=0o700, exist_ok=False)
    manifest = json.loads((Path(__file__).resolve().parents[1] / "coworld_manifest_template.json").read_text())
    for variant in manifest["variants"]:
        captured = run(game, player, destination, variant, True)
        ordinary = run(game, player, destination, variant, False)
        assert captured == ordinary, "capture changed the native outcome or replay"
    failed_captured = run(game, player, destination, manifest["variants"][0], True, fail_turn=1)
    failed_ordinary = run(game, player, destination, manifest["variants"][0], False, fail_turn=1)
    assert failed_captured == failed_ordinary, "capture changed failed episode outcome or replay"
    print(
        "native games: capture parity, exact requests, retries, outer fence, "
        "explicit action_failure without fallback, privacy and missing usage pass"
    )


if __name__ == "__main__":
    main()
