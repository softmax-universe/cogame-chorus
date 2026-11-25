#!/usr/bin/env python3
"""Deterministic local Anthropic Messages fixture for Chorus CI and offline certification.

Serves POST /v1/messages with canonical Chorus JSON bars matching the turn in
the user prompt, without making any external/paid LLM calls.
"""

import argparse
import json
import re
import signal
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_handler(stats_path: Path | None):
    state = {"requests": 0, "slots": []}

    def write_stats():
        if stats_path is not None:
            stats_path.write_text(json.dumps(state))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/healthz":
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path != "/v1/messages":
                self.send_response(404)
                self.end_headers()
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.loads(raw)
            user = body["messages"][0]["content"]
            match = re.search(r"Turn (\d+) of", user)
            turn = int(match.group(1)) if match else 0
            slot = self.headers.get("X-Coworld-Player-Slot", "")
            # Canonical 16-step motif varying cleanly across turns.
            if turn % 2 == 0:
                steps = [0, -1, -1, -1, 4, -1, -1, -1, 2, -1, -1, -1, 4, -1, -1, -1]
            else:
                steps = [0, -1, 2, -1, 4, -1, -1, -1, 7, -1, -1, -1, 4, -1, 2, -1]
            action = {
                "target": turn,
                "steps": steps,
                "notes": f"fixture turn {turn}",
            }
            payload = json.dumps(
                {
                    "id": f"msg_fixture_{turn}_{slot}",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model", "ci-deterministic-fixture"),
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": json.dumps(action)}],
                    "usage": {"input_tokens": 120, "output_tokens": 32},
                }
            ).encode("utf-8")
            state["requests"] += 1
            if slot not in state["slots"]:
                state["slots"].append(slot)
            write_stats()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header(
                "X-Softmax-Llm-Call-Id",
                str(uuid.uuid5(uuid.NAMESPACE_URL, f"{turn}:{slot}:{state['requests']}")),
            )
            self.end_headers()
            self.wfile.write(payload)

    write_stats()
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path, default=None)
    parser.add_argument("--stats-file", type=Path, default=None)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.stats_file))
    actual_port = server.server_port
    if args.port_file is not None:
        args.port_file.write_text(str(actual_port))
    print(f"chorus llm fixture listening on {args.host}:{actual_port}", flush=True)

    def handle_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
