from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.blackjack_service import BlackjackAppSession, MODEL_PRESETS


STATIC_DIR = Path(__file__).resolve().parent / "static"


class AppState:
    def __init__(self, *, model_key: str, seed: int | None, device: str) -> None:
        self.model_key = model_key
        self.seed = seed
        self.device = device
        self.session: BlackjackAppSession | None = None

    def get_session(self) -> BlackjackAppSession:
        if self.session is None:
            self.session = BlackjackAppSession(model_key=self.model_key, seed=self.seed, device=self.device)
        return self.session


APP_STATE: AppState | None = None


def _read_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON payload must be an object")
    return data


class BlackjackRequestHandler(SimpleHTTPRequestHandler):
    server_version = "BlackjackRLApp/0.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, exc: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": str(exc)}, status=status)

    def _session(self) -> BlackjackAppSession:
        if APP_STATE is None:
            raise RuntimeError("App state has not been initialized")
        return APP_STATE.get_session()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            try:
                self._send_json(self._session().state())
            except Exception as exc:
                self._send_error_json(exc, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/models":
            self._send_json({"models": MODEL_PRESETS})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            data = _read_json(self)
            session = self._session()
            if parsed.path == "/api/action":
                action = str(data.get("action") or "")
                if not action:
                    raise ValueError("Missing action")
                self._send_json(session.step(action))
                return
            if parsed.path == "/api/play-suggestion":
                self._send_json(session.play_suggestion())
                return
            if parsed.path == "/api/autoplay":
                max_steps = int(data.get("max_steps", query.get("max_steps", [20])[0]))
                self._send_json(session.autoplay(max_steps=max_steps))
                return
            if parsed.path == "/api/new-round":
                self._send_json(session.new_round())
                return
            if parsed.path == "/api/new-table":
                model_key = data.get("model_key")
                seed_value = data.get("seed")
                seed = int(seed_value) if seed_value not in (None, "") else None
                self._send_json(session.new_table(model_key=model_key, seed=seed))
                return
            self._send_error_json(ValueError(f"Unknown endpoint: {parsed.path}"), HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error_json(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Blackjack RL casino app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="05A")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    global APP_STATE
    APP_STATE = AppState(model_key=args.model, seed=args.seed, device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), BlackjackRequestHandler)
    print(f"Blackjack RL app: http://{args.host}:{args.port}")
    print(f"Default model: {args.model}")
    print("The first page load initializes the checkpoint and can take a few seconds.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Blackjack RL app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
