#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser


ROOT = Path(__file__).resolve().parents[1] / "output/research/cad-line-ontology-authority-pack-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local CAD line-ontology authority review")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    server = None
    selected_port = args.port
    for port in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer((args.host, port), handler)
            selected_port = port
            break
        except OSError as exc:
            if exc.errno != 48:
                raise
    if server is None:
        raise RuntimeError(f"no free review port in {args.port}..{args.port + 9}")
    url = f"http://{args.host}:{selected_port}/review_app.html"
    print(f"Authority review: {url}", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
