"""Wildfire detector website (prototype).

    python app/server.py        then open http://127.0.0.1:8780

Click anywhere on the map: the server finds the latest clear Sentinel-2 photo (last 15 days) of that
10 km square, runs the models and shows where fire is. The NASA FIRMS key stays on the server.
"""
import json
import sys
import traceback
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import Detector  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
PAGES = {"/": "index.html", "/about": "about.html"}
PORT = 8780
detector: Detector


class Handler(BaseHTTPRequestHandler):
    def send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj: dict, code: int = 200) -> None:
        self.send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in PAGES:
                self.send((STATIC / PAGES[u.path]).read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/analyze":
                lat, lon = float(q["lat"]), float(q["lon"])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    return self.json(dict(ok=False, message="That point is off the map."), 400)
                when = date.fromisoformat(q["date"]) if q.get("date") else None
                self.json(detector.analyze(lon, lat, when))
            elif u.path == "/api/search":                          # place name -> coordinates (OpenStreetMap)
                r = detector.http.get("https://nominatim.openstreetmap.org/search",
                                      params=dict(q=q.get("q", "")[:200], format="json", limit=1),
                                      headers={"User-Agent": "wildfire-detector-prototype/0.1"}, timeout=15)
                hits = r.json() if r.ok else []
                if not hits:
                    return self.json(dict(ok=False, message=f"No place found for “{q.get('q', '')}”."))
                h = hits[0]
                self.json(dict(ok=True, lat=float(h["lat"]), lon=float(h["lon"]), name=h.get("display_name", "")))
            elif u.path == "/api/nasa":
                self.json(detector.nasa(q.get("id", "")))
            else:
                self.send(b"not found", "text/plain", 404)
        except Exception as e:                                     # show a readable message, log details
            traceback.print_exc()
            self.json(dict(ok=False, message=f"Something went wrong: {type(e).__name__}. Try again."), 500)

    def do_HEAD(self) -> None:
        self.send_response(200 if urlparse(self.path).path in PAGES else 404)
        self.end_headers()

    def log_message(self, fmt, *args) -> None:
        if args and "/api/" in str(args[0]):
            sys.stderr.write("%s\n" % (fmt % args))


def main() -> None:
    global detector
    print("loading models...", flush=True)
    detector = Detector()
    print(f"open http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
