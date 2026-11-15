"""Eye check: you label the tiles where the model and the automatic labels disagree.

    python src/review.py            then open http://127.0.0.1:8770

Shows one tile at a time: true colour, infrared (fire glows orange), and where it sits in the photo.
Keys: F = fire, N = not fire, S = not sure, B = back. Every answer is saved immediately to
data/review_labels.json, which prepare.py uses to override the automatic labels.

The queue (data/review_queue.json) is built by make_review_queue.py from the trained model's scores.
Blind method comparison: python src/review.py --queue data/method_review_queue.json --labels data/method_review_labels.json
"""
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "data" / "review_queue.json"
LABELS = ROOT / "data" / "review_labels.json"
CHIPS = ROOT / "data" / "chips"
TILE, CTX = 32, 48           # show the tile with 48 px (~1 km) of surroundings each side

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Fire eye check</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font:16px system-ui,sans-serif;margin:0;background:#111;color:#eee;text-align:center}
img{max-width:100%;height:auto;margin:8px 0}
button{font-size:20px;padding:14px 28px;margin:6px;border:0;border-radius:8px;cursor:pointer}
.f{background:#e8541e;color:#fff}.n{background:#2e7d32;color:#fff}.s{background:#555;color:#fff}.b{background:#333;color:#ccc}
#info{color:#aaa;margin:8px}
</style></head><body>
<div id="info">loading...</div>
<img id="img" alt="tile to review">
<div>
<button class="f" onclick="answer('fire')">Fire (F)</button>
<button class="n" onclick="answer('not')">Not fire (N)</button>
<button class="s" onclick="answer('unsure')">Not sure (S)</button>
<button class="b" onclick="go(i-1)">Back (B)</button>
</div>
<p id="hint" style="color:#888;font-size:14px">Orange/yellow glow in the infrared view = something very hot.
The white box is the tile being judged.</p>
<script>
let q=[], a={}, i=0;
async function load(){const s=await (await fetch('/state')).json(); q=s.queue; a=s.answers;
  i=q.findIndex((_,k)=>!(k in a)); if(i<0) i=q.length; go(i);}
function go(k){i=Math.max(0,Math.min(k,q.length)); const done=Object.keys(a).length;
  if(i>=q.length){document.getElementById('info').textContent=`All ${q.length} done. Thank you! You can close this page.`;
    document.getElementById('img').style.display='none'; return;}
  document.getElementById('img').style.display=''; document.getElementById('img').src='/img/'+i;
  const prev=a[i]?` (your answer: ${a[i]})`:'';
  document.getElementById('info').textContent=`Tile ${i+1} of ${q.length}, ${done} answered${prev}`;}
async function answer(v){await fetch('/answer',{method:'POST',body:JSON.stringify({i:i,label:v})}); a[i]=v; go(i+1);}
document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();
  if(k==='f')answer('fire'); else if(k==='n')answer('not'); else if(k==='s')answer('unsure'); else if(k==='b')go(i-1);});
load();
</script></body></html>"""


def load_answers() -> dict:
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def render(item: dict) -> bytes:
    with np.load(CHIPS / f"{item['chip']}.npz") as z:
        rgb, swir = z["rgb"], np.nan_to_num(z["swir"].astype(np.float32))
    r0, c0 = item["tr"] * TILE, item["tc"] * TILE
    a, b = max(r0 - CTX, 0), max(c0 - CTX, 0)
    e, f = min(r0 + TILE + CTX, 512), min(c0 + TILE + CTX, 512)
    false_colour = np.clip(swir.transpose(1, 2, 0) / 0.5, 0, 1) ** 0.8
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="#111")
    ax[0].imshow(rgb[a:e, b:f], interpolation="nearest")
    ax[0].set_title("True colour (zoomed)", color="#eee")
    ax[1].imshow(false_colour[a:e, b:f], interpolation="nearest")
    ax[1].set_title("Infrared (zoomed): fire glows orange", color="#eee")
    for k in (0, 1):
        ax[k].add_patch(Rectangle((c0 - b - 0.5, r0 - a - 0.5), TILE, TILE, fill=False, ec="white", lw=1.5))
    ax[2].imshow(false_colour)
    ax[2].add_patch(Rectangle((c0, r0), TILE, TILE, fill=False, ec="white", lw=1.5))
    ax[2].set_title("Whole photo (10 km), infrared", color="#eee")
    for x in ax:
        x.set_xticks([])
        x.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    queue: list = []

    def send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/state":
            answers = {int(k): v["label"] for k, v in load_answers().items()}
            self.send(json.dumps(dict(queue=self.queue, answers=answers)).encode(), "application/json")
        elif self.path.startswith("/img/"):
            k = int(self.path.split("/")[-1])
            self.send(render(self.queue[k]), "image/png")
        else:
            self.send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        if self.path != "/answer":
            self.send(b"not found", "text/plain", 404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        k, label = int(body["i"]), body["label"]
        if label not in ("fire", "not", "unsure") or not 0 <= k < len(self.queue):
            self.send(b"bad request", "text/plain", 400)
            return
        answers = load_answers()
        answers[str(k)] = dict(self.queue[k], label=label)
        LABELS.write_text(json.dumps(answers, indent=1))
        self.send(b"ok", "text/plain")

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    import argparse
    global QUEUE, LABELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(QUEUE), help="tiles to show (default: the eye-check queue)")
    ap.add_argument("--labels", default=str(LABELS), help="where answers are saved")
    args = ap.parse_args()
    QUEUE, LABELS = Path(args.queue), Path(args.labels)
    Handler.queue = json.loads(QUEUE.read_text())
    print(f"{len(Handler.queue)} tiles to review, {len(load_answers())} already answered")
    print("open http://127.0.0.1:8770")
    ThreadingHTTPServer(("127.0.0.1", 8770), Handler).serve_forever()


if __name__ == "__main__":
    main()
