"""Demo pictures for the README: real checks through the running website (python app/server.py).
Each picture: true colour | infrared (B12, B11, B8A) with the detected fires outlined in yellow."""
import base64, io, json, sys, urllib.request
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
PLACES = [("Burkina Faso", 13.5620, -1.4714, "2026-11-14"), ("Botswana", -21.8696, 24.1582, "2026-11-12"),
          ("Northern Territory, Australia", -12.46, 130.84, None)]


def img(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # local server: no proxy
    for name, lat, lon, day in PLACES:
        q = f"lat={lat}&lon={lon}" + (f"&date={day}" if day else "")
        r = json.load(opener.open(f"http://127.0.0.1:8780/api/analyze?{q}", timeout=300))
        if not r.get("ok"):
            print(name, "no photo:", r.get("message")); continue
        tc, ir, mask = img(r["true_colour"]), img(r["infrared"]).convert("RGBA"), img(r["fire_mask"])
        ir.alpha_composite(mask)
        fires, ha = len(r["detections"]), sum(d["hectares"] for d in r["detections"])
        fig, ax = plt.subplots(1, 2, figsize=(10, 5.4), facecolor="#0b0f17")
        for a, im, t in [(ax[0], tc, "True colour"), (ax[1], ir, "Infrared (B12, B11, B8A) + detected fire outline")]:
            a.imshow(im); a.set_axis_off(); a.set_title(t, color="#cbd5e1", fontsize=11)
        fig.suptitle(f"{name} · Sentinel-2, {r['taken'][:10]} · {fires} fire{'s' if fires != 1 else ''}, "
                     f"≈ {ha:.1f} ha burning · 10 km square", color="white", fontsize=13)
        fig.tight_layout()
        f = OUT / f"demo_{name.split(',')[0].lower().replace(' ', '_')}.png"
        fig.savefig(f, dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)
        print(f"{name}: {fires} fires, {ha:.1f} ha, compared with {r.get('compared')} -> {f.name}", flush=True)


if __name__ == "__main__":
    main()
