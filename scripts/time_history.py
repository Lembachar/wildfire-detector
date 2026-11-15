"""How many extra seconds would the before+now model add to one website check?
Times, per place: the normal photo download, then the earlier-photo search and the download of
3 earlier photos (cloud map + B12 + B11 on the same 10 km square), exactly what the model needs."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app"))
import detector as D

PLACES = [("Burkina Faso", 13.5620, -1.4714, date(2026, 11, 14)), ("Botswana", -21.8696, 24.1582, date(2026, 11, 12)),
          ("Paris", 48.8566, 2.3522, None), ("Honduras", 15.0893, -84.9123, None),
          ("California", 38.58, -121.49, None), ("Australia NT", -12.46, 130.84, None)]

d = D.Detector()
rows = []
for name, lat, lon, when in PLACES:
    when = when or date.today()
    t0 = time.time()
    p, err = d.photo(lon, lat, when)
    t_photo = time.time() - t0
    if err:
        print(f"{name:14s} no photo: {err}"); continue
    t1 = time.time()
    items = d.earlier(p["item"], lon, lat, p["transform"])[:3]
    t_search = time.time() - t1
    t2 = time.time()
    with ThreadPoolExecutor(9) as pool:
        futs = [pool.submit(d.read, it["assets"]["scl"], win=p["win"], scl=True) for it in items]
        futs += [pool.submit(d.read, it["assets"][k], win=p["win"], props=it["properties"])
                 for it in items for k in ("swir22", "swir16")]
        ok = sum(1 for f in futs if f.exception() is None)
    t_read = time.time() - t2
    rows.append((t_photo, t_search, t_read))
    print(f"{name:14s} photo {t_photo:5.1f}s | earlier: search {t_search:4.1f}s + {len(items)} photos {t_read:5.1f}s"
          f" ({ok}/{len(futs)} layers ok) | extra {t_search + t_read:5.1f}s", flush=True)

n = len(rows)
print(f"\naverage over {n}: photo {sum(r[0] for r in rows)/n:.1f}s, extra {sum(r[1]+r[2] for r in rows)/n:.1f}s")
