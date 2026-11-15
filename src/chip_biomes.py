"""Biome of every photo (at its centre), from the RESOLVE Ecoregions 2017 map (Dinerstein et al. 2017), so the
Hu et al. 2021 method can use the criteria made for that biome. Writes data/biomes.json.

    data/ecoregions/Ecoregions2017.shp  from https://storage.googleapis.com/teow2016/Ecoregions2017.zip
    python src/chip_biomes.py
"""
import json
from collections import Counter

import shapefile
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from train_v2 import DATA, ROOT


def main() -> None:
    sf = shapefile.Reader(str(ROOT / "data" / "ecoregions" / "Ecoregions2017.shp"), encoding="latin-1")
    fields = [f[0] for f in sf.fields[1:]]
    geoms, names = [], []
    for sr in sf.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        geoms.append(shape(sr.shape.__geo_interface__))
        names.append(rec["BIOME_NAME"])
    tree = STRtree(geoms)
    info = json.loads((DATA / "chips.json").read_text())
    out = {}
    for name, m in zip(info["chips"], info["metas"]):
        p = Point(m["lon"], m["lat"])
        hits = [i for i in tree.query(p) if geoms[i].contains(p)]
        if not hits:                                       # on a coast / in a lake: nearest ecoregion
            hits = [int(tree.nearest(p))]
        out[name] = names[hits[0]]
    (ROOT / "data" / "biomes.json").write_text(json.dumps(out, indent=0))
    print(Counter(out.values()).most_common())


if __name__ == "__main__":
    main()
