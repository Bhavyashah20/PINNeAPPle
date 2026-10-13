"""Build the compact star and constellation catalogues shipped in pinneapple_physics/astro/data.

Sources (open data, see pinneapple_physics/astro/data/README.md):
  * Yale Bright Star Catalogue, 5th revised ed. (Hoffleit & Warren 1991, NASA ADC / CDS V/50), as the JSON
    conversion bsc5-all.json of github.com/brettonw/YaleBrightStarCatalog (MIT).
  * d3-celestial data (c) 2015 Olaf Frohn, BSD-3-Clause: constellations.json, constellations.lines.json,
    starnames.json (npm package d3-celestial).

    python scripts/astro/build_catalogs.py BSC5_ALL_JSON D3_CELESTIAL_DATA_DIR
"""
import gzip
import json
import os
import sys

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "pinneapple_physics", "astro", "data")


def num(s, default=None):
    try:
        return float(str(s).replace("+", ""))
    except (TypeError, ValueError):
        return default


def main(bsc_path, d3_dir):
    bsc = json.load(open(bsc_path, encoding="utf-8"))
    names = json.load(open(os.path.join(d3_dir, "starnames.json"), encoding="utf-8"))
    by_hd = {}
    for v in names.values():
        hd = str(v.get("hd", "")).replace("HD", "").strip()
        if hd and v.get("name"):
            by_hd[hd] = v["name"]
    cols = {k: [] for k in ("hr", "name", "desig", "con", "ra", "dec", "pmra", "pmdec", "v", "bv", "sp", "plx",
                            "plx_dyn", "rv")}
    for s in bsc:
        if "RAh" not in s or num(s.get("Vmag")) is None:
            continue                                     # the 14 retired entries (novae, non-stellar objects)
        ra = (num(s["RAh"]) + num(s["RAm"]) / 60 + num(s["RAs"]) / 3600) * 15
        dec = num(s["DEd"]) + num(s["DEm"]) / 60 + num(s["DEs"]) / 3600
        if s.get("DE-") == "-":
            dec = -dec
        bayer = s.get("Bayer", "")
        flam = s.get("Flamsteed", "")
        con = s.get("Constellation", "")
        desig = f"{bayer} {con}".strip() if bayer else (f"{flam} {con}".strip() if flam else f"HR {s['HR']}")
        name = by_hd.get(str(s.get("HD", "")).strip()) or s.get("Common", "")
        dyn = any(n.get("Category") == "Dynamical parallaxes" for n in s.get("Notes", []))
        cols["hr"].append(int(s["HR"]))
        cols["name"].append(name)
        cols["desig"].append(desig)
        cols["con"].append(con)
        cols["ra"].append(round(ra, 5))
        cols["dec"].append(round(dec, 5))
        cols["pmra"].append(round(num(s.get("pmRA"), 0.0) * 1000, 1))      # mas/yr (mu_alpha cos delta)
        cols["pmdec"].append(round(num(s.get("pmDE"), 0.0) * 1000, 1))
        cols["v"].append(num(s["Vmag"]))
        cols["bv"].append(num(s.get("B-V")))
        cols["sp"].append(s.get("SpType", ""))
        cols["plx"].append(None if num(s.get("Parallax")) is None else round(num(s["Parallax"]) * 1000, 1))
        cols["plx_dyn"].append(dyn)
        cols["rv"].append(num(s.get("RadVel")))
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, "bsc5.json.gz"), "wt", encoding="utf-8") as fh:
        json.dump(cols, fh, separators=(",", ":"), ensure_ascii=False)

    lines = json.load(open(os.path.join(d3_dir, "constellations.lines.json"), encoding="utf-8"))
    meta = {f["id"]: f for f in json.load(open(os.path.join(d3_dir, "constellations.json"),
                                               encoding="utf-8"))["features"]}
    cons = []
    for f in lines["features"]:
        m = meta.get(f["id"], {})
        segs = [[[round(lon % 360, 4), round(lat, 4)] for lon, lat in seg] for seg in f["geometry"]["coordinates"]]
        c = m.get("geometry", {}).get("coordinates", [0, 0])
        cons.append({"id": f["id"], "name": m.get("properties", {}).get("name", f["id"]),
                     "label": [round(c[0] % 360, 3), round(c[1], 3)], "rank": int(f["properties"].get("rank", 3)),
                     "lines": segs})
    with gzip.open(os.path.join(OUT, "constellations.json.gz"), "wt", encoding="utf-8") as fh:
        json.dump(cons, fh, separators=(",", ":"), ensure_ascii=False)
    milky_way(os.path.join(d3_dir, "mw.json"))
    print(len(cols["hr"]), "stars,", len(cons), "constellations")


def milky_way(path, width=2048, height=1024):
    """Rasterise d3-celestial's five Milky Way brightness contours into an equirectangular grey JPEG in equatorial
    coordinates (x = RA / 360 * width, y = (90 - Dec) / 180 * height).

    The contours are drawn in galactic coordinates, where the two rings that circle the whole sky (the band's
    upper and lower edges) are closed through the south galactic pole; every level is filled with the even-odd
    rule (as d3 does), the levels are summed, blurred, and resampled to RA/Dec."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    G = np.array([[-0.0548755604, -0.8734370902, -0.4838350155], [0.4941094279, -0.4448296300, 0.7469822445],
                  [-0.8676661490, -0.1980763734, 0.4559837762]])           # ICRS -> galactic
    feats = json.load(open(path, encoding="utf-8"))["features"]
    acc = np.zeros((height, width), np.float32)
    for f in feats:
        mask = np.zeros((height, width), bool)
        for poly in f["geometry"]["coordinates"]:
            for ring in poly:
                a = np.radians(np.array(ring, float))
                v = np.stack([np.cos(a[:, 1]) * np.cos(a[:, 0]), np.cos(a[:, 1]) * np.sin(a[:, 0]),
                              np.sin(a[:, 1])], 1) @ G.T
                lon = np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 0])))
                lat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
                circles = abs(lon[-1] - lon[0]) > 180
                if circles:                                   # close through the south galactic pole
                    lon = np.r_[lon, lon[-1], lon[0]]
                    lat = np.r_[lat, -90.0, -90.0]
                layer = Image.new("1", (3 * width, height), 0)
                draw = ImageDraw.Draw(layer)
                lon = lon - (lon[0] - lon[0] % 360.0)            # start in [0, 360)
                for shift in (-360.0, 0.0, 360.0):
                    x = (lon + shift) / 360.0 * width + width
                    y = (90.0 - lat) / 180.0 * (height - 1)
                    draw.polygon(list(zip(x, y, strict=True)), fill=1)
                m = np.asarray(layer, bool)
                mask ^= m[:, width:2 * width]
        acc += mask
    gal = Image.fromarray(np.clip(acc / len(feats) * 255, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(5))
    gal = np.asarray(gal, np.float32)
    # resample to equatorial: pixel centre (RA, Dec) -> (l, b) -> galactic raster
    ra = (np.arange(width) + 0.5) / width * 2 * np.pi
    dec = np.radians(90 - (np.arange(height) + 0.5) / height * 180)
    RA, DEC = np.meshgrid(ra, dec)
    v = np.stack([np.cos(DEC) * np.cos(RA), np.cos(DEC) * np.sin(RA), np.sin(DEC)], -1) @ G.T
    glon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360
    b = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    xi = (glon / 360 * width).astype(int) % width
    yi = np.clip(((90 - b) / 180 * (height - 1)).round().astype(int), 0, height - 1)
    Image.fromarray(gal[yi, xi].astype("uint8")).save(os.path.join(OUT, "milkyway.jpg"), quality=90)


if __name__ == "__main__":
    main(*sys.argv[1:3])
