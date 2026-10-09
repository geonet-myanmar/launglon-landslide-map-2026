"""Step 09 - build the interactive dashboard (docs/index.html + docs/overlays/*.webp), served by GitHub Pages.

Raster overlays are reprojected to Web Mercator at 10.3 m (= 10 m on the ground at 13.9 N), so the map shows
the products at their native resolution. Vectors and statistics are embedded in the page, so index.html opens
straight from disk (no web server) and also publishes as-is.
"""
import json

import geopandas as gpd
import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy import ndimage

import config as C

OV = C.DASH / "overlays"
OV.mkdir(parents=True, exist_ok=True)
TEMPLATE = C.ROOT / "workflow" / "dashboard_template.html"
LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"
VEC = C.DATA / "vectors.gpkg"


def r1(p, b=1):
    with rasterio.open(p) as s:
        return s.read(b)


def to_3857(arr, resampling, nodata):
    src_tr = C.grid_transform()
    dst_res = C.RES / np.cos(np.radians(13.875))
    tr, w, h = calculate_default_transform(C.CRS, "EPSG:3857", C.WIDTH, C.HEIGHT, C.X0, C.Y0, C.X1, C.Y1, resolution=dst_res)
    out = np.full((h, w), nodata, dtype=arr.dtype)
    reproject(arr, out, src_transform=src_tr, src_crs=C.CRS, dst_transform=tr, dst_crs="EPSG:3857",
              resampling=resampling, src_nodata=nodata, dst_nodata=nodata)
    return out, tr, w, h


BOUNDS = None


def save_rgba(name, layers, resampling=Resampling.nearest, quality=80, lossless=False):
    """layers: (r, g, b, a) uint8 arrays on the analysis grid."""
    global BOUNDS
    bands = []
    for a in layers:
        o, tr, w, h = to_3857(a.astype("uint8"), resampling, 0)
        bands.append(o)
    from pyproj import Transformer
    t = Transformer.from_crs(3857, 4326, always_xy=True)
    lon0, lat1 = t.transform(tr.c, tr.f)
    lon1, lat0 = t.transform(tr.c + tr.a * w, tr.f + tr.e * h)
    BOUNDS = [[lat0, lon0], [lat1, lon1]]
    im = Image.fromarray(np.dstack(bands), "RGBA")
    im.save(OV / name, "WEBP", quality=quality, lossless=lossless, method=6)
    print(f"  {name}: {w}x{h}px  {(OV / name).stat().st_size / 1e6:.1f} MB")


def ramp(vals, stops):
    """Piecewise-linear colour ramp; stops = [(v, (r,g,b)), ...]."""
    v = np.asarray(vals, "float32")
    xs = [s[0] for s in stops]
    return [np.interp(v, xs, [s[1][i] for s in stops]).astype("uint8") for i in range(3)]


def overlays():
    land = C.aoi_mask(150)
    dem = r1(C.GRID_DIR / "dem.tif")
    # hillshade (sun from NW, 45 deg), with land/sea tint so it works as a basemap
    gy, gx = np.gradient(ndimage.gaussian_filter(dem, 0.8), C.RES)
    slope = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    hs = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - asp), 0, 1)
    wc = r1(C.GRID_DIR / "worldcover.tif")
    sea = (wc == 80) & (dem <= 2)
    base = (60 + 180 * hs).astype("uint8")
    elev_t = np.clip(dem / 400, 0, 1)
    r = np.where(sea, 205, base * (0.93 + 0.05 * elev_t)).astype("uint8")
    g = np.where(sea, 220, base * (0.97 - 0.02 * elev_t)).astype("uint8")
    b = np.where(sea, 230, base * (0.90 - 0.04 * elev_t)).astype("uint8")
    save_rgba("hillshade.webp", (r, g, b, np.full_like(r, 255)), Resampling.bilinear, quality=72)

    with rasterio.open(C.S2_DIR / "post_event.tif") as s:
        rgb = s.read([4, 5, 6]); ok = s.read(2) > 0
    rgb = np.clip(np.nan_to_num(rgb) * 3.2, 0, 1) ** (1 / 1.25) * 255
    a = np.where(ok & C.aoi_mask(2000), 255, 0)
    save_rgba("s2_post.webp", (rgb[0], rgb[1], rgb[2], a), Resampling.bilinear, quality=78)

    with rasterio.open(C.DATA / "features" / "ASC070_event.tif") as s:
        desc = list(s.descriptions)
        pre_vh = s.read(desc.index("pre_vh") + 1); post_vh = s.read(desc.index("post_vh") + 1)
        dvh = s.read(desc.index("d_vh_m3") + 1); dvv = s.read(desc.index("d_vv_m3") + 1)
    def sc(x, lo=-20, hi=-6):
        return (np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo), 0, 1) * 255)
    a = np.where(land & np.isfinite(pre_vh), 255, 0)
    save_rgba("s1_rgb.webp", (sc(pre_vh), sc(post_vh), sc(post_vh), a), Resampling.bilinear, quality=72)

    dd = np.nan_to_num((dvh + dvv) / 2)
    col = ramp(dd, [(-6, (150, 30, 30)), (-3, (214, 72, 60)), (-1, (240, 160, 140)), (0, (240, 240, 236)),
                    (1, (150, 190, 230)), (3, (60, 120, 200)), (6, (25, 60, 130))])
    alpha = np.where(land, np.clip((np.abs(dd) - 0.75) / 2.25, 0, 1) * 235, 0)
    save_rgba("s1_change.webp", (*col, alpha), Resampling.bilinear, quality=75)

    prob = r1(C.OUT / "landslide_probability.tif").astype("float32")
    prob[prob == 255] = 0
    col = ramp(prob, [(30, (253, 219, 138)), (50, (247, 151, 72)), (70, (219, 72, 46)), (100, (128, 15, 38))])
    alpha = np.where((prob >= 30) & land, 90 + (prob - 30) / 70 * 160, 0)
    save_rgba("probability.webp", (*col, alpha), Resampling.nearest, quality=85)

    det = r1(C.OUT / "landslides.tif").astype("float32")
    dens = ndimage.gaussian_filter(det, 30) * 100        # % of ground in landslides, ~300 m kernel
    m = max(np.percentile(dens[dens > 0.05], 99.5), 1) if (dens > 0.05).any() else 1
    col = ramp(dens / m, [(0, (255, 237, 160)), (0.3, (254, 178, 76)), (0.6, (240, 59, 32)), (1, (128, 0, 38))])
    alpha = np.where(land & (dens > 0.05), np.clip(dens / m * 2.5, 0.25, 0.85) * 255, 0)
    save_rgba("density.webp", (*col, alpha), Resampling.bilinear, quality=80)
    if (C.OUT / "placebo_detections.tif").exists():
        p = r1(C.OUT / "placebo_detections.tif")
        z = np.zeros_like(p)
        save_rgba("placebo.webp", (z + 120, z + 60, z + 200, np.where(p > 0, 255, 0)), Resampling.nearest, lossless=True)
    return {"densityMaxPct": round(float(m), 2)}


def gj(gdf, cols, simplify=None, precision=5):
    g = gdf.to_crs(4326)
    if simplify:
        g["geometry"] = g.geometry.simplify(simplify, preserve_topology=True)
    g = g[cols + ["geometry"]]
    return json.loads(g.to_json(drop_id=True, to_wgs84=False).replace("NaN", "null")) if len(g) else {"type": "FeatureCollection", "features": []}


def utm_simplify(g, tol):
    """Simplify in metres; tol 0 only drops collinear vertices (pixel-edge outlines keep their exact shape)."""
    g = g.to_crs(C.CRS).copy()
    g["geometry"] = g.geometry.simplify(tol, preserve_topology=True)
    return g


def rnd(o, p=5):
    if isinstance(o, float):
        return round(o, p)
    if isinstance(o, list):
        return [rnd(x, p) for x in o]
    if isinstance(o, dict):
        return {k: rnd(v, p) for k, v in o.items()}
    return o


def main():
    print("overlays:")
    extra = overlays()
    an = C.read_json(C.OUT / "analysis.json")
    val = C.read_json(C.OUT / "validation.json")
    plc = C.read_json(C.OUT / "placebo.json") if (C.OUT / "placebo.json").exists() else None
    inv = C.read_json(C.DATA / "s1_inventory.json")
    ls = gpd.read_file(C.OUT / "landslides.gpkg", layer="landslides_s1")
    vt = gpd.read_file(VEC, layer="vtracts")
    vill = gpd.read_file(VEC, layer="villages")
    roads = gpd.read_file(VEC, layer="roads")
    roads = roads[(roads.source == "MIMU") | roads.cls.isin(["trunk", "primary", "secondary", "tertiary", "unclassified", "track"])]
    vec = {
        "landslides": gj(utm_simplify(ls, 0.0), ["id", "area_ha", "confidence", "prob_mean", "slope_mean", "slope_max", "elev_min", "elev_max",
                              "landcover", "d_vv_mean", "d_vh_mean", "optical", "s2_dndvi", "VT"]),
        # drone outlines are digitised at cm detail; 1 m is ten times finer than the map and keeps the page light
        "drone": gj(utm_simplify(gpd.read_file(C.OUT / "drone_reference.geojson"), 1.0), ["id", "name", "type", "area_m2", "site"]),
        "survey": gj(gpd.read_file(C.OUT / "drone_survey_extents.geojson"), ["site"], simplify=0.00005),
        "vtracts": gj(vt, ["VT", "VT_MMR", "VT_PCODE"], simplify=0.0002),
        "township": gj(C.aoi(), ["TS", "TS_MMR"], simplify=0.0001),
        "villages": gj(vill, ["VILLAGE", "VLG_MMR", "VLG_PCODE", "VT"]),
        "roads": gj(roads, ["source", "cls", "name"], simplify=0.00005),
        "schools": gj(gpd.read_file(VEC, layer="schools"), ["name"]),
        "health": gj(gpd.read_file(VEC, layer="health"), ["name", "level"]),
    }
    vec = rnd(vec, 6)
    data = {"bounds": BOUNDS, "extra": extra, "analysis": rnd(an, 4), "validation": rnd(val, 4), "placebo": plc,
            "s1": {k: {"pre": [a["date"] for a in v["pre"]], "post": [a["date"] for a in v["post"]],
                       "platforms": sorted({a["platform"] for a in v["pre"] + v["post"]})} for k, v in inv["tracks"].items()},
            "vec": vec}
    css = requests.get(LEAFLET_CSS, timeout=60).text
    html = TEMPLATE.read_text(encoding="utf8")
    html = html.replace("/*__LEAFLET_CSS__*/", css)
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    # artifact.html: page body only (the publishing host adds the document skeleton);
    # index.html: the same page as a full standards-mode document, to open from disk
    (C.DATA / "artifact_body.html").write_text(html, encoding="utf8")
    (C.DASH / ".nojekyll").write_text("", encoding="utf8")     # serve files as-is, no Jekyll build
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           '<style>html{color-scheme:light dark}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n'
           '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
    (C.DASH / "index.html").write_text(doc, encoding="utf8")
    print(f"docs/index.html {(C.DASH / 'index.html').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
