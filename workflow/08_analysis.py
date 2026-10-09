"""Step 08 - polygons, attributes and the analyses shown on the dashboard.

  * landslide polygons on the 10 m pixel edges, with probability, terrain, land cover, village tract,
    Sentinel-2 optical support and a confidence class
  * per-village-tract totals; exposure of MIMU villages, schools, health facilities, OSM buildings and roads
  * frequency ratio of conditioning factors (slope, elevation, aspect, land cover, distance to stream)
  * Sentinel-1 backscatter time series over the drone-mapped landslides vs. stable ground (why it works)
  * rainfall: Open-Meteo daily series over the township and the 1991-2026 record at Ka Det Nge Htein

Outputs: outputs/landslides.gpkg (+ .geojson), outputs/analysis.json, outputs/*.csv
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from scipy import ndimage
from shapely.geometry import shape

import config as C

WC_NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse",
            70: "Snow", 80: "Water", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss / lichen"}
VEC = C.DATA / "vectors.gpkg"
HIGH_PROB = 0.96             # mean SAR probability that counts as high confidence without an optical view


def r1(p, b=1):
    with rasterio.open(p) as s:
        return s.read(b)


def feat(name, tag="ASC070_event"):
    with rasterio.open(C.DATA / "features" / f"{tag}.tif") as s:
        return s.read(list(s.descriptions).index(name) + 1)


def polygons(det, prob, slope, dem, wc, d_vv, d_vh, ndvi_pre, s2):
    lab, n = ndimage.label(det, structure=np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    st = pd.DataFrame({"obj": idx})
    st["pixels"] = ndimage.sum(np.ones_like(lab), lab, idx).astype(int)
    st["area_ha"] = st.pixels * C.RES ** 2 / 1e4
    st["prob_mean"] = ndimage.mean(prob.astype("float32"), lab, idx) / 100
    st["prob_max"] = ndimage.maximum(prob, lab, idx) / 100
    st["slope_mean"] = ndimage.mean(slope, lab, idx)
    st["slope_max"] = ndimage.maximum(slope, lab, idx)
    st["elev_min"] = ndimage.minimum(dem, lab, idx)
    st["elev_max"] = ndimage.maximum(dem, lab, idx)
    st["d_vv_mean"] = ndimage.mean(np.nan_to_num(d_vv), lab, idx)
    st["d_vh_mean"] = ndimage.mean(np.nan_to_num(d_vh), lab, idx)
    # dominant land cover
    wcs = sorted(WC_NAMES)
    counts = np.stack([ndimage.sum((wc == k).astype("uint8"), lab, idx) for k in wcs], axis=1)
    st["landcover"] = [WC_NAMES[wcs[i]] for i in counts.argmax(1)]
    # Sentinel-2 optical support (independent of the SAR model): clear-sky share and NDVI change
    ndvi_post, clear = s2
    dn = np.where(clear & np.isfinite(ndvi_pre) & np.isfinite(ndvi_post), ndvi_post - ndvi_pre, np.nan)
    okc = np.isfinite(dn)
    st["s2_clear_frac"] = ndimage.sum(okc.astype("uint8"), lab, idx) / st.pixels
    st["s2_dndvi"] = np.where(st.s2_clear_frac > 0, ndimage.sum(np.nan_to_num(dn), lab, idx) /
                              np.maximum(ndimage.sum(okc.astype("uint8"), lab, idx), 1), np.nan)
    st["optical"] = np.select([st.s2_clear_frac < 0.5, st.s2_dndvi <= -0.15], ["cloud", "supported"], "not seen")
    # every object already passed the SAR threshold; confidence adds the independent optical evidence:
    # high = Sentinel-2 saw an NDVI drop, or cloud-hidden with a very strong SAR probability;
    # medium = cloud-hidden; low = Sentinel-2 saw clear ground there and NO vegetation loss (conflicting evidence)
    st["confidence"] = np.select(
        [st.optical == "supported", (st.optical == "cloud") & (st.prob_mean >= HIGH_PROB), st.optical == "cloud"],
        ["high", "high", "medium"], "low")
    geoms = {}
    for g, v in shapes(lab.astype("int32"), mask=lab > 0, connectivity=8, transform=C.grid_transform()):
        geoms.setdefault(int(v), []).append(shape(g))
    from shapely.ops import unary_union
    st["geometry"] = [unary_union(geoms[i]) for i in st.obj]
    g = gpd.GeoDataFrame(st, crs=C.CRS)
    return g, lab


def freq_ratio(det, valid, layer, bins, labels):
    cls = np.digitize(layer, bins) - 1 if bins is not None else layer
    rows = []
    tot_ls, tot = det[valid].sum(), valid.sum()
    keys = range(len(labels)) if bins is not None else labels
    for i, k in enumerate(keys):
        m = valid & (cls == k)
        a, l = m.sum(), (det & m).sum()
        if a == 0:
            continue
        rows.append({"class": labels[i] if bins is not None else WC_NAMES.get(k, str(k)), "area_km2": round(a * 1e-4, 2),
                     "landslide_ha": round(l * 1e-2, 2), "density_pct": round(100 * l / a, 3),
                     "fr": round((l / tot_ls) / (a / tot), 2) if tot_ls else 0})
    return rows


def s1_timeseries():
    inv = C.read_json(C.DATA / "s1_inventory.json")
    lab = r1(C.GRID_DIR / "labels.tif")
    ls, st = lab == 2, lab == 1
    rows = []
    for a in inv["tracks"]["ASC070"]["pre"] + inv["tracks"]["ASC070"]["post"]:
        with rasterio.open(C.S1_DIR / "ASC070" / f"{a['date'].replace('-', '')}.tif") as s:
            x = s.read().astype("float32")
        x[x == 0] = np.nan
        db = x / C.DB_SCALE - C.DB_OFFSET
        row = {"date": a["date"], "platform": a["platform"]}
        for pol, i in (("vv", 0), ("vh", 1)):
            for nm, m in (("landslide", ls), ("stable", st)):
                v = db[i][m]
                # mean in linear power, reported in dB
                row[f"{pol}_{nm}"] = round(float(10 * np.log10(np.nanmean(10 ** (v / 10)))), 2)
        rows.append(row)
    return rows


def rainfall():
    d = C.RAW / "rain"
    js = json.loads((d / "openmeteo_grid_daily.json").read_text())
    js = js if isinstance(js, list) else [js]
    t = js[0]["daily"]["time"]
    arr = np.array([[v if v is not None else np.nan for v in j["daily"]["precipitation_sum"]] for j in js])
    daily = [{"date": t[i], "mean_mm": round(float(np.nanmean(arr[:, i])), 1), "max_mm": round(float(np.nanmax(arr[:, i])), 1)}
             for i in range(len(t))]
    lt = json.loads((d / "openmeteo_kadet_1991_2026.json").read_text())["daily"]
    s = pd.Series(lt["precipitation_sum"], index=pd.to_datetime(lt["time"])).astype(float)
    r3 = s.rolling(3).sum()
    r5 = s.rolling(5).sum()
    ann = pd.DataFrame({"max1": s.groupby(s.index.year).max(), "max3": r3.groupby(r3.index.year).max(),
                        "max5": r5.groupby(r5.index.year).max(), "total": s.groupby(s.index.year).sum()})
    ev3 = float(r3["2026-09-25":"2026-10-01"].max())
    ev5 = float(r5["2026-09-25":"2026-10-03"].max())
    hist = ann.loc[:2025]
    hr = json.loads((d / "openmeteo_kadet_hourly.json").read_text())["hourly"]
    return {"daily": daily, "points": len(js),
            "annual_max": [{"year": int(y), "max1": round(r.max1, 1), "max3": round(r.max3, 1), "max5": round(r.max5, 1)} for y, r in ann.iterrows()],
            "event_max3_mm": round(ev3, 1), "event_max5_mm": round(ev5, 1),
            "rank3": int((hist.max3 >= ev3).sum()) + 1, "rank5": int((hist.max5 >= ev5).sum()) + 1, "years": int(len(hist)),
            "record3_before": {"year": int(hist.max3.idxmax()), "mm": round(float(hist.max3.max()), 1)},
            "hourly": [{"t": a, "mm": b} for a, b in zip(hr["time"], hr["precipitation"])],
            "source": "Open-Meteo historical weather API (ERA5 / ECMWF IFS reanalysis-model blend), daily sums, Asia/Yangon"}


def main():
    det = r1(C.OUT / "landslides.tif").astype(bool)
    prob = r1(C.OUT / "landslide_probability.tif")
    valid = prob != 255
    prob = np.where(valid, prob, 0)
    slope, dem, wc = r1(C.GRID_DIR / "slope.tif"), r1(C.GRID_DIR / "dem.tif"), r1(C.GRID_DIR / "worldcover.tif")
    aspect, dstream = r1(C.GRID_DIR / "aspect.tif"), r1(C.GRID_DIR / "dist_stream.tif")
    d_vv, d_vh = feat("d_vv"), feat("d_vh")
    ndvi_pre = r1(C.S2_DIR / "ndvi_pre_2026JanMar.tif")
    with rasterio.open(C.S2_DIR / "post_event.tif") as s:
        ndvi_post, s2ok = s.read(1), s.read(2) > 0
    # drop pixels near clouds (SCL misses cloud edges and thin shadow): 3 px erosion of the clear mask
    s2ok = ndimage.binary_erosion(s2ok, iterations=3)

    g, lab = polygons(det, prob, slope, dem, wc, d_vv, d_vh, ndvi_pre, (ndvi_post, s2ok))
    vt = gpd.read_file(VEC, layer="vtracts")
    cent = g.copy(); cent["geometry"] = g.representative_point()
    j = gpd.sjoin(cent[["obj", "geometry"]], vt[["VT", "VT_PCODE", "geometry"]], how="left", predicate="within")
    g = g.merge(j[["obj", "VT", "VT_PCODE"]].drop_duplicates("obj"), on="obj", how="left")
    g["id"] = [f"S1-{i:04d}" for i in range(1, len(g) + 1)]
    g = g.sort_values("area_ha", ascending=False).reset_index(drop=True)
    for c in ("area_ha", "prob_mean", "prob_max", "slope_mean", "slope_max", "d_vv_mean", "d_vh_mean", "s2_clear_frac", "s2_dndvi"):
        g[c] = g[c].astype(float).round(3)
    g["elev_min"] = g.elev_min.round(0); g["elev_max"] = g.elev_max.round(0)
    cols = ["id", "area_ha", "pixels", "confidence", "prob_mean", "prob_max", "slope_mean", "slope_max", "elev_min", "elev_max",
            "landcover", "d_vv_mean", "d_vh_mean", "s2_clear_frac", "s2_dndvi", "optical", "VT", "VT_PCODE", "geometry"]
    g = g[cols]
    out_g = C.OUT / "landslides.gpkg"
    if out_g.exists():
        out_g.unlink()
    g.to_file(out_g, layer="landslides_s1", driver="GPKG")
    g.to_crs(4326).to_file(C.OUT / "landslides.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)
    g.drop(columns="geometry").to_csv(C.OUT / "landslides.csv", index=False)
    print(f"{len(g)} landslide polygons, {g.area_ha.sum():.1f} ha; confidence {g.confidence.value_counts().to_dict()}; optical {g.optical.value_counts().to_dict()}")

    # optical background rate: same NDVI test on random undetected slopes, for comparison
    rng = np.random.default_rng(0)
    bg = valid & ~det & (slope >= 12) & s2ok & np.isfinite(ndvi_pre)
    rr, cc = np.nonzero(bg)
    pick = rng.choice(len(rr), size=min(200000, len(rr)), replace=False)
    dn_bg = ndvi_post[rr[pick], cc[pick]] - ndvi_pre[rr[pick], cc[pick]]
    dpx = det & s2ok & np.isfinite(ndvi_pre)
    dn_det = ndvi_post[dpx] - ndvi_pre[dpx]
    optical = {"det_px_clear": int(dpx.sum()), "det_px_total": int(det.sum()),
               "det_share_drop015": round(float(np.mean(dn_det <= -0.15)), 3) if dn_det.size else None,
               "bg_share_drop015": round(float(np.mean(dn_bg <= -0.15)), 3),
               "det_median_dndvi": round(float(np.median(dn_det)), 3) if dn_det.size else None,
               "bg_median_dndvi": round(float(np.median(dn_bg)), 3),
               "hist_bins": np.round(np.arange(-0.8, 0.41, 0.05), 2).tolist(),
               "hist_det": np.histogram(dn_det, bins=np.arange(-0.8, 0.41, 0.05))[0].tolist(),
               "hist_bg": np.histogram(dn_bg, bins=np.arange(-0.8, 0.41, 0.05))[0].tolist()}

    # village tracts
    vtr = rasterize([(geom, i + 1) for i, geom in enumerate(vt.geometry)], (C.HEIGHT, C.WIDTH), transform=C.grid_transform(), dtype="int16")
    rows = []
    for i, r in vt.iterrows():
        m = vtr == i + 1
        land = (m & valid).sum()
        a = (m & det).sum()
        sub = g[g.VT_PCODE == r.VT_PCODE]
        rows.append({"VT": r.VT, "VT_MMR": r.VT_MMR, "VT_PCODE": r.VT_PCODE, "land_km2": round(land * 1e-4, 2),
                     "landslides": int(len(sub)), "high_conf": int((sub.confidence == "high").sum()),
                     "landslide_ha": round(a * 1e-2, 2), "density_pct": round(100 * a / land, 3) if land else 0,
                     "per_km2": round(len(sub) / (land * 1e-4), 2) if land else 0})
    vt_stats = sorted(rows, key=lambda x: -x["landslide_ha"])
    pd.DataFrame(vt_stats).to_csv(C.OUT / "village_tract_summary.csv", index=False)

    # exposure
    gl = g[["id", "confidence", "geometry"]]
    b = gpd.read_file(VEC, layer="buildings")
    b_in = gpd.sjoin(b, gl, how="inner", predicate="within")
    b_near = gpd.sjoin(b, gl.assign(geometry=gl.buffer(50)), how="inner", predicate="within")
    roads = gpd.read_file(VEC, layer="roads")
    rcut = gpd.overlay(roads[["source", "cls", "name", "geometry"]], gl, how="intersection", keep_geom_type=True)
    rcut["len_m"] = rcut.length
    vill = gpd.read_file(VEC, layer="villages")
    union = gl.geometry.union_all()
    vrows = []
    for _, v in vill.iterrows():
        dist = float(v.geometry.distance(union)) if len(gl) else np.nan
        buf = v.geometry.buffer(1000)
        a1 = float(gl.geometry.intersection(buf).area.sum()) / 1e4
        nb = int(b.geometry.within(buf).sum())
        vrows.append({"village": v.VILLAGE, "village_mmr": v.VLG_MMR, "pcode": v.VLG_PCODE, "vt": v.VT,
                      "dist_m": round(dist), "ls_ha_1km": round(a1, 2), "buildings_1km": nb,
                      "lon": round(v.geometry.centroid.x, 1), "lat": round(v.geometry.centroid.y, 1)})
    vdf = pd.DataFrame(vrows)
    pts = gpd.GeoSeries(gpd.points_from_xy(vdf.lon, vdf.lat), crs=C.CRS).to_crs(4326)
    vdf["lon"], vdf["lat"] = pts.x.round(5), pts.y.round(5)
    vdf = vdf.sort_values(["ls_ha_1km", "dist_m"], ascending=[False, True])
    vdf.to_csv(C.OUT / "village_exposure.csv", index=False)
    fac = {}
    for lay in ("schools", "health"):
        f = gpd.read_file(VEC, layer=lay)
        dd = f.geometry.apply(lambda p: p.distance(union)) if len(gl) else pd.Series(dtype=float)
        fac[lay] = {"total": int(len(f)), "within_250m": int((dd <= 250).sum()), "within_500m": int((dd <= 500).sum())}
    roads_by = rcut.groupby("source").len_m.sum().round(0).to_dict() if len(rcut) else {}
    exposure = {"buildings_total": int(len(b)), "buildings_inside": int(b_in.osm_id.nunique()),
                "buildings_within_50m": int(b_near.osm_id.nunique()),
                "road_m_inside": {k: float(v) for k, v in roads_by.items()},
                "road_crossings": int(rcut.id.nunique()) if len(rcut) else 0,
                "villages_within_500m": int((vdf.dist_m <= 500).sum()), "villages_total": int(len(vdf)), "facilities": fac}

    # conditioning factors
    fr = {
        "slope": freq_ratio(det, valid, slope, [0, 5, 10, 15, 20, 25, 30, 35, 90], ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", ">35"]),
        "elevation": freq_ratio(det, valid, dem, [-50, 25, 50, 100, 150, 200, 300, 400, 1000], ["<25", "25-50", "50-100", "100-150", "150-200", "200-300", "300-400", ">400"]),
        "aspect": freq_ratio(det, valid & (slope >= 5), (aspect + 22.5) % 360, [0, 45, 90, 135, 180, 225, 270, 315, 360], ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        "landcover": freq_ratio(det, valid, wc, None, [k for k in WC_NAMES if (wc[valid] == k).any()]),
        "dist_stream": freq_ratio(det, valid, dstream, [0, 50, 100, 200, 400, 800, 1e9], ["<50 m", "50-100 m", "100-200 m", "200-400 m", "400-800 m", ">800 m"]),
    }

    an = {"n_landslides": int(len(g)), "area_ha": round(float(g.area_ha.sum()), 2),
          "confidence": g.confidence.value_counts().to_dict(), "optical_class": g.optical.value_counts().to_dict(),
          "size_hist": np.histogram(g.area_ha, bins=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 1000])[0].tolist(),
          "size_bins": ["0.05-0.1", "0.1-0.2", "0.2-0.5", "0.5-1", "1-2", "2-5", "5-10", ">10"],
          "valid_land_km2": round(float(valid.sum()) * 1e-4, 1), "village_tracts": vt_stats,
          "villages": vdf.head(40).to_dict("records"), "exposure": exposure, "factors": fr, "optical": optical,
          "s1_timeseries": s1_timeseries(), "rain": rainfall()}
    C.write_json(C.OUT / "analysis.json", an)
    print(json.dumps({k: an[k] for k in ("n_landslides", "area_ha", "confidence", "optical_class")}))
    print("exposure:", json.dumps(exposure))
    print("optical:", {k: v for k, v in optical.items() if not k.startswith("hist")})


if __name__ == "__main__":
    main()
