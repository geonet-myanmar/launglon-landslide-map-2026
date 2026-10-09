"""Step 07 - landslide detection, validated against the drone surveys with leave-one-site-out testing.

Three detectors run through the same masks, minimum mapping unit and scoring:
  M0  classic bi-temporal log-ratio: |post - last pre pass| >= 3 dB in VV or VH (what a 1-pre/1-post workflow gives)
  M1  multi-temporal anomaly (unsupervised): backscatter decrease vs. the 10-pass pre-event mean in VV and VH,
      hysteresis-grown on the 3x3 mean decrease (seeds >= 3 dB with |z| >= 2, growth >= 1.5 dB)
  M2  gradient-boosted classifier on the multi-temporal SAR features + terrain + land cover + pre-event NDVI.
      Positives and event-time negatives come from the drone surveys. "Temporal negatives" come from two
      no-event pairs (27 Aug and 8 Sep, each against the passes before it): every pixel there, including the
      survey ground that slid later, is known to hold no new landslide, so the model learns ordinary monsoon
      change across the whole township, and learns that the change, not the terrain, marks a landslide.
      Scored by leave-one-site-out (LOSO): each site is mapped by a model that never saw it.

False alarms are measured, not assumed: the frozen model is run on no-event pairs. The 20 Sep pair calibrates
the threshold (best LOSO F1 among thresholds whose placebo area stays <= MAX_FA_RATIO of the event area);
the 15 Aug pair is never used for any choice and gives the independent false-alarm estimate.

The 10 m grid is kept end to end: features, probabilities, masks and polygons are all on the native grid.

Outputs (outputs/):  landslide_probability.tif (uint8 %, 255 = no data), landslides.tif (uint8 0/1),
                     placebo_detections.tif, validation.json, placebo.json, model_M2.joblib
"""
import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import config as C

FEAT = C.DATA / "features"
TRACK = "ASC070"
EVENT_TAG = f"{TRACK}_event"
NEG_TAGS = [f"{TRACK}_placebo_20260827", f"{TRACK}_placebo_20260908"]   # temporal negatives (training)
CAL_TAG = f"{TRACK}_placebo_20260920"                                  # threshold calibration
TEST_TAG = f"{TRACK}_placebo_20260815"                                 # independent false-alarm test
SAR = ["pre_vv", "pre_vh", "std_vv", "std_vh", "post_vv", "post_vh", "d_vv", "d_vh", "z_vv", "z_vh",
       "dl_vv", "dl_vh", "d_cr", "d_vv_m3", "d_vh_m3", "d_vv_m7", "d_vh_m7", "d_vv_m51", "d_vh_m51"]
AUX = ["lia", "slope", "tpi", "curv", "ndvi_pre", "wc", "dist_stream"]
FEATURES = SAR + AUX
MIN_SOURCE_SLOPE = 12.0     # an object must reach this slope somewhere (a source area); runout may be flatter
WATER_WC = (80,)            # WorldCover permanent water
MAX_FA_RATIO = 0.20         # placebo area / event area allowed at the chosen threshold
N_TEMPORAL_NEG = 200_000    # random township pixels per no-event pair (plus every survey pixel)
THRESHOLDS = np.round(np.arange(0.40, 0.96, 0.05), 2)
MMUS = (10, 20)


# ------------------------------------------------------------------ data access
def aux_sources():
    return {"lia": (C.S1_DIR / TRACK / "geometry.tif", 1), "slope": (C.GRID_DIR / "slope.tif", 1),
            "tpi": (C.GRID_DIR / "tpi.tif", 1), "curv": (C.GRID_DIR / "curv.tif", 1),
            "ndvi_pre": (C.S2_DIR / "ndvi_pre_2026JanMar.tif", 1), "wc": (C.GRID_DIR / "worldcover.tif", 1),
            "dist_stream": (C.GRID_DIR / "dist_stream.tif", 1), "shadow": (C.S1_DIR / TRACK / "geometry.tif", 2),
            "labels": (C.GRID_DIR / "labels.tif", 1), "sites": (C.GRID_DIR / "sites.tif", 1)}


def read_stack(tag, win, names):
    out = {}
    with rasterio.open(FEAT / f"{tag}.tif") as s:
        desc = list(s.descriptions)
        for n in names:
            if n in desc:
                out[n] = s.read(desc.index(n) + 1, window=win)
    for n in names:
        if n not in out:
            p, b = aux_sources()[n]
            with rasterio.open(p) as s:
                out[n] = s.read(b, window=win).astype("float32")
    return out


def valid_mask(d):
    ok = np.isfinite(d["d_vv"]) & np.isfinite(d["d_vh"]) & (d["shadow"] == 0)
    return ok & ~np.isin(d["wc"], WATER_WC)


def X_of(d, sel=None):
    return np.stack([d[f] if sel is None else d[f][sel] for f in FEATURES], axis=-1).astype("float32")


# ------------------------------------------------------------------ detectors
def m0(d):
    return (np.abs(d["dl_vv"]) >= 3) | (np.abs(d["dl_vh"]) >= 3)


def m1(d):
    """Backscatter DECREASE in both polarisations vs. the multi-temporal reference (canopy loss, smooth wet debris)."""
    s = -(np.nan_to_num(d["d_vv_m3"]) + np.nan_to_num(d["d_vh_m3"])) / 2
    z = -np.minimum(np.nan_to_num(d["z_vv"]), np.nan_to_num(d["z_vh"]))
    seed, grow = (s >= 3) & (z >= 2), s >= 1.5
    lab, _ = ndimage.label(grow, structure=np.ones((3, 3)))
    keep = np.unique(lab[seed & (lab > 0)])
    return np.isin(lab, keep[keep > 0])


def postprocess(binary, d, mmu_px):
    """Masks, minimum mapping unit, and a source-slope test on 8-connected objects (10 m grid untouched)."""
    b = binary & valid_mask(d)
    lab, n = ndimage.label(b, structure=np.ones((3, 3)))
    if n == 0:
        return b
    idx = np.arange(1, n + 1)
    size = ndimage.sum(np.ones_like(lab, dtype="uint8"), lab, idx)
    smax = ndimage.maximum(d["slope"], lab, idx)
    keep = np.zeros(n + 1, bool)
    keep[idx[(size >= mmu_px) & (smax >= MIN_SOURCE_SLOPE)]] = True
    return keep[lab]


def model():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=80,
                                          l2_regularization=1.0, class_weight="balanced",
                                          categorical_features=[FEATURES.index("wc")], random_state=0)


# ------------------------------------------------------------------ scoring
def score(pred, lab):
    tp = int((pred & (lab == 2)).sum()); fp = int((pred & (lab == 1)).sum()); fn = int((~pred & (lab == 2)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "iou": round(tp / (tp + fp + fn), 3) if tp + fp + fn else 0.0}


def pool(parts):
    tot = {f: sum(p[f] for p in parts) for f in ("tp", "fp", "fn")}
    p = tot["tp"] / max(tot["tp"] + tot["fp"], 1); r = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    return {**tot, "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "iou": round(tot["tp"] / max(tot["tp"] + tot["fp"] + tot["fn"], 1), 3)}


def object_hits(pred, win, site_code):
    """Drone-mapped landslides the map touches with >= 3 pixels and >= 10 % of their area."""
    gp = C.RAW / "drone_inventory" / "launglon_landslides_2026.gpkg"
    ls = gpd.read_file(gp, layer=f"{site_code}_landslides").to_crs(C.CRS)
    tr = rasterio.windows.transform(win, C.grid_transform())
    rows = []
    for _, r in ls.iterrows():
        m = rasterize([r.geometry], (int(win.height), int(win.width)), transform=tr, fill=0, default_value=1, dtype="uint8").astype(bool)
        n = int(m.sum())
        if n == 0:
            continue
        h = int((pred & m).sum())
        rows.append({"id": r["id"], "area_ha": round(r.geometry.area / 1e4, 3), "px": n, "hit_px": h,
                     "hit": bool(h >= 3 and h >= 0.1 * n)})
    return rows


def site_windows(margin=60):
    sites = C.read_json(C.GRID_DIR / "sites.json")
    with rasterio.open(C.GRID_DIR / "sites.tif") as s:
        sid = s.read(1)
    out = []
    for m in sites:
        rr, cc = np.where(sid == m["index"])
        r0, r1 = max(rr.min() - margin, 0), min(rr.max() + margin, C.HEIGHT)
        c0, c1 = max(cc.min() - margin, 0), min(cc.max() + margin, C.WIDTH)
        out.append((m, Window(c0, r0, c1 - c0, r1 - r0)))
    return out


# ------------------------------------------------------------------ training data
def temporal_negatives(tag, seed):
    """Every labelled survey pixel + a random township sample, from a no-event pair: all label 0."""
    rng = np.random.default_rng(seed)
    aoi = C.aoi_mask()
    p = N_TEMPORAL_NEG / aoi.sum()
    Xs, ss = [], []
    for r0 in range(0, C.HEIGHT, 512):
        w = Window(0, r0, C.WIDTH, min(512, C.HEIGHT - r0))
        d = read_stack(tag, w, FEATURES + ["shadow", "labels", "sites"])
        a = aoi[r0:r0 + int(w.height)]
        sel = valid_mask(d) & a & ((rng.random(a.shape) < p) | (d["labels"] > 0))
        if sel.any():
            Xs.append(X_of(d, sel)); ss.append(d["sites"][sel])
    X, site = np.concatenate(Xs), np.concatenate(ss)
    print(f"  temporal negatives {tag}: {len(X)} px ({int((site > 0).sum())} on survey ground)", flush=True)
    return X, site


# ------------------------------------------------------------------ township prediction
def predict_township(tag, clf):
    """Probability (uint8 %) on the whole grid, predicted in 512-row blocks, plus what post-processing needs."""
    aoi = C.aoi_mask()
    prob = np.zeros((C.HEIGHT, C.WIDTH), "uint8")
    valid = np.zeros((C.HEIGHT, C.WIDTH), bool)
    for r0 in range(0, C.HEIGHT, 512):
        w = Window(0, r0, C.WIDTH, min(512, C.HEIGHT - r0))
        d = read_stack(tag, w, FEATURES + ["shadow"])
        sel = valid_mask(d) & aoi[r0:r0 + int(w.height)]
        valid[r0:r0 + int(w.height)] = sel
        if sel.any():
            blk = np.zeros(sel.shape, "float32")
            blk[sel] = clf.predict_proba(X_of(d, sel))[:, 1]
            prob[r0:r0 + int(w.height)] = np.round(blk * 100).astype("uint8")
    aux = read_stack(tag, Window(0, 0, C.WIDTH, C.HEIGHT), ["slope", "wc", "shadow", "d_vv", "d_vh"])
    return prob, valid, aux


def township_map(prob, valid, aux, t, mmu):
    return postprocess((prob >= round(t * 100)) & valid, aux, mmu)


def km2(b):
    return float(b.sum()) * C.RES ** 2 / 1e6


# ------------------------------------------------------------------ main
def main():
    wins = site_windows()
    data = {m["code"]: (m, w, read_stack(EVENT_TAG, w, FEATURES + ["shadow", "labels", "sites"])) for m, w in wins}
    tabs = {}
    for code, (m, w, d) in data.items():
        sel = (d["sites"] == m["index"]) & np.isin(d["labels"], (1, 2)) & valid_mask(d)
        tabs[code] = (X_of(d, sel), (d["labels"][sel] == 2).astype(int))
    negs = [temporal_negatives(t, i) for i, t in enumerate(NEG_TAGS)]
    Xn, sn = np.concatenate([n[0] for n in negs]), np.concatenate([n[1] for n in negs])

    # leave-one-site-out
    prob_oof, auc = {}, {}
    for code, (m, w, d) in data.items():
        keep = sn != m["index"]          # the held-out site's own pre-event ground stays out as well
        Xtr = np.concatenate([tabs[k][0] for k in tabs if k != code] + [Xn[keep]])
        ytr = np.concatenate([tabs[k][1] for k in tabs if k != code] + [np.zeros(int(keep.sum()), int)])
        clf = model().fit(Xtr, ytr)
        p = clf.predict_proba(X_of(d).reshape(-1, len(FEATURES)))[:, 1].reshape(d["d_vv"].shape)
        prob_oof[code] = p
        lab = np.where(d["sites"] == m["index"], d["labels"], 0)
        sel = np.isin(lab, (1, 2)) & valid_mask(d)
        auc[code] = round(float(roc_auc_score(lab[sel] == 2, p[sel])), 3)
        print(f"  LOSO {m['name']:20s} AUC {auc[code]}", flush=True)

    grid = []
    for t in THRESHOLDS:
        for mmu in MMUS:
            parts = []
            for code, (m, w, d) in data.items():
                lab = np.where(d["sites"] == m["index"], d["labels"], 0)
                parts.append(score(postprocess(prob_oof[code] >= t, d, mmu), lab))
            grid.append({"t": float(t), "mmu": mmu, **{k: v for k, v in pool(parts).items() if k in ("precision", "recall", "f1")}})

    # final model on everything, then township maps for the event and the calibration placebo
    Xall = np.concatenate([t[0] for t in tabs.values()] + [Xn])
    yall = np.concatenate([t[1] for t in tabs.values()] + [np.zeros(len(Xn), int)])
    clf = model().fit(Xall, yall)
    ev = predict_township(EVENT_TAG, clf)
    cal = predict_township(CAL_TAG, clf)
    for g in grid:
        g["event_km2"] = round(km2(township_map(*ev, g["t"], g["mmu"])), 3)
        g["cal_placebo_km2"] = round(km2(township_map(*cal, g["t"], g["mmu"])), 3)
        g["fa_ratio"] = round(g["cal_placebo_km2"] / max(g["event_km2"], 1e-9), 3)
        print(f"  t={g['t']:.2f} mmu={g['mmu']:2d}  F1 {g['f1']:.3f} (P {g['precision']:.2f} R {g['recall']:.2f})  "
              f"event {g['event_km2']:6.2f} km2  placebo {g['cal_placebo_km2']:6.2f} km2  ratio {g['fa_ratio']:.2f}", flush=True)
    ok = [g for g in grid if g["fa_ratio"] <= MAX_FA_RATIO]
    best = max(ok, key=lambda g: g["f1"]) if ok else min(grid, key=lambda g: g["fa_ratio"])
    T, MMU = best["t"], best["mmu"]
    print(f"  chosen threshold {T}, MMU {MMU} px ({MMU * C.RES ** 2 / 1e4:.2f} ha)  [{'meets' if ok else 'NO threshold meets'} fa_ratio <= {MAX_FA_RATIO}]")

    # per-site scores of all three detectors at the chosen settings
    results = {"M0": {}, "M1": {}, "M2": {"_auc": auc}}
    for code, (m, w, d) in data.items():
        lab = np.where(d["sites"] == m["index"], d["labels"], 0)
        maps = {"M0": postprocess(m0(d), d, MMU), "M1": postprocess(m1(d), d, MMU), "M2": postprocess(prob_oof[code] >= T, d, MMU)}
        for k, pred in maps.items():
            s = score(pred, lab)
            hits = object_hits(pred, w, code)
            s.update(objects=len(hits), objects_hit=sum(h["hit"] for h in hits),
                     area_ha_hit=round(sum(h["area_ha"] for h in hits if h["hit"]), 2), area_ha_all=round(sum(h["area_ha"] for h in hits), 2))
            if k == "M2":
                s["hits"] = hits
            results[k][code] = s
    pooled = {}
    for k in results:
        sites_k = [results[k][c] for c in data]
        pooled[k] = {**pool(sites_k), "objects": sum(s["objects"] for s in sites_k), "objects_hit": sum(s["objects_hit"] for s in sites_k),
                     "object_detection_rate": round(sum(s["objects_hit"] for s in sites_k) / max(sum(s["objects"] for s in sites_k), 1), 3),
                     "area_weighted_detection": round(sum(s["area_ha_hit"] for s in sites_k) / max(sum(s["area_ha_all"] for s in sites_k), 1e-9), 3)}
        print(f"  {k}: P {pooled[k]['precision']:.2f} R {pooled[k]['recall']:.2f} F1 {pooled[k]['f1']:.2f} "
              f"objects {pooled[k]['objects_hit']}/{pooled[k]['objects']} (area-weighted {pooled[k]['area_weighted_detection']:.2f})")

    # write the event map, the calibration placebo and the independent placebo
    det = township_map(*ev, T, MMU)
    prob, valid, _ = ev
    with rasterio.open(C.OUT / "landslide_probability.tif", "w", **C.grid_profile("uint8", 1, 255)) as dst:
        dst.write(np.where(valid, prob, 255).astype("uint8"), 1)
    with rasterio.open(C.OUT / "landslides.tif", "w", **C.grid_profile("uint8", 1, None)) as dst:
        dst.write(det.astype("uint8"), 1)
    del cal
    test = predict_township(TEST_TAG, clf)
    tdet = township_map(*test, T, MMU)
    with rasterio.open(C.OUT / "placebo_detections.tif", "w", **C.grid_profile("uint8", 1, None)) as dst:
        dst.write(tdet.astype("uint8"), 1)
    land = km2(valid)
    plc = {"event_km2": round(km2(det), 3), "placebo_km2": round(km2(tdet), 3), "valid_land_km2": round(land, 1),
           "calibration_placebo_km2": best["cal_placebo_km2"], "max_fa_ratio": MAX_FA_RATIO,
           "placebo_period": C.read_json(FEAT / f"{TEST_TAG}.json"), "calibration_period": C.read_json(FEAT / f"{CAL_TAG}.json"),
           "training_negative_periods": [C.read_json(FEAT / f"{t}.json") for t in NEG_TAGS]}
    C.write_json(C.OUT / "placebo.json", plc)
    print(f"event map {plc['event_km2']:.2f} km2 of {land:.0f} km2 valid land; independent placebo (15 Aug) {plc['placebo_km2']:.2f} km2")

    C.write_json(C.OUT / "validation.json", {"threshold": T, "mmu_px": MMU, "features": FEATURES, "pooled": pooled,
                                             "sites": {m["code"]: m for m, _ in wins}, "per_site": results, "threshold_grid": grid,
                                             "n_train": {"landslide": int(sum(t[1].sum() for t in tabs.values())),
                                                         "stable_event": int(sum((t[1] == 0).sum() for t in tabs.values())),
                                                         "temporal_negatives": int(len(Xn))}})
    joblib.dump({"model": clf, "features": FEATURES, "threshold": T, "mmu_px": MMU}, C.OUT / "model_M2.joblib")


if __name__ == "__main__":
    main()
