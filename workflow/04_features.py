"""Step 04 - multi-temporal SAR change features on the 10 m grid.

Speckle in a single GRDH pass (ENL ~4.4) is about +/-2 dB, the same size as many landslide signals. The old
workflow had one pre and one post scene, so it could only fight speckle with a spatial filter (= coarser
effective resolution). Here the pre-event reference is the TEMPORAL mean of every pre-event pass on the same
track (10 passes, Jun-Sep 2026, ENL ~44), and each pixel's own pre-event variability (temporal std in dB) turns
the change into a standardised anomaly. Nothing is resampled; every feature stays on the 10 m grid.

Bands of data/features/<tag>.tif (float32, NaN = no data):
  pre_vv, pre_vh            temporal mean of pre-event passes (dB, mean taken in linear power)
  std_vv, std_vh            temporal std of pre-event passes (dB)
  post_vv, post_vh          post-event pass (mean of passes if several) (dB)
  d_vv, d_vh                post - pre (dB)
  z_vv, z_vh                (post - pre) / max(std, 0.5 dB)
  dl_vv, dl_vh              post - last pre-event pass (dB): shortest-baseline change
  d_cr                      change of the VH/VV cross-pol ratio (dB): vegetation loss lowers it
  d_vv_m3, d_vh_m3          3x3 mean of d_vv / d_vh (local context, same 10 m grid)
  d_vv_m7, d_vh_m7          7x7 mean of d_vv / d_vh
  d_vv_m51, d_vh_m51        51x51 (~500 m) mean of d_vv / d_vh: the regional background change (e.g. a wet day)

Usage: python 04_features.py                    event features (pre: all pre-event passes, post: post-event passes)
       python 04_features.py placebo YYYYMMDD   no-event control: that pre-event pass vs. all passes before it
                                                (default: the last pre-event pass)
"""
import sys

import numpy as np
import rasterio
from scipy import ndimage

import config as C

BANDS = ["pre_vv", "pre_vh", "std_vv", "std_vh", "post_vv", "post_vh", "d_vv", "d_vh", "z_vv", "z_vh",
         "dl_vv", "dl_vh", "d_cr", "d_vv_m3", "d_vh_m3", "d_vv_m7", "d_vh_m7", "d_vv_m51", "d_vh_m51"]
FEAT_DIR = C.DATA / "features"
FEAT_DIR.mkdir(exist_ok=True)


def read_db(path):
    with rasterio.open(path) as s:
        a = s.read().astype("float32")
    a[a == 0] = np.nan
    return a / C.DB_SCALE - C.DB_OFFSET          # (2, H, W): VV, VH in dB


def nanmean_filter(a, size):
    ok = np.isfinite(a)
    num = ndimage.uniform_filter(np.where(ok, a, 0).astype("float32"), size)
    den = ndimage.uniform_filter(ok.astype("float32"), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.3, num / den, np.nan).astype("float32")


def build(tag, pre_files, post_files):
    out = FEAT_DIR / f"{tag}.tif"
    n = len(pre_files)
    s_lin = np.zeros((2, C.HEIGHT, C.WIDTH), "float32")
    s_db = np.zeros_like(s_lin)
    s_db2 = np.zeros_like(s_lin)
    cnt = np.zeros_like(s_lin, dtype="uint8")
    for f in pre_files:
        db = read_db(f)
        ok = np.isfinite(db)
        s_lin += np.where(ok, 10 ** (db / 10), 0)
        s_db += np.where(ok, db, 0)
        s_db2 += np.where(ok, db * db, 0)
        cnt += ok
    last = db                                      # most recent pre-event pass (files are date-sorted)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = cnt >= max(3, n // 2)
        pre = np.where(valid, 10 * np.log10(s_lin / cnt), np.nan).astype("float32")
        mu = s_db / cnt
        std = np.where(valid, np.sqrt(np.maximum(s_db2 / cnt - mu * mu, 0)), np.nan).astype("float32")
    del s_lin, s_db, s_db2, mu

    p_lin = np.zeros((2, C.HEIGHT, C.WIDTH), "float32")
    p_cnt = np.zeros_like(p_lin, dtype="uint8")
    for f in post_files:
        db = read_db(f)
        ok = np.isfinite(db)
        p_lin += np.where(ok, 10 ** (db / 10), 0)
        p_cnt += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        post = np.where(p_cnt > 0, 10 * np.log10(p_lin / p_cnt), np.nan).astype("float32")
    del p_lin

    d = post - pre
    z = d / np.maximum(std, 0.5)
    dl = post - last
    d_cr = (post[1] - post[0]) - (pre[1] - pre[0])
    layers = {"pre_vv": pre[0], "pre_vh": pre[1], "std_vv": std[0], "std_vh": std[1], "post_vv": post[0],
              "post_vh": post[1], "d_vv": d[0], "d_vh": d[1], "z_vv": z[0], "z_vh": z[1], "dl_vv": dl[0],
              "dl_vh": dl[1], "d_cr": d_cr}
    prof = C.grid_profile("float32", len(BANDS), np.nan)
    with rasterio.open(out, "w", **prof) as dst:
        for i, b in enumerate(BANDS, 1):
            if b in layers:
                arr = layers[b]
            else:
                pol = 0 if "_vv_" in b else 1
                arr = nanmean_filter(d[pol], int(b.rsplit("_m", 1)[1]))
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, b)
    m = C.aoi_mask()
    print(f"{tag}: pre {n} passes, post {len(post_files)} passes -> {out.name}; "
          f"AOI median d_vv {np.nanmedian(d[0][m]):+.2f} dB, d_vh {np.nanmedian(d[1][m]):+.2f} dB, "
          f"median std_vh {np.nanmedian(std[1][m]):.2f} dB")
    C.write_json(FEAT_DIR / f"{tag}.json", {"pre": [f.stem for f in pre_files], "post": [f.stem for f in post_files], "bands": BANDS})


def main():
    inv = C.read_json(C.DATA / "s1_inventory.json")
    args = sys.argv[1:]
    for trk in inv["usable_tracks"]:
        d = C.S1_DIR / trk
        pre = [d / f"{a['date'].replace('-', '')}.tif" for a in inv["tracks"][trk]["pre"]]
        post = [d / f"{a['date'].replace('-', '')}.tif" for a in inv["tracks"][trk]["post"]]
        if args and args[0] == "placebo":
            # same features on a period with no known event: one pre-event pass vs. the passes before it
            day = args[1] if len(args) > 1 else pre[-1].stem
            k = [f.stem for f in pre].index(day)
            build(f"{trk}_placebo_{day}", pre[:k], pre[k:k + 1])
        else:
            build(f"{trk}_event", pre, post)


if __name__ == "__main__":
    main()
