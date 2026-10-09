"""Step 02 - Sentinel-1 gamma0 RTC at 10 m from the CDSE Sentinel Hub Process API.

For every pass of every usable track it requests VV and VH as terrain-flattened gamma0
(backCoeff GAMMA0_TERRAIN, Copernicus DEM GLO-30 orthorectification), with NO speckle filter and NO
multilooking, on the fixed 10 m UTM 47N grid of config.py. Speckle is reduced later by temporal (not spatial)
averaging, so the 10 m pixel grid is never coarsened.

Once per track it also stores the SAR geometry: local incidence angle and the layover/shadow flag.

Outputs (UINT16 dB, see config.DB_OFFSET/DB_SCALE; deflate-compressed GeoTIFF):
  data/s1/<TRACK>/<YYYYMMDD>.tif      band 1 VV, band 2 VH (0 = no data)
  data/s1/<TRACK>/geometry.tif        band 1 local incidence angle (deg), band 2 shadow/layover flag
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import requests
from rasterio.windows import Window

import config as C

EVAL_BS = """//VERSION=3
function setup(){return{input:[{bands:["VV","VH","dataMask"]}],output:{bands:2,sampleType:"UINT16"}};}
function enc(v,m){if(!m||!(v>0))return 0;var d=10*Math.log(v)/Math.LN10;return Math.max(1,Math.min(65535,Math.round((d+%g)*%g)));}
function evaluatePixel(s){return[enc(s.VV,s.dataMask),enc(s.VH,s.dataMask)];}
""" % (C.DB_OFFSET, C.DB_SCALE)

EVAL_GEOM = """//VERSION=3
function setup(){return{input:[{bands:["localIncidenceAngle","shadowMask","dataMask"]}],output:{bands:3,sampleType:"FLOAT32"}};}
function evaluatePixel(s){return[s.localIncidenceAngle,s.shadowMask,s.dataMask];}
"""


def tiles():
    aoi_buf = C.aoi().geometry.union_all().buffer(C.BUFFER_M)
    from shapely.geometry import box
    for r0 in range(0, C.HEIGHT, C.TILE):
        for c0 in range(0, C.WIDTH, C.TILE):
            h, w = min(C.TILE, C.HEIGHT - r0), min(C.TILE, C.WIDTH - c0)
            x0, y1 = C.X0 + c0 * C.RES, C.Y1 - r0 * C.RES
            bb = (x0, y1 - h * C.RES, x0 + w * C.RES, y1)
            if box(*bb).intersects(aoi_buf):
                yield Window(c0, r0, w, h), bb


def request(ses, bb, w, h, date, orbit, evalscript):
    body = {
        "input": {
            "bounds": {"bbox": list(bb), "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/32647"}},
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {"timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"},
                               "acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH",
                               "orbitDirection": orbit.upper()},
                "processing": {"backCoeff": "GAMMA0_TERRAIN", "orthorectify": True, "demInstance": "COPERNICUS_30",
                               "upsampling": "BILINEAR", "downsampling": "NEAREST", "speckleFilter": {"type": "NONE"}},
            }]},
        "output": {"width": w, "height": h, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript,
    }
    for attempt in range(6):
        r = requests.post(C.SH_PROCESS, json=body, headers=ses.headers(), timeout=600)
        if r.status_code == 200:
            with rasterio.open(io.BytesIO(r.content)) as src:
                return src.read()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (attempt + 1))
            continue
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    raise RuntimeError("Process API kept failing")


def fetch(ses, out, date, orbit, evalscript, count, dtype, nodata):
    if out.exists():
        return "cached"
    tmp = out.with_suffix(".part.tif")
    jobs = list(tiles())
    with rasterio.open(tmp, "w", **C.grid_profile(dtype, count, nodata)) as dst:
        def one(job):
            win, bb = job
            return win, request(ses, bb, int(win.width), int(win.height), date, orbit, evalscript)
        with ThreadPoolExecutor(4) as ex:
            for win, arr in ex.map(one, jobs):
                dst.write(arr[:count].astype(dtype), window=win)
    tmp.replace(out)
    return f"{len(jobs)} tiles"


def main(only=None):
    ses = C.CDSESession()
    inv = C.read_json(C.DATA / "s1_inventory.json")
    for trk in inv["usable_tracks"]:
        d = C.S1_DIR / trk
        d.mkdir(exist_ok=True)
        acq = inv["tracks"][trk]["pre"] + inv["tracks"][trk]["post"]
        orbit = acq[0]["orbit"]
        if only:
            acq = [a for a in acq if a["date"] in only]
        for a in acq:
            t0 = time.time()
            msg = fetch(ses, d / f"{a['date'].replace('-', '')}.tif", a["date"], orbit, EVAL_BS, 2, "uint16", 0)
            print(f"{trk} {a['date']} {a['platform']}: {msg} ({time.time() - t0:.0f} s)", flush=True)
        # geometry layers from the most recent post-event pass
        g = d / "geometry.tif"
        last = inv["tracks"][trk]["post"][-1]["date"]
        msg = fetch(ses, g, last, orbit, EVAL_GEOM, 3, "float32", None)
        print(f"{trk} geometry ({last}): {msg}", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
