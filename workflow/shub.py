"""CDSE Sentinel Hub Process API helpers: tile the 10 m analysis grid and write results straight onto it."""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import rasterio
import requests
from rasterio.windows import Window
from shapely.geometry import box

import config as C

CRS_URL = "http://www.opengis.net/def/crs/EPSG/0/32647"


def tiles(tile=C.TILE):
    """Windows of the analysis grid (<= 2500 px) that touch the buffered township."""
    aoi_buf = C.aoi().geometry.union_all().buffer(C.BUFFER_M)
    for r0 in range(0, C.HEIGHT, tile):
        for c0 in range(0, C.WIDTH, tile):
            h, w = min(tile, C.HEIGHT - r0), min(tile, C.WIDTH - c0)
            x0, y1 = C.X0 + c0 * C.RES, C.Y1 - r0 * C.RES
            bb = (x0, y1 - h * C.RES, x0 + w * C.RES, y1)
            if box(*bb).intersects(aoi_buf):
                yield Window(c0, r0, w, h), bb


def post(ses, body):
    for attempt in range(8):
        try:
            r = requests.post(C.SH_PROCESS, json=body, headers=ses.headers(), timeout=900)
        except requests.RequestException:
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code == 200:
            with rasterio.open(io.BytesIO(r.content)) as src:
                return src.read()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(15 * (attempt + 1))
            continue
        raise RuntimeError(f"Process API {r.status_code}: {r.text[:600]}")
    raise RuntimeError("Process API kept failing")


def to_grid(ses, out, data, evalscript, count, dtype, nodata=None, workers=4):
    """Run one Process API request per tile and mosaic the answers onto the 10 m grid (GeoTIFF)."""
    if out.exists():
        return "cached"
    tmp = out.with_name(out.stem + ".part.tif")
    jobs = list(tiles())

    def one(job):
        win, bb = job
        body = {"input": {"bounds": {"bbox": list(bb), "properties": {"crs": CRS_URL}}, "data": data},
                "output": {"width": int(win.width), "height": int(win.height),
                           "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
                "evalscript": evalscript}
        return win, post(ses, body)

    with rasterio.open(tmp, "w", **C.grid_profile(dtype, count, nodata)) as dst:
        with ThreadPoolExecutor(workers) as ex:
            for win, arr in ex.map(one, jobs):
                dst.write(arr[:count].astype(dtype), window=win)
    tmp.replace(out)
    return f"{len(jobs)} tiles"
