"""Shared settings for the Launglon Sentinel-1 landslide workflow.

Every step imports this module. The analysis grid is fixed here: UTM 47N (EPSG:32647), 10 m pixels,
which is the native pixel spacing of Sentinel-1 IW GRDH. No step resamples the SAR data to a coarser grid.
"""
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
AOI_FILE = ROOT / "Launglon_Township.geojson"
CRED_FILE = ROOT / "credentials_CDSE.txt"

DATA = ROOT / "data"
RAW = DATA / "raw"           # third-party downloads (MIMU, OSM, DEM, WorldCover, drone inventory)
S1_DIR = DATA / "s1"         # Sentinel-1 gamma0 RTC stacks, one folder per track
S2_DIR = DATA / "s2"         # Sentinel-2 NDVI (optical cross-check only)
GRID_DIR = DATA / "grid"     # ancillary layers resampled onto the 10 m analysis grid
OUT = ROOT / "outputs"
DASH = ROOT / "docs"            # GitHub Pages serves this folder (main branch, /docs)
for d in (RAW, S1_DIR, S2_DIR, GRID_DIR, OUT, DASH):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- analysis grid
CRS = "EPSG:32647"
RES = 10.0
BUFFER_M = 1000              # context margin around the township polygon
# Snapped to whole 10 m pixels; derived from the AOI bounds (381200-417946 E, 1495848-1572262 N) + 1 km.
X0, Y0, X1, Y1 = 380200.0, 1494840.0, 418950.0, 1573270.0
WIDTH = int(round((X1 - X0) / RES))     # 3875
HEIGHT = int(round((Y1 - Y0) / RES))    # 7843
TILE = 2000                  # Sentinel Hub Process API limit is 2500 px per side

# ---------------------------------------------------------------- event
EVENT_START = "2026-09-26"   # rain began on the night of 26 Sep; debris flows 26-29 Sep (Sentinel Asia, MoeMaKa)
EVENT_END = "2026-09-29"
PRE_START = "2026-06-01"     # pre-event reference stack: same monsoon season, same track
POST_END = "2026-10-31"      # picks up later post-event passes automatically when they exist

# ---------------------------------------------------------------- CDSE endpoints
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_CATALOG = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
SH_PROCESS = "https://sh.dataspace.copernicus.eu/api/v1/process"
ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# dB values are stored as UINT16: DN = (dB + DB_OFFSET) * DB_SCALE, 0 = no data (1 mdB precision)
DB_OFFSET, DB_SCALE = 50.0, 1000.0


class CDSESession:
    """Password-grant token for the cdse-public client, refreshed before its 10 min expiry."""

    def __init__(self):
        lines = CRED_FILE.read_text().splitlines()
        self._user, self._pw = lines[0].strip(), lines[1].strip()
        self._tok, self._exp = None, 0.0

    def token(self):
        if time.time() > self._exp - 90:
            r = requests.post(TOKEN_URL, data={"client_id": "cdse-public", "username": self._user,
                                               "password": self._pw, "grant_type": "password"}, timeout=60)
            r.raise_for_status()
            js = r.json()
            self._tok, self._exp = js["access_token"], time.time() + js.get("expires_in", 600)
        return self._tok

    def headers(self):
        return {"Authorization": f"Bearer {self.token()}"}


def aoi(crs=CRS):
    import geopandas as gpd
    return gpd.read_file(AOI_FILE).to_crs(crs)


def grid_transform():
    from rasterio.transform import from_origin
    return from_origin(X0, Y1, RES, RES)


def grid_profile(dtype="float32", count=1, nodata=None):
    p = dict(driver="GTiff", width=WIDTH, height=HEIGHT, count=count, dtype=dtype, crs=CRS,
             transform=grid_transform(), tiled=True, blockxsize=512, blockysize=512,
             compress="deflate", predictor=2 if "int" in dtype else 3, BIGTIFF="IF_SAFER")
    if nodata is not None:
        p["nodata"] = nodata
    return p


def aoi_mask(buffer_m=0.0):
    """Boolean array on the analysis grid: True inside the township polygon (optionally buffered)."""
    from rasterio.features import geometry_mask
    g = aoi().geometry.buffer(buffer_m) if buffer_m else aoi().geometry
    return ~geometry_mask(list(g), out_shape=(HEIGHT, WIDTH), transform=grid_transform())


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf8")
