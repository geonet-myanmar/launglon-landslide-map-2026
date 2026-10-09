"""Step 03 - ancillary layers: terrain, land cover, optical NDVI, MIMU and OSM vectors, rainfall, drone inventory.

Rasters are written onto the same 10 m grid as the SAR stack (data/grid/). Coarser sources are only ever
UPSAMPLED to 10 m (DEM 30 m -> bilinear); 10 m sources (WorldCover, Sentinel-2) keep their pixel size.

Sources
  Copernicus DEM GLO-30 ........ AWS open data (COG)              -> dem.tif, slope.tif, aspect.tif, tpi.tif, curv.tif
  ESA WorldCover 2021 v200 ..... AWS open data (COG, 10 m)        -> worldcover.tif
  Sentinel-2 L2A (CDSE SH) ..... Jan-Mar 2026 dry-season median NDVI, and the clearest post-event NDVI/RGB
  MIMU GeoNode WFS ............. township, village tracts, villages, roads, rivers, health facilities, schools
  OpenStreetMap (Overpass) ..... buildings, roads, waterways
  Open-Meteo archive ........... daily rainfall on a 0.1 deg grid + 35-year record at Ka Det Nge Htein
  geonet-myanmar drone survey .. 10 post-event drone inventories (validation only)

Usage: python 03_ancillary.py [terrain worldcover s2 mimu osm rain drone]   (no argument = all)
"""
import json
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import Resampling, reproject
from scipy import ndimage

import config as C
import shub

G = C.GRID_DIR


def save(name, arr, dtype="float32", nodata=None):
    with rasterio.open(G / name, "w", **C.grid_profile(dtype, 1, nodata)) as dst:
        dst.write(arr.astype(dtype), 1)


def warp_to_grid(urls, resampling, dtype="float32", fill=np.nan):
    out = np.full((C.HEIGHT, C.WIDTH), fill, dtype=dtype)
    for url in urls:
        try:
            with rasterio.open("/vsicurl/" + url) as src:
                tmp = np.full_like(out, fill)
                reproject(rasterio.band(src, 1), tmp, dst_transform=C.grid_transform(), dst_crs=C.CRS,
                          resampling=resampling, src_nodata=src.nodata, dst_nodata=fill)
        except rasterio.errors.RasterioIOError:
            print("  missing (sea-only tile?):", url.rsplit("/", 1)[-1])
            continue
        good = ~np.isnan(tmp) if np.isnan(fill) else tmp != fill
        out[good] = tmp[good]
    return out


# ------------------------------------------------------------------ terrain
def terrain():
    if (G / "curv.tif").exists():
        return print("terrain: cached")
    base = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{0}_00_{1}_00_DEM/Copernicus_DSM_COG_10_{0}_00_{1}_00_DEM.tif"
    urls = [base.format(n, e) for n in ("N13", "N14") for e in ("E097", "E098")]
    dem = warp_to_grid(urls, Resampling.bilinear)
    dem[np.isnan(dem)] = 0.0                       # GLO-30 omits open sea; sea level there
    save("dem.tif", dem)
    gy, gx = np.gradient(dem, C.RES)               # rows run north -> south
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360   # 0 = north, clockwise; dz/dy is +south
    save("slope.tif", slope)
    save("aspect.tif", aspect)
    save("tpi.tif", dem - ndimage.uniform_filter(dem, 25))   # topographic position over 250 m
    sm = ndimage.gaussian_filter(dem, 1.5)
    save("curv.tif", -ndimage.laplace(sm) / C.RES ** 2 * 100)  # >0 convex (ridges), <0 concave (hollows)
    print(f"terrain: DEM {np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} m, slope p95 {np.percentile(slope, 95):.1f} deg")


# ------------------------------------------------------------------ land cover
def worldcover():
    if (G / "worldcover.tif").exists():
        return print("worldcover: cached")
    url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N12E096_Map.tif"
    wc = warp_to_grid([url], Resampling.nearest, dtype="uint8", fill=0)
    save("worldcover.tif", wc, "uint8", 0)
    vals, n = np.unique(wc, return_counts=True)
    print("worldcover classes:", dict(zip(vals.tolist(), (n / n.sum()).round(3).tolist())))


# ------------------------------------------------------------------ Sentinel-2
EVAL_S2_PRE = """//VERSION=3
function setup(){return{input:[{bands:["B04","B08","SCL","dataMask"]}],output:{bands:2,sampleType:"FLOAT32"},mosaicking:"ORBIT"};}
function med(a){a.sort(function(x,y){return x-y});var n=a.length;return n%2?a[(n-1)/2]:(a[n/2-1]+a[n/2])/2;}
function evaluatePixel(ss){var v=[];for(var i=0;i<ss.length;i++){var s=ss[i];
 if(s.dataMask&&(s.SCL==4||s.SCL==5||s.SCL==6)){var d=s.B08+s.B04;if(d>0)v.push((s.B08-s.B04)/d);}}
 return v.length?[med(v),v.length]:[NaN,0];}
"""

# latest clear observation per pixel; band 3 = day of month of that observation
EVAL_S2_POST = """//VERSION=3
function setup(){return{input:[{bands:["B02","B03","B04","B08","SCL","dataMask"]}],output:{bands:6,sampleType:"FLOAT32"},mosaicking:"ORBIT"};}
function evaluatePixel(ss,sc){var orb=(sc&&sc.orbits)?sc.orbits:sc;var best=-1,bt=0;for(var i=0;i<ss.length;i++){var s=ss[i];
 if(s.dataMask&&(s.SCL==4||s.SCL==5||s.SCL==6)){var t=new Date(orb[i].dateFrom).getTime();if(t>bt){bt=t;best=i;}}}
 if(best<0)return[NaN,0,0,0,0,0];var s=ss[best];var d=s.B08+s.B04;
 return[d>0?(s.B08-s.B04)/d:NaN,1,new Date(bt).getUTCDate(),s.B04,s.B03,s.B02];}
"""


def s2():
    ses = C.CDSESession()
    pre = [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": "2026-01-01T00:00:00Z", "to": "2026-03-31T23:59:59Z"},
                                                     "maxCloudCoverage": 25}}]
    print("s2 pre (Jan-Mar 2026 median NDVI):", shub.to_grid(ses, C.S2_DIR / "ndvi_pre_2026JanMar.tif", pre, EVAL_S2_PRE, 2, "float32"))
    post = [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": "2026-09-30T00:00:00Z", "to": "2026-10-09T23:59:59Z"},
                                                      "maxCloudCoverage": 80}}]
    print("s2 post (latest clear, 30 Sep - 9 Oct):", shub.to_grid(ses, C.S2_DIR / "post_event.tif", post, EVAL_S2_POST, 6, "float32"))


# ------------------------------------------------------------------ MIMU
MIMU_WFS = "https://geonode.themimu.info/geoserver/wfs"
MIMU_LAYERS = {   # key: (GeoNode layer, CQL filter or None for a bbox query)
    "township": ("mmr_polbnda_adm3_250k_mimu_1", "TS='Launglon'"),
    "vtracts": ("mmr_tni_polbnda_adm4_unhcr_mimu_250k", "TS='Launglon'"),
    "villages": ("mmr_tni_pplp2_250k_mimu", "TS='Launglon'"),
    "roads": ("mmr_rdsl_mimu_250k", None),
    "rivers": ("myanmar_river_network", None),
    "health": ("health_facilities_myanmar2020_v20241016", None),
    "schools": ("formal_sector_school_location_lowermyanmar_2019", None),
}
BBOX = (97.85, 13.45, 98.30, 14.30)


def mimu():
    d = C.RAW / "mimu"
    d.mkdir(exist_ok=True)
    for key, (layer, cql) in MIMU_LAYERS.items():
        f = d / f"{key}.geojson"
        if not f.exists():
            p = dict(service="WFS", version="1.0.0", request="GetFeature", typeName=f"geonode:{layer}",
                     outputFormat="application/json", srsName="EPSG:4326")
            if cql:
                p["CQL_FILTER"] = cql
            else:
                p["BBOX"] = ",".join(map(str, BBOX)) + ",EPSG:4326"
            r = requests.get(MIMU_WFS, params=p, timeout=600)
            r.raise_for_status()
            f.write_text(r.text, encoding="utf8")
        g = gpd.read_file(f)
        print(f"mimu {key:9s} {len(g):5d} features  columns={list(g.columns)[:12]}")


# ------------------------------------------------------------------ OSM
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]


def osm():
    d = C.RAW / "osm"
    d.mkdir(exist_ok=True)
    bb = f"{BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]}"
    q = {"buildings": f'[out:json][timeout:600];(way["building"]({bb});relation["building"]({bb}););out center tags;',
         "roads": f'[out:json][timeout:600];way["highway"]({bb});out geom tags;',
         "waterways": f'[out:json][timeout:600];way["waterway"~"^(river|stream|canal)$"]({bb});out geom tags;'}
    for k, query in q.items():
        f = d / f"{k}.json"
        if not f.exists():
            for u in OVERPASS * 3:
                try:
                    r = requests.post(u, data={"data": query}, timeout=900,
                                      headers={"User-Agent": "launglon-s1-landslide-workflow"})
                    r.raise_for_status()
                    json.loads(r.text)
                    f.write_text(r.text, encoding="utf8")
                    break
                except Exception as e:
                    print("  overpass retry", u, str(e)[:120])
                    time.sleep(30)
        print(f"osm {k:9s} {len(json.loads(f.read_text(encoding='utf8'))['elements'])} elements")


# ------------------------------------------------------------------ rainfall
def rain():
    d = C.RAW / "rain"
    d.mkdir(exist_ok=True)
    poly = C.aoi("EPSG:4326").geometry.union_all()
    from shapely.geometry import Point
    pts = [(round(y, 2), round(x, 2)) for y in np.arange(13.55, 14.25, 0.1) for x in np.arange(97.95, 98.30, 0.1)
           if poly.buffer(0.03).contains(Point(x, y))]
    f = d / "openmeteo_grid_daily.json"
    if not f.exists():
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=300, params={
            "latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
            "start_date": "2026-06-01", "end_date": "2026-10-08", "daily": "precipitation_sum", "timezone": "Asia/Yangon"})
        r.raise_for_status()
        f.write_text(r.text, encoding="utf8")
    f2 = d / "openmeteo_kadet_1991_2026.json"
    if not f2.exists():
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=300, params={
            "latitude": 13.895, "longitude": 98.135, "start_date": "1991-01-01", "end_date": "2026-10-08",
            "daily": "precipitation_sum", "timezone": "Asia/Yangon"})
        r.raise_for_status()
        f2.write_text(r.text, encoding="utf8")
    f3 = d / "openmeteo_kadet_hourly.json"
    if not f3.exists():
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=300, params={
            "latitude": 13.895, "longitude": 98.135, "start_date": "2026-09-20", "end_date": "2026-10-03",
            "hourly": "precipitation", "timezone": "Asia/Yangon"})
        r.raise_for_status()
        f3.write_text(r.text, encoding="utf8")
    js = json.loads(f.read_text())
    js = js if isinstance(js, list) else [js]
    tot = [sum(v or 0 for v, t in zip(j["daily"]["precipitation_sum"], j["daily"]["time"]) if "2026-09-25" <= t <= "2026-09-29") for j in js]
    print(f"rain: {len(js)} grid points, 25-29 Sep totals {min(tot):.0f}-{max(tot):.0f} mm")


# ------------------------------------------------------------------ drone inventory (validation)
def drone():
    d = C.RAW / "drone_inventory"
    d.mkdir(exist_ok=True)
    base = "https://raw.githubusercontent.com/geonet-myanmar/launglon-landslides-2026/HEAD/outputs/"
    for f in ("launglon_landslides_2026.gpkg", "summary.json", "landslides.csv"):
        if not (d / f).exists():
            r = requests.get(base + f, timeout=600)
            r.raise_for_status()
            (d / f).write_bytes(r.content)
    import pyogrio
    layers = [l for l, _ in pyogrio.list_layers(d / "launglon_landslides_2026.gpkg")]
    sites = sorted({l.split("_")[0] for l in layers if l.endswith("_survey_extent")})
    print("drone sites:", sites)


STEPS = {"terrain": terrain, "worldcover": worldcover, "s2": s2, "mimu": mimu, "osm": osm, "rain": rain, "drone": drone}

if __name__ == "__main__":
    for k in (sys.argv[1:] or STEPS):
        STEPS[k]()
