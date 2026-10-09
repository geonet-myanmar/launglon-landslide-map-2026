"""Step 06 - clean the MIMU and OSM vectors to the township, and rasterise distance layers on the 10 m grid.

Outputs: data/vectors.gpkg (layers: township, vtracts, villages, schools, health, roads, waterways, buildings)
         data/grid/dist_stream.tif, data/grid/dist_road.tif (metres)
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import LineString, Point

import config as C

VEC = C.DATA / "vectors.gpkg"


def osm_lines(path, tag):
    rows = []
    for e in json.loads(path.read_text(encoding="utf8"))["elements"]:
        if e.get("type") == "way" and len(e.get("geometry", [])) > 1:
            rows.append({tag: e.get("tags", {}).get(tag), "name": e.get("tags", {}).get("name"), "osm_id": e["id"],
                         "geometry": LineString([(p["lon"], p["lat"]) for p in e["geometry"]])})
    return gpd.GeoDataFrame(rows, crs=4326)


def main():
    m = C.RAW / "mimu"
    aoi = C.aoi()
    poly = aoi.geometry.union_all()
    out = {}
    out["township"] = gpd.read_file(m / "township.geojson").to_crs(C.CRS)
    out["vtracts"] = gpd.read_file(m / "vtracts.geojson").to_crs(C.CRS)[["VT", "VT_PCODE", "VT_MMR", "geometry"]]
    out["villages"] = gpd.read_file(m / "villages.geojson").to_crs(C.CRS)[["VILLAGE", "VLG_PCODE", "VLG_MMR", "VT", "VT_PCODE", "geometry"]]
    sch = gpd.read_file(m / "schools.geojson").to_crs(C.CRS)
    out["schools"] = sch[sch.within(poly.buffer(200))][["schoolname", "geometry"]].rename(columns={"schoolname": "name"})
    hf = gpd.read_file(m / "health.geojson").to_crs(C.CRS)
    out["health"] = hf[hf.within(poly.buffer(200))][["nmHsp_eng", "lvlHsp_eng", "geometry"]].rename(columns={"nmHsp_eng": "name", "lvlHsp_eng": "level"})

    mroads = gpd.read_file(m / "roads.geojson").to_crs(C.CRS).clip(poly.buffer(100))
    oroads = osm_lines(C.RAW / "osm" / "roads.json", "highway").to_crs(C.CRS).clip(poly.buffer(100))
    keep = ["trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "track", "service", "road"]
    oroads = oroads[oroads.highway.isin(keep)]
    out["roads"] = gpd.GeoDataFrame(pd.concat([
        oroads.assign(source="OSM", cls=oroads.highway)[["source", "cls", "name", "geometry"]],
        mroads.assign(source="MIMU", cls=mroads.Road_Type, name=mroads.Route)[["source", "cls", "name", "geometry"]]],
        ignore_index=True), crs=C.CRS).explode(index_parts=False)
    ww = osm_lines(C.RAW / "osm" / "waterways.json", "waterway").to_crs(C.CRS).clip(poly.buffer(100))
    riv = gpd.read_file(m / "rivers.geojson").to_crs(C.CRS).clip(poly.buffer(100))
    out["waterways"] = gpd.GeoDataFrame(pd.concat([ww.assign(source="OSM")[["source", "waterway", "name", "geometry"]],
                                                   riv.assign(source="MIMU", waterway="river", name=riv.NAME)[["source", "waterway", "name", "geometry"]]],
                                                  ignore_index=True), crs=C.CRS)
    rows = []
    for e in json.loads((C.RAW / "osm" / "buildings.json").read_text(encoding="utf8"))["elements"]:
        c = e.get("center") or ({"lat": e["lat"], "lon": e["lon"]} if "lat" in e else None)
        if c:
            rows.append({"osm_id": e["id"], "building": e.get("tags", {}).get("building"), "geometry": Point(c["lon"], c["lat"])})
    b = gpd.GeoDataFrame(rows, crs=4326).to_crs(C.CRS)
    out["buildings"] = b[b.within(poly)]

    # village tract of every village/building, for the per-tract tables
    vt = out["vtracts"][["VT", "VT_PCODE", "geometry"]]
    out["buildings"] = gpd.sjoin(out["buildings"], vt, how="left", predicate="within").drop(columns="index_right")

    if VEC.exists():
        VEC.unlink()
    for k, g in out.items():
        g.to_file(VEC, layer=k, driver="GPKG")
        print(f"{k:10s} {len(g):6d}")

    shape, tr = (C.HEIGHT, C.WIDTH), C.grid_transform()
    for name, g in (("dist_stream.tif", out["waterways"]), ("dist_road.tif", out["roads"])):
        r = rasterize(list(g.geometry), shape, transform=tr, fill=0, default_value=1, dtype="uint8", all_touched=True)
        d = ndimage.distance_transform_edt(r == 0) * C.RES
        with rasterio.open(C.GRID_DIR / name, "w", **C.grid_profile("float32", 1, None)) as dst:
            dst.write(np.minimum(d, 5000).astype("float32"), 1)
    print("distance rasters written")


if __name__ == "__main__":
    main()
