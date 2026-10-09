"""Step 05 - reference labels on the 10 m grid from the ten post-event drone surveys (2-8 Oct 2026, 4-10 cm).

Inside each survey extent every landslide was outlined by an interpreter, so the extent is a complete
reference area: pixels are landslide (2) or not (1). Ambiguous ground is set aside (3) and used neither to
train nor to score: interpreter exclusions (roads, ground already bare in Jan 2026), outwash sediment on
paddy, and a 15 m ring around each outline (S1 geolocation/sidelobe tolerance, ~1.5 px).

Outputs: data/grid/labels.tif (uint8: 0 outside surveys, 1 stable, 2 landslide, 3 ignore)
         data/grid/sites.tif  (uint8: 0 none, 1..10 site index)   data/grid/sites.json
         outputs/drone_reference.geojson (landslide outlines in WGS84, for the dashboard)
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize

import config as C

GPKG = C.RAW / "drone_inventory" / "launglon_landslides_2026.gpkg"
NAMES = {"tby": "Tha Byar", "pny": "Pa Nyit road", "nmt": "Ngone Min Taung", "kdnh": "Ka Det Nge Htein",
         "kdg": "Ka Det Gyi", "tawkye": "Taw Kye", "thw": "Tha Win", "lhl": "Lel Hla", "pgz": "Pyin Gyi - Za Lut",
         "rbe": "Ra Be road"}
RING_M = 15.0


def layer(site, kind):
    names = {l for l, _ in pyogrio.list_layers(GPKG)}
    lname = f"{site}_{kind}"
    if lname not in names:
        return gpd.GeoDataFrame(geometry=[], crs=C.CRS)
    return gpd.read_file(GPKG, layer=lname).to_crs(C.CRS)


def main():
    shape, tr = (C.HEIGHT, C.WIDTH), C.grid_transform()
    labels = np.zeros(shape, "uint8")
    sites = np.zeros(shape, "uint8")
    meta, refs = [], []
    for i, (code, name) in enumerate(NAMES.items(), 1):
        ext = layer(code, "survey_extent")
        ls = layer(code, "landslides")
        ow = layer(code, "outwash")
        ex = layer(code, "exclusions")
        # survey extents of neighbouring sites overlap (sites.py minus_sites): first site keeps the overlap
        e = rasterize(list(ext.geometry), shape, transform=tr, fill=0, default_value=1, dtype="uint8").astype(bool)
        e &= sites == 0
        sites[e] = i
        lab = np.where(e, 1, 0).astype("uint8")
        ign = [g for g in list(ow.geometry) + list(ex.geometry) + list(ls.geometry.buffer(RING_M)) if not g.is_empty]
        if ign:
            lab[e & rasterize(ign, shape, transform=tr, fill=0, default_value=1, dtype="uint8").astype(bool)] = 3
        if len(ls):
            lab[e & rasterize(list(ls.geometry), shape, transform=tr, fill=0, default_value=1, dtype="uint8").astype(bool)] = 2
        labels[e] = lab[e]
        meta.append({"code": code, "name": name, "index": i, "survey_ha": round(float(ext.area.sum()) / 1e4, 1),
                     "n_landslides": int(len(ls)), "landslide_ha": round(float(ls.area.sum()) / 1e4, 2),
                     "outwash_ha": round(float(ow.area.sum()) / 1e4, 2),
                     "px_landslide": int((labels == 2)[sites == i].sum()), "px_stable": int((labels == 1)[sites == i].sum()),
                     "centroid": [round(v, 5) for v in ext.to_crs(4326).geometry.union_all().centroid.coords[0]]})
        if len(ls):
            r = ls[["id", "name", "type", "area_m2", "geometry"]].copy() if "type" in ls else ls[["id", "name", "area_m2", "geometry"]].copy()
            r["site"] = name
            refs.append(r)
        print(f"{name:20s} survey {meta[-1]['survey_ha']:7.1f} ha  landslides {len(ls):3d} ({meta[-1]['landslide_ha']:6.1f} ha)  "
              f"px L={meta[-1]['px_landslide']:6d} S={meta[-1]['px_stable']:7d}")
    for name, arr in (("labels.tif", labels), ("sites.tif", sites)):
        with rasterio.open(C.GRID_DIR / name, "w", **C.grid_profile("uint8", 1, None)) as dst:
            dst.write(arr, 1)
    C.write_json(C.GRID_DIR / "sites.json", meta)
    ref = gpd.GeoDataFrame(pd.concat(refs, ignore_index=True), crs=C.CRS)
    ext_all = pd.concat([layer(c, "survey_extent").assign(site=n) for c, n in NAMES.items()], ignore_index=True)
    gpd.GeoDataFrame(ext_all, crs=C.CRS).to_crs(4326).to_file(C.OUT / "drone_survey_extents.geojson", driver="GeoJSON")
    ref.to_crs(4326).to_file(C.OUT / "drone_reference.geojson", driver="GeoJSON")
    print(f"total: {sum(m['n_landslides'] for m in meta)} landslides, {sum(m['landslide_ha'] for m in meta):.1f} ha "
          f"in {sum(m['survey_ha'] for m in meta):.0f} ha surveyed")


if __name__ == "__main__":
    main()
