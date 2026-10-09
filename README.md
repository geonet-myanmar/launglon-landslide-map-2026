# Launglon Township landslide mapping with Sentinel-1 (CDSE)

Event: heavy rain on 26–29 September 2026 triggered debris flows across Launglon Township
(Dawei District, Tanintharyi; MIMU P-code MMR006002).

**Live dashboard:** https://geonet-myanmar.github.io/launglon-landslide-map-2026/

Run everything with `python workflow/run_all.py`. Each step caches its downloads, so a re-run only
fetches what is new. The dashboard is written to `docs/` (`index.html` + `overlays/`); open
`docs/index.html` locally, or push to update the live page.

## Deployment (GitHub Pages)

GitHub Pages serves the `docs/` folder of the `main` branch (`docs/.nojekyll` turns off Jekyll). The page is
static: data and vectors are embedded in `index.html`, the 10 m overlays are WebP files in `docs/overlays/`,
and Leaflet, d3 and Google Fonts load from public CDNs. To update it, re-run the workflow (or just
`python workflow/09_dashboard.py`), then commit `docs/` and push.

To run the workflow yourself, put your Copernicus Data Space Ecosystem login in `credentials_CDSE.txt` at the
project root (line 1 e-mail, line 2 password). The file is git-ignored, as is `data/` (about 22 GB of downloads
and intermediate rasters, rebuilt by the workflow).

## Why the old workflow was replaced

`launglon_landslides.py` (kept for reference) had five problems:

| Problem | Effect |
|---|---|
| Hard-coded box from 98.03 E | Missed the western part of the township (it reaches 97.90 E) and its islands |
| "Top 5 products" per window, any track, either orbit direction | Pre and post scenes from different geometries cannot be differenced |
| Whole GRD scenes downloaded (~1.7 GB each) | Hours of download for a 789 km² township |
| No calibration, terrain correction or geocoding | Raw GRD is not comparable between dates on steep slopes |
| One pre / one post scene | Speckle (±2 dB) is as large as the landslide signal; the only fix would be a spatial filter, which lowers resolution |

## New workflow (`workflow/`)

| Step | Script | What it does |
|---|---|---|
| 01 | `01_s1_inventory.py` | CDSE Sentinel Hub catalogue search over the real AOI polygon, grouped per orbit track and pass |
| 02 | `02_s1_rtc_download.py` | VV and VH **gamma0 RTC** (terrain-flattened with Copernicus DEM GLO-30) from the CDSE Process API, on a fixed **10 m** UTM 47N grid; no speckle filter, no multilooking; local incidence angle and layover/shadow flags |
| 03 | `03_ancillary.py` | Copernicus DEM (slope, aspect, TPI, curvature), ESA WorldCover 10 m, Sentinel-2 NDVI (Jan–Mar 2026 median; post-event 30 Sep / 5 Oct), **MIMU** layers, OSM, Open-Meteo rainfall, drone inventory |
| 04 | `04_features.py` | Multi-temporal change features: post-event vs. the mean of all pre-event passes (temporal multilooking), z-scores against each pixel's own pre-event variability, local and regional context; plus the same features for no-event "placebo" pairs |
| 05 | `05_labels.py` | Rasterises the ten drone surveys (2–8 Oct 2026) into reference labels on the 10 m grid |
| 06 | `06_vectors.py` | Cleans MIMU and OSM layers to the township; distance-to-stream and distance-to-road rasters |
| 07 | `07_detect.py` | Three detectors (classic pair M0, unsupervised multi-temporal M1, trained classifier M2), leave-one-site-out validation, false-alarm control with placebo pairs, township map |
| 08 | `08_analysis.py` | Landslide polygons and attributes, village-tract totals, exposure, frequency ratios, optical cross-check, rainfall |
| 09 | `09_dashboard.py` | Builds `docs/index.html` (the GitHub Pages site) with 10 m overlays |

**Resolution.** The SAR data is never resampled to a coarser grid. Sentinel-1 IW GRDH has a 10 m pixel
spacing (about 20 × 22 m true resolution), and every product here is on that 10 m grid. Speckle is reduced by
averaging **in time** (10 pre-event passes), not in space. Coarser ancillary layers (the 30 m DEM) are only
upsampled to 10 m. Dashboard overlays are Web Mercator at 10.3 m, which is 10 m on the ground at 13.9° N.

## Results (run of 9 Oct 2026)

- **1,068 landslides, 24.1 km²** mapped on the 2 Oct pass (536 high, 374 medium, 158 low confidence), 99 % in tree cover.
- **Held-out validation** against ten drone surveys (165 landslides, 241 ha; each site scored by a model that never saw it):

  | Method | Precision | Recall | F1 |
  |---|---|---|---|
  | M0 classic pre/post pair, ±3 dB | 0.19 | 0.66 | 0.29 |
  | M1 multi-temporal anomaly (unsupervised) | 0.61 | 0.65 | 0.63 |
  | **M2 trained classifier (the map)** | **0.77** | **0.61** | **0.68** |

  M2 touches 90 of 165 drone landslides, and those hold 94 % of the drone-mapped landslide area; misses are mostly < 0.5 ha.
- **False alarms:** the same model flags 2.5 km² on a no-event pair (20 Sep, used to set the threshold) and 0.0 km²
  on an independent no-event pair (15 Aug).
- **Optical check (not used in training):** where Sentinel-2 saw the mapped ground clear (36 %), 75 % of it lost
  ≥ 0.15 NDVI, against 15 % of undetected slopes.
- **Exposure:** 29 of 113 MIMU villages lie within 500 m of a mapped landslide; 142 OSM buildings inside and
  560 within 50 m; 84 landslides cross a road (18.5 km of road inside landslide outlines); 20 schools within 500 m.
- **Trigger:** about 600 mm fell on 25–29 Sep (township mean). The 3-day peak at Ka Det Nge Htein (510 mm) is the
  highest since 1991 (previous record 318 mm in 2019); model rainfall, not a gauge.

Key files:

- `outputs/landslides.gpkg` / `.geojson` / `.csv`: mapped landslides with confidence, terrain, land cover, Sentinel-2 support and MIMU village tract
- `outputs/landslide_probability.tif`, `outputs/landslides.tif`: 10 m rasters (UTM 47N)
- `outputs/validation.json`: per-site and pooled scores of all three methods, plus the threshold table
- `outputs/model_M2.joblib` is not in the repository; `workflow/07_detect.py` rebuilds it
- `outputs/placebo.json`: false-alarm tests
- `outputs/village_tract_summary.csv`, `outputs/village_exposure.csv`

## Data sources

Copernicus Sentinel-1 and Sentinel-2 (CDSE Sentinel Hub), Copernicus DEM GLO-30, ESA WorldCover 2021,
MIMU GeoNode (township, village tracts, villages, roads, rivers, schools, health facilities),
OpenStreetMap, Open-Meteo historical weather API, and the geonet-myanmar/launglon-landslides-2026 drone
inventories (training and testing only).

## Limits

- So far only one post-event pass exists (2 Oct, ascending track 70). Descending track 135 has acquired nothing
  since 24 Sep. Re-running the workflow picks up new passes automatically: a second post-event pass averages
  down the post-event speckle, and a descending pass covers the east-facing slopes the ascending geometry sees poorly.
- Landslides smaller than the minimum mapping unit are mostly missed.
- The map ranks where to look. It does not replace a field check.
