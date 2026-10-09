"""Step 01 - Sentinel-1 IW GRDH acquisitions over the township, grouped by track and pass.

Replaces the old script's "top 5 products per window" search, which mixed tracks and orbit directions and
downloaded whole ~1.7 GB scenes. Here every pass (two adjacent slices on the same day) is one acquisition,
and pre/post sets are built per track so geometry never mixes in a change image.

Output: data/s1_inventory.json
"""
from collections import defaultdict

import requests
from shapely.geometry import mapping

import config as C


def main():
    ses = C.CDSESession()
    geom = C.aoi("EPSG:4326").geometry.union_all().buffer(0.005)
    body = {"collections": ["sentinel-1-grd"],
            "datetime": f"{C.PRE_START}T00:00:00Z/{C.POST_END}T23:59:59Z",
            "intersects": mapping(geom.convex_hull), "limit": 100}
    feats, nxt = [], None
    while True:
        if nxt:
            body["next"] = nxt
        r = requests.post(C.SH_CATALOG, json=body, headers=ses.headers(), timeout=120)
        r.raise_for_status()
        js = r.json()
        feats += js["features"]
        nxt = js.get("context", {}).get("next")
        if not nxt:
            break

    passes = defaultdict(lambda: {"slices": []})
    for f in feats:
        p = f["properties"]
        if p.get("sar:instrument_mode") != "IW" or p.get("s1:polarization") != "DV" or "GRDH" not in f["id"]:
            continue
        key = (p["sat:orbit_state"], p.get("sat:relative_orbit"), p["datetime"][:10])
        d = passes[key]
        d.update(orbit=p["sat:orbit_state"], track=p.get("sat:relative_orbit"), date=p["datetime"][:10],
                 platform=p.get("platform"))
        d["slices"].append({"id": f["id"], "time": p["datetime"]})

    tracks = defaultdict(lambda: {"pre": [], "post": []})
    for (orb, trk, date), d in sorted(passes.items(), key=lambda kv: kv[0][2]):
        name = f"{orb[:3].upper()}{trk:03d}"
        phase = "pre" if date < C.EVENT_START else ("post" if date > C.EVENT_END else "during")
        if phase == "during":
            continue
        tracks[name][phase].append({k: d[k] for k in ("date", "platform", "orbit", "track", "slices")})

    usable = {k: v for k, v in tracks.items() if len(v["pre"]) >= 4 and v["post"]}
    inv = {"event": [C.EVENT_START, C.EVENT_END], "tracks": tracks, "usable_tracks": sorted(usable)}
    C.write_json(C.DATA / "s1_inventory.json", inv)
    for k, v in sorted(tracks.items()):
        print(f"{k}: {len(v['pre'])} pre ({v['pre'][0]['date'] if v['pre'] else '-'} .. "
              f"{v['pre'][-1]['date'] if v['pre'] else '-'}), {len(v['post'])} post "
              f"{[p['date'] for p in v['post']]}  {'USABLE' if k in usable else 'no post-event pass yet'}")


if __name__ == "__main__":
    main()
