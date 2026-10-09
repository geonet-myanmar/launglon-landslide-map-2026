"""Run the whole Launglon Sentinel-1 landslide workflow in order. Every step caches its downloads, so a re-run
only fetches what is new (for example a later post-event pass, which step 01 picks up automatically)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ["01_s1_inventory.py"],
    ["02_s1_rtc_download.py"],
    ["03_ancillary.py"],
    ["04_features.py"],
    ["04_features.py", "placebo", "20260815"],
    ["04_features.py", "placebo", "20260827"],
    ["04_features.py", "placebo", "20260908"],
    ["04_features.py", "placebo", "20260920"],
    ["05_labels.py"],
    ["06_vectors.py"],
    ["07_detect.py"],
    ["08_analysis.py"],
    ["09_dashboard.py"],
]

for step in STEPS:
    if step[0] == "04_features.py" and len(step) > 1:
        tag = f"ASC070_placebo_{step[2]}.tif"
        if (HERE.parent / "data" / "features" / tag).exists():
            continue
    print("==>", " ".join(step), flush=True)
    subprocess.run([sys.executable, str(HERE / step[0]), *step[1:]], check=True, cwd=HERE)
