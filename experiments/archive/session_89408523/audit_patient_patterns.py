from pathlib import Path
import re
from collections import defaultdict, Counter

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

print("=== DEEP AUDIT OF PATIENT IDENTIFIERS ===")

# Examine all filename patterns across all splits and classes
patterns = Counter()
files_by_pattern = defaultdict(list)

for split in ["train", "val", "test"]:
    for cls in ["BACTERIAL", "NORMAL", "VIRAL"]:
        folder = data_dir / split / cls
        for f in folder.glob("*.*"):
            stem = f.stem
            if stem.startswith("person"):
                # personXXXX_bacteria_YYYY or personXXXX_virus_YYYY
                parts = stem.split("_")
                patterns["person_prefix"] += 1
            elif "IM-" in stem:
                patterns["IM_prefix"] += 1
            else:
                patterns["other"] += 1
                files_by_pattern["other"].append(stem)

print("Filename patterns found:", dict(patterns))
if files_by_pattern["other"]:
    print("Sample 'other' filenames:", files_by_pattern["other"][:10])
