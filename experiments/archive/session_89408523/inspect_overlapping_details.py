from pathlib import Path
import re

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

overlapping_pids = []
for p in range(1, 2000):
    b = list(data_dir.rglob(f"person{p}_bacteria_*"))
    v = list(data_dir.rglob(f"person{p}_virus_*"))
    if b and v:
        overlapping_pids.append((p, [x.name for x in b], [x.name for x in v]))

print(f"Total overlapping person IDs: {len(overlapping_pids)}")
print("\nFirst 10 overlapping person IDs:")
for p, b, v in overlapping_pids[:10]:
    print(f"person{p}:")
    print(f"  bacteria: {b}")
    print(f"  virus:    {v}")
