from pathlib import Path
import re
from collections import defaultdict

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

bact_by_pid = defaultdict(list)
viral_by_pid = defaultdict(list)

# Single pass over all files
for p in data_dir.rglob("*.jpeg"):
    m = re.match(r"(person\d+)_([a-z]+)_(\d+)", p.name)
    if m:
        pid, subtype, img_id = m.groups()
        if subtype == "bacteria":
            bact_by_pid[pid].append(p.name)
        elif subtype == "virus":
            viral_by_pid[pid].append(p.name)

common_pids = set(bact_by_pid.keys()).intersection(set(viral_by_pid.keys()))
print(f"Total common person IDs: {len(common_pids)}")

# Sort by numeric ID
sorted_pids = sorted(list(common_pids), key=lambda x: int(x.replace("person", "")))
print("\nFirst 15 common person IDs:")
for pid in sorted_pids[:15]:
    print(f"{pid}:")
    print(f"  bacteria: {bact_by_pid[pid]}")
    print(f"  virus:    {viral_by_pid[pid]}")
