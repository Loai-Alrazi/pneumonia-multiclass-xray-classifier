from pathlib import Path
import re
from collections import defaultdict

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

person_bact_splits = defaultdict(set)
person_viral_splits = defaultdict(set)

for split in ["train", "val", "test"]:
    for f in (data_dir / split / "BACTERIAL").glob("*.*"):
        m = re.match(r"(person\d+)_bacteria_", f.name)
        if m:
            person_bact_splits[m.group(1)].add(split)
            
    for f in (data_dir / split / "VIRAL").glob("*.*"):
        m = re.match(r"(person\d+)_virus_", f.name)
        if m:
            person_viral_splits[m.group(1)].add(split)

overlapping_pids = set(person_bact_splits.keys()).intersection(set(person_viral_splits.keys()))
print(f"Number of overlapping person IDs between Bacterial and Viral: {len(overlapping_pids)}")

same_split_count = 0
diff_split_count = 0

diff_split_examples = []

for pid in overlapping_pids:
    b_splits = person_bact_splits[pid]
    v_splits = person_viral_splits[pid]
    if b_splits == v_splits:
        same_split_count += 1
    else:
        diff_split_count += 1
        diff_split_examples.append((pid, b_splits, v_splits))

print(f"Overlapping person IDs where Bacterial and Viral are in the SAME split: {same_split_count}")
print(f"Overlapping person IDs where Bacterial and Viral are in DIFFERENT splits: {diff_split_count}")

if diff_split_examples:
    print(f"\nSample cases in DIFFERENT splits (COUNT={len(diff_split_examples)}):")
    for pid, bs, vs in diff_split_examples[:20]:
        print(f"  {pid}: Bacterial in {bs}, Viral in {vs}")
