from pathlib import Path
import re
from collections import defaultdict

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

# Let's inspect the distribution of person IDs and IM IDs
bact_persons = set()
viral_persons = set()
norm_ids = set()

for split in ["train", "val", "test"]:
    # Bacterial
    for f in (data_dir / split / "BACTERIAL").glob("*.*"):
        m = re.match(r"(person\d+)_bacteria_(\d+)", f.name)
        if m:
            bact_persons.add(m.group(1))
        else:
            print("Unmatched bacterial:", f.name)
            
    # Viral
    for f in (data_dir / split / "VIRAL").glob("*.*"):
        m = re.match(r"(person\d+)_virus_(\d+)", f.name)
        if m:
            viral_persons.add(m.group(1))
        else:
            print("Unmatched viral:", f.name)

    # Normal
    for f in (data_dir / split / "NORMAL").glob("*.*"):
        stem = f.stem
        # check patterns
        norm_ids.add(stem)

print(f"Total unique person IDs in BACTERIAL: {len(bact_persons)}")
print(f"Total unique person IDs in VIRAL:     {len(viral_persons)}")
overlap_bact_viral = bact_persons.intersection(viral_persons)
print(f"Overlap between BACTERIAL person IDs and VIRAL person IDs: {len(overlap_bact_viral)}")
print(f"Sample overlapping IDs: {sorted(list(overlap_bact_viral), key=lambda x: int(x.replace('person','')))[:20]}")

# Check image indices for overlapping person IDs
print("\nInspecting sample overlapping person IDs:")
for pid in sorted(list(overlap_bact_viral), key=lambda x: int(x.replace('person','')))[:10]:
    b_files = list(data_dir.rglob(f"{pid}_bacteria_*"))
    v_files = list(data_dir.rglob(f"{pid}_virus_*"))
    print(f"{pid}:")
    print(f"  Bacterial files: {[p.name for p in b_files]}")
    print(f"  Viral files:     {[p.name for p in v_files]}")
