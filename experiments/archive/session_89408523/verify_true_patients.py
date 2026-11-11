import re
from pathlib import Path
from collections import defaultdict

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

def get_true_patient_id(path_str):
    name = Path(path_str).name
    # Bacterial: personX_bacteria_Y -> patient ID is personX_bacteria
    mb = re.match(r"(person\d+_bacteria)_\d+", name)
    if mb:
        return mb.group(1)
    # Viral: personX_virus_Y -> patient ID is personX_virus
    mv = re.match(r"(person\d+_virus)_\d+", name)
    if mv:
        return mv.group(1)
    # Normal: NORMAL2-IM-0865-0001 or IM-0115-0001
    # The last 4 digits (-0001) are the image number, the prefix is the patient/study ID
    parts = Path(path_str).stem.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:-1])
    return Path(path_str).stem

print("=== AUDIT WITH TRUE PATIENT IDENTIFIERS ===")
all_splits_patients = defaultdict(lambda: defaultdict(set))
all_patients = set()

for split in ["train", "val", "test"]:
    for cls in ["BACTERIAL", "NORMAL", "VIRAL"]:
        folder = data_dir / split / cls
        for f in folder.glob("*.*"):
            pid = get_true_patient_id(f.name)
            all_splits_patients[split][cls].add(pid)
            all_patients.add(pid)

for split in ["train", "val", "test"]:
    print(f"\nSplit: {split}")
    total_split_patients = set()
    for cls in ["BACTERIAL", "NORMAL", "VIRAL"]:
        pids = all_splits_patients[split][cls]
        total_split_patients.update(pids)
        print(f"  {cls}: {len(pids)} unique patients")
    print(f"  Sum of per-class patients: {sum(len(all_splits_patients[split][c]) for c in ['BACTERIAL', 'NORMAL', 'VIRAL'])}")
    print(f"  Total unique patients in split: {len(total_split_patients)}")
    assert sum(len(all_splits_patients[split][c]) for c in ['BACTERIAL', 'NORMAL', 'VIRAL']) == len(total_split_patients), "Still a mismatch!"

print(f"\nTotal unique patients across ALL splits: {len(all_patients)}")

# Now check cross-split patient overlap with TRUE patient IDs!
train_pats = set.union(*all_splits_patients["train"].values())
val_pats = set.union(*all_splits_patients["val"].values())
test_pats = set.union(*all_splits_patients["test"].values())

print("\nCross-split overlap check with TRUE patient IDs:")
print(f"Train vs Val overlap:  {len(train_pats.intersection(val_pats))}")
print(f"Train vs Test overlap: {len(train_pats.intersection(test_pats))}")
print(f"Val vs Test overlap:   {len(val_pats.intersection(test_pats))}")
