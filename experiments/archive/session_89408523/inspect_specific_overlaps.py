from pathlib import Path

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

for pid in ["person78", "person255", "person1000", "person1016"]:
    print(f"\n=== {pid} ===")
    for f in data_dir.rglob(f"{pid}_*"):
        print(f"  {f.relative_to(data_dir)}")
