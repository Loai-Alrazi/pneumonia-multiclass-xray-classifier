from pathlib import Path
import re
from collections import defaultdict

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

bact_files = list(data_dir.rglob("*_bacteria_*"))
viral_files = list(data_dir.rglob("*_virus_*"))

print(f"Total bacterial files: {len(bact_files)}")
print(f"Total viral files:     {len(viral_files)}")

# Look at person1, person2, person3...
for i in range(1, 15):
    b = [f.name for f in bact_files if f.name.startswith(f"person{i}_bacteria_")]
    v = [f.name for f in viral_files if f.name.startswith(f"person{i}_virus_")]
    print(f"person{i}:")
    print(f"  bacteria ({len(b)}): {b}")
    print(f"  virus    ({len(v)}): {v}")
