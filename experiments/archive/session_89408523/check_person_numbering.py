import re
from pathlib import Path
import numpy as np

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

bact_pnums = []
viral_pnums = []

for f in data_dir.rglob("*_bacteria_*"):
    m = re.match(r"person(\d+)_bacteria_", f.name)
    if m:
        bact_pnums.append(int(m.group(1)))

for f in data_dir.rglob("*_virus_*"):
    m = re.match(r"person(\d+)_virus_", f.name)
    if m:
        viral_pnums.append(int(m.group(1)))

bact_set = set(bact_pnums)
viral_set = set(viral_pnums)

print(f"Bacterial person IDs: min={min(bact_pnums)}, max={max(bact_pnums)}, unique={len(bact_set)}")
print(f"Viral person IDs:     min={min(viral_pnums)}, max={max(viral_pnums)}, unique={len(viral_set)}")
print(f"Intersection of person IDs: {len(bact_set.intersection(viral_set))}")

# Check if person IDs in Kermany were independently indexed:
# If bacterial was person1..person1700, and viral was person1..person1700, then they are INDEPENDENT IDENTIFIERS!
# That is: person100_bacteria is Patient #100 in the bacterial cohort, while person100_virus is Patient #100 in the viral cohort!
print("\nChecking if bacterial and viral have independent sequential person numbering:")
print(f"Bacterial IDs <= 100: {len([x for x in bact_set if x <= 100])} out of 100")
print(f"Viral IDs <= 100:     {len([x for x in viral_set if x <= 100])} out of 100")

print(f"Bacterial IDs <= 500: {len([x for x in bact_set if x <= 500])} out of 500")
print(f"Viral IDs <= 500:     {len([x for x in viral_set if x <= 500])} out of 500")

print(f"Bacterial IDs <= 1000: {len([x for x in bact_set if x <= 1000])} out of 1000")
print(f"Viral IDs <= 1000:     {len([x for x in viral_set if x <= 1000])} out of 1000")
