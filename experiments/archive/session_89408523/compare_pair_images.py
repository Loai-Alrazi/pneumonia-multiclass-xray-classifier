from PIL import Image
import numpy as np
from pathlib import Path

data_dir = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

# Let's inspect a few pairs of (personX_bacteria, personX_virus)
pairs_to_check = ["person78", "person255", "person1000", "person1016"]

for pid in pairs_to_check:
    b_files = list(data_dir.rglob(f"{pid}_bacteria_*"))
    v_files = list(data_dir.rglob(f"{pid}_virus_*"))
    print(f"\nComparing {pid}:")
    b_img = Image.open(b_files[0])
    v_img = Image.open(v_files[0])
    b_arr = np.array(b_img)
    v_arr = np.array(v_img)
    print(f"  Bacterial: {b_files[0].name}, shape={b_arr.shape}, mean={b_arr.mean():.2f}, std={b_arr.std():.2f}")
    print(f"  Viral:     {v_files[0].name}, shape={v_arr.shape}, mean={v_arr.mean():.2f}, std={v_arr.std():.2f}")
