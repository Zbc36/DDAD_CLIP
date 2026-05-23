import os
import torch
from tqdm import tqdm

# ==== 配置 ====
RSNA_BAND_DIR = "/mnt/data/baichuanzhang/DDAD-ASR/data/RSNA_BAND"  # PT 文件根目录
splits = ["train", "test"]

# ==== 检查函数 ====
def check_split(split_dir):
    pt_files = [f for f in os.listdir(split_dir) if f.endswith(".pt")]
    print(f"Checking {split_dir}, total files: {len(pt_files)}")

    nan_files = []
    inf_files = []

    for f in tqdm(pt_files):
        path = os.path.join(split_dir, f)
        try:
            item = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[Error] cannot load {path}: {e}")
            continue

        x = item.get("fused", None)
        if x is None:
            print(f"[Warning] fused not found in {f}")
            continue

        if torch.isnan(x).any():
            nan_files.append(f)
        if torch.isinf(x).any():
            inf_files.append(f)

    return nan_files, inf_files

# ==== 主程序 ====
for split in splits:
    split_dir = os.path.join(RSNA_BAND_DIR, split)
    nan_files, inf_files = check_split(split_dir)

    print(f"\nSplit: {split}")
    print(f"NaN files: {len(nan_files)}")
    if nan_files:
        print(nan_files)
    print(f"Inf files: {len(inf_files)}")
    if inf_files:
        print(inf_files)