import glob
import json

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

PATH = "/data/train"
imgs_post = sorted(glob.glob(f"{PATH}/images/*post*"))
imgs_pre = sorted(glob.glob(f"{PATH}/images/*pre*"))
lbls_post = sorted(glob.glob(f"{PATH}/targets/*post*"))
lbls_pre = sorted(glob.glob(f"{PATH}/targets/*pre*"))
exclude_idx = json.load(open("exclude.txt", "r"))


def get_foreground(img_pre, img_post):
    h_pre, w_pre, _ = np.where(img_pre > 0)
    h_post, w_post, _ = np.where(img_post > 0)
    min_h, max_h = max(min(h_pre), min(h_post)), min(max(h_pre), max(h_post))
    min_w, max_w = max(min(w_pre), min(w_post)), min(max(w_pre), max(w_post))
    return np.s_[min_h:max_h, min_w:max_w]


def get_row(idx):
    if idx in exclude_idx:
        return []
    img_post, img_pre = cv2.imread(imgs_post[idx]), cv2.imread(imgs_pre[idx])
    img_post = img_post[get_foreground(img_pre, img_post)]
    if img_post.shape[0] < 512 or img_post.shape[1] < 512:
        return []
    row = {"idx": idx, "1": 0, "2": 0, "3": 0, "4": 0}
    classes = list(np.unique(cv2.imread(lbls_post[idx], cv2.IMREAD_UNCHANGED)))
    for cls_ in [1, 2, 3, 4]:
        if cls_ in classes:
            row[str(cls_)] = 1
    return row


if __name__ == "__main__":
    N = len(imgs_post)
    rows = Parallel(n_jobs=-1)(delayed(get_row)(idx) for idx in tqdm(range(N), total=N))
    rows = list(filter(None, rows))
    df = pd.DataFrame(rows)
    df.to_csv("/workspace/xview2/utils/index.csv", index=False)
