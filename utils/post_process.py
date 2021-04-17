import os
from glob import glob
from subprocess import call

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_fill_holes
from tqdm import tqdm


def make_empty_dir(path):
    call(["rm", "-rf", path])
    os.makedirs(path)


def save(img, dir, fname):
    Image.fromarray(img.astype(np.uint8)).save(os.path.join("/results", dir, fname))


def post_process(pre_path, post_path):
    pre = np.zeros((1024, 1024))
    loc, dmg = np.load(pre_path), np.load(post_path)

    if dmg.shape[0] == 4:
        dmg[0] *= 0.65
        post = np.argmax(dmg, axis=0) + 1
    else:
        post = dmg
    idx = np.logical_or(loc > 0.3, np.logical_and(loc > 0.1, post > 1))
    pre[idx] = 1
    pre = binary_fill_holes(pre).astype(np.uint8)

    post = post * pre
    components, n = label(post > 0)
    for building_idx in range(1, n + 1):
        labels, counts = np.unique(post[components == building_idx], return_counts=True)
        post[components == building_idx] = labels[np.argmax(counts)]

    save(pre, "predictions", "".join(os.path.basename(pre_path).replace(".npy", "_prediction.png")))
    save(post, "predictions", "".join(os.path.basename(post_path).replace(".npy", "_prediction.png")))


if __name__ == "__main__":
    pred_dir = os.path.join("/results", "predictions")
    if not os.path.exists(pred_dir):
        make_empty_dir(pred_dir)

    pre_pred = sorted(glob("/results/probs/*localization*"))
    post_pred = sorted(glob("/results/probs/*damage*"))
    Parallel(n_jobs=-1)(
        delayed(post_process)(pre_path, post_path)
        for pre_path, post_path in tqdm(zip(pre_pred, post_pred), total=len(pre_pred))
    )
