import glob
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import cv2
import numpy as np
from joblib import Parallel, delayed
from shapely.wkt import loads
from tqdm import tqdm

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", type=str, required=True, help="Path for saving preprocessed data")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs")


class Converter:
    def __init__(self, args):
        self.args = args
        self.pre = self.load_jsons("pre")
        self.post = self.load_jsons("post")
        self.save_path = os.path.join(args.data, "targets")
        os.makedirs(self.save_path, exist_ok=True)
        self.damage_dict = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4, "un-classified": 255}

    def load_jsons(self, mode):
        return sorted(glob.glob(os.path.join(self.args.data, "labels", f"*{mode}*")))

    def run(self):
        self.run_parallel(self.pre, "pre")
        self.run_parallel(self.post, "post")

    def run_parallel(self, files, mode):
        Parallel(n_jobs=self.args.n_jobs)(
            delayed(self.convert_label)(json_file, mode) for json_file in tqdm(files, total=len(files))
        )

    def convert_label(self, json_file, mode):
        fname = os.path.basename(json_file).replace(".json", ".png")
        json_file = json.load(open(json_file))
        mask = np.zeros((1024, 1024), dtype=np.uint8)

        for feat in json_file["features"]["xy"]:
            poly = loads(feat["wkt"])
            features_mask = np.zeros((1024, 1024), dtype=np.uint8)
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            exteriors = [int_coords(poly.exterior.coords)]
            cv2.fillPoly(features_mask, exteriors, 1)

            if mode == "pre":
                mask[features_mask > 0] = 1
            else:
                subtype = feat["properties"]["subtype"]
                mask[features_mask > 0] = self.damage_dict[subtype]

        cv2.imwrite(os.path.join(self.save_path, fname), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == "__main__":
    Converter(parser.parse_args()).run()
