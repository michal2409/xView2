from sklearn.model_selection import train_test_split
from shutil import copy
import tqdm
import glob
import os

test_size = 0.2
seed = 100

train, val = [], []
for dname in glob.glob(os.path.join('/scidatasm', 'michalf', 'train1/*')):
    tr, va = train_test_split(glob.glob(os.path.join(dname, 'masks/*')), test_size=test_size, random_state=seed)
    train = train + tr
    val = val + va

train_dict = os.path.join('/scidatasm', 'michalf', 'training')
val_dict = os.path.join('/scidatasm', 'michalf', 'validation')

if not os.path.isdir(train_dict):
    os.makedirs(train_dict)
    os.makedirs(os.path.join(train_dict, 'masks'))
    os.makedirs(os.path.join(train_dict, 'labels'))
    os.makedirs(os.path.join(train_dict, 'images'))

if not os.path.isdir(val_dict):
    os.makedirs(val_dict)
    os.makedirs(os.path.join(val_dict, 'masks'))
    os.makedirs(os.path.join(val_dict, 'labels'))
    os.makedirs(os.path.join(val_dict, 'images'))

for mask_path in tqdm.tqdm(train):
    copy(mask_path, os.path.join(train_dict, 'masks'))
    copy(mask_path.replace('masks', 'labels').replace('png', 'json'), os.path.join(train_dict, 'labels'))
    copy(mask_path.replace('masks', 'images'), os.path.join(train_dict, 'images'))

for mask_path in tqdm.tqdm(val):
    copy(mask_path, os.path.join(val_dict, 'masks'))
    copy(mask_path.replace('masks', 'labels').replace('png', 'json'), os.path.join(val_dict, 'labels'))
    copy(mask_path.replace('masks', 'images'), os.path.join(val_dict, 'images'))
