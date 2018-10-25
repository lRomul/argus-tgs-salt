import os
from os.path import join
import cv2
import numpy as np
import pandas as pd
from skimage.restoration import inpaint_biharmonic
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
from itertools import combinations
from random import shuffle

from src.utils import make_dir
from src import config
from src.config import TRAIN_DIR


ORIG_SIZE = (101, 101)
SAVE_SIZE = (109, 109)
SAVE_NAME = '_double_109'
TRAIN_FOLDS_PATH = '/workdir/data/train_folds_148.csv'
TARGET_THRESHOLD = 0.7 
N_WORKERS = mp.cpu_count()


make_dir(join(TRAIN_DIR, 'images'+SAVE_NAME))
make_dir(join(TRAIN_DIR, 'masks'+SAVE_NAME))

diff = (SAVE_SIZE - np.array(ORIG_SIZE))
pad_left = diff // 2
pad_right = diff - pad_left
PAD_WIDTH = ((pad_left[0], pad_right[0]), (pad_left[0], pad_right[1]))
print('Pad width:', PAD_WIDTH)

MASK_INPAINT = np.zeros(ORIG_SIZE, dtype=np.uint8)
MASK_INPAINT = np.pad(MASK_INPAINT, PAD_WIDTH, mode='constant', constant_values=255)
MASK_INPAINT = np.concatenate([MASK_INPAINT, MASK_INPAINT], axis=1)


def inpaint_double_train(img_files):
    img_file_1, img_file_2, fold = img_files
    img_1 = cv2.imread(join(TRAIN_DIR, 'images', img_file_1), cv2.IMREAD_GRAYSCALE)
    trg_1 = cv2.imread(join(TRAIN_DIR, 'masks', img_file_1), cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread(join(TRAIN_DIR, 'images', img_file_2), cv2.IMREAD_GRAYSCALE)
    trg_2 = cv2.imread(join(TRAIN_DIR, 'masks', img_file_2), cv2.IMREAD_GRAYSCALE)
    
    if not np.sum(img_1) or not np.sum(img_2):
        return 
    
    img_1 = np.pad(img_1, PAD_WIDTH, mode='constant')
    trg_1 = np.pad(trg_1, PAD_WIDTH, mode='constant')
    img_2 = np.pad(img_2, PAD_WIDTH, mode='constant')
    trg_2 = np.pad(trg_2, PAD_WIDTH, mode='constant')
    img = np.concatenate([img_1, img_2], axis=1)
    trg = np.concatenate([trg_1, trg_2], axis=1)
    img = (inpaint_biharmonic(img, MASK_INPAINT)*255).astype(np.uint8)
    trg = inpaint_biharmonic(trg, MASK_INPAINT)
    trg = np.where(trg > TARGET_THRESHOLD, 255, 0)

    img_file = img_file_1[:-4] + '_' + img_file_2[:-4] + '.png'
    make_dir(join(TRAIN_DIR, 'images' + SAVE_NAME, 'fold_%d'%fold))
    make_dir(join(TRAIN_DIR, 'masks'+SAVE_NAME, 'fold_%d'%fold))
    cv2.imwrite(join(TRAIN_DIR, 'images'+SAVE_NAME, 'fold_%d'%fold, img_file), img)
    cv2.imwrite(join(TRAIN_DIR, 'masks'+SAVE_NAME, 'fold_%d'%fold, img_file), trg)
    return img, trg, img_file


if __name__ == '__main__':

    train_folds_df = pd.read_csv(TRAIN_FOLDS_PATH)
    fold2names = defaultdict(list)
    for i, row in train_folds_df.iterrows():
        name = os.path.basename(row.image_path)
        fold2names[row.fold].append(name)

    double_image_lst = []

    for fold in range(config.N_FOLDS):
        print(fold)
        fold_names = list(combinations(fold2names[fold], 2))
        shuffle(fold_names)
        for name1, name2 in fold_names[:20000]:
            double_image_lst.append((name1, name2, fold))

    shuffle(double_image_lst)
    
    with Pool(processes=N_WORKERS) as pool:
        pool.map(inpaint_double_train, double_image_lst)
    
    print('Inpaint complete')
