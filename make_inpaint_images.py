import os
from os.path import join
import cv2
import numpy as np
from skimage.restoration import inpaint_biharmonic
import multiprocessing as mp
from multiprocessing import Pool

from src.utils import make_dir
from src.config import TRAIN_DIR
from src.config import TEST_DIR


ORIG_SIZE = (101, 101)
SAVE_SIZE = (148, 148)
SAVE_NAME = '148'
TARGET_THRESHOLD = 0.7 
N_WORKERS = mp.cpu_count()


make_dir(join(TRAIN_DIR, 'images'+SAVE_NAME))
make_dir(join(TRAIN_DIR, 'masks'+SAVE_NAME))
make_dir(join(TEST_DIR, 'images'+SAVE_NAME))

diff = (SAVE_SIZE - np.array(ORIG_SIZE))
pad_left = diff // 2
pad_right = diff - pad_left
PAD_WIDTH = ((pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]))
print('Pad width:', PAD_WIDTH)

MASK_INPAINT = np.zeros(ORIG_SIZE, dtype=np.uint8)
MASK_INPAINT = np.pad(MASK_INPAINT, PAD_WIDTH, mode='constant', constant_values=255)


def inpaint_train(img_file):
    img = cv2.imread(join(TRAIN_DIR, 'images', img_file), cv2.IMREAD_GRAYSCALE)
    trg = cv2.imread(join(TRAIN_DIR, 'masks', img_file), cv2.IMREAD_GRAYSCALE)
    img = np.pad(img, PAD_WIDTH, mode='constant')
    trg = np.pad(trg, PAD_WIDTH, mode='constant')
    img = (inpaint_biharmonic(img, MASK_INPAINT)*255).astype(np.uint8)
    trg = inpaint_biharmonic(trg, MASK_INPAINT)
    trg = np.where(trg > TARGET_THRESHOLD, 255, 0)
    cv2.imwrite(join(TRAIN_DIR, 'images'+SAVE_NAME, img_file), img)
    cv2.imwrite(join(TRAIN_DIR, 'masks'+SAVE_NAME, img_file), trg)
    

def inpaint_test(img_file):
    img = cv2.imread(join(TEST_DIR, 'images', img_file), cv2.IMREAD_GRAYSCALE)
    img = np.pad(img, PAD_WIDTH, mode='constant')
    img = (inpaint_biharmonic(img, MASK_INPAINT)*255).astype(np.uint8)
    cv2.imwrite(join(TEST_DIR, 'images'+SAVE_NAME, img_file), img)
    

if __name__ == '__main__':

    # Train
    print('Start train inpaint')
    with Pool(processes=N_WORKERS) as pool:
        pool.map(inpaint_train, os.listdir(join(TRAIN_DIR, 'images')))
    
    # Test 
    print('Start test inpaint')
    with Pool(processes=N_WORKERS) as pool:
        pool.map(inpaint_test, os.listdir(join(TEST_DIR, 'images')))

    print('Inpaint complete')
