from os.path import join
import cv2
import numpy as np
from skimage.restoration import inpaint_biharmonic
import multiprocessing as mp
from multiprocessing import Pool

from src.utils import make_dir
from mosaic.mosaic_api import SaltData
from src.config import DATA_DIR, TRAIN_DIR, TEST_DIR
from src.transforms import CenterCrop


ORIG_SIZE = (101, 101)
SAVE_SIZE = (148, 148)
CENTER_CROP = CenterCrop(SAVE_SIZE)
SAVE_NAME = '_pazzles_6013_148'
TARGET_THRESHOLD = 0.7
N_WORKERS = mp.cpu_count()

make_dir(join(TRAIN_DIR, 'images' + SAVE_NAME))
make_dir(join(TRAIN_DIR, 'masks' + SAVE_NAME))
make_dir(join(TEST_DIR, 'images' + SAVE_NAME))

IMAGES_DIR_NAME = "images_hist_mean"
MASKS_DIR_NAME = "masks"
MOSAIC_CSV_PATH = join(DATA_DIR, "mosaic", "pazzles_6013.csv")
SALT_DATA = SaltData(images_dir_name=IMAGES_DIR_NAME,
                     masks_dir_name=MASKS_DIR_NAME,
                     mosaic_csv_path=MOSAIC_CSV_PATH)


def inpaint_train(id):
    neighbors = SALT_DATA.get_neighbors(id)
    stacked_image, image_unknown_mask = SALT_DATA.get_stacked_images(neighbors, return_unknown_mask=True)
    center_stacked_image = CENTER_CROP(stacked_image)
    center_image_unknown_mask = CENTER_CROP(image_unknown_mask)
    if np.sum(center_image_unknown_mask):
        inpaint_image = inpaint_biharmonic(center_stacked_image, center_image_unknown_mask)
        inpaint_image = (inpaint_image * 255).astype(np.uint8)
    else:
        inpaint_image = center_stacked_image

    stacked_mask, mask_unknown_mask = SALT_DATA.get_stacked_masks(neighbors, return_unknown_mask=True)
    center_stacked_mask = CENTER_CROP(stacked_mask)
    center_mask_unknown_mask = CENTER_CROP(mask_unknown_mask)
    if np.sum(center_mask_unknown_mask):
        inpaint_mask = inpaint_biharmonic(center_stacked_mask, center_mask_unknown_mask)
        inpaint_mask= np.where(inpaint_mask > TARGET_THRESHOLD, 255, 0)
    else:
        inpaint_mask = center_stacked_mask

    cv2.imwrite(join(TRAIN_DIR, 'images' + SAVE_NAME, f"{id}.png"), inpaint_image)
    cv2.imwrite(join(TRAIN_DIR, 'masks' + SAVE_NAME, f"{id}.png"), inpaint_mask)


def inpaint_test(id):
    neighbors = SALT_DATA.get_neighbors(id)
    stacked_image, image_unknown_mask = SALT_DATA.get_stacked_images(neighbors, return_unknown_mask=True)
    center_stacked_image = CENTER_CROP(stacked_image)
    center_image_unknown_mask = CENTER_CROP(image_unknown_mask)
    if np.sum(center_image_unknown_mask):
        inpaint_image = inpaint_biharmonic(center_stacked_image, center_image_unknown_mask)
        inpaint_image = (inpaint_image * 255).astype(np.uint8)
    else:
        inpaint_image = center_stacked_image

    cv2.imwrite(join(TEST_DIR, 'images' + SAVE_NAME, f"{id}.png"), inpaint_image)


if __name__ == '__main__':
    # Train
    print('Start train inpaint')
    with Pool(processes=N_WORKERS) as pool:
        pool.map(inpaint_train, list(SALT_DATA.train_ids))

    # Test
    print('Start test inpaint')
    with Pool(processes=N_WORKERS) as pool:
        pool.map(inpaint_test, list(SALT_DATA.test_ids))

    print('Inpaint complete')
