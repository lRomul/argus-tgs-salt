import glob
import cv2
import tqdm
import numpy as np
from os.path import join
from mosaic.hist_match import hist_match
from src.config import MEAN_HIST_PATH, TRAIN_DIR, TEST_DIR
from src.utils import make_dir

IMAGES_DIR_NAME = 'images_hist_mean'


mean_hist = np.load(MEAN_HIST_PATH)

images_paths = glob.glob(join(TRAIN_DIR, 'images', '*.png'))
images_paths += glob.glob(join(TEST_DIR, 'images', '*.png'))

make_dir(join(TEST_DIR, IMAGES_DIR_NAME))
make_dir(join(TRAIN_DIR, IMAGES_DIR_NAME))


for image_path in tqdm.tqdm(images_paths):
    image = cv2.imread(image_path)
    hist_image = hist_match(image, mean_hist)
    hist_image = np.round(hist_image).astype(np.uint8)
    save_image_path = image_path.replace('/images/', f'/{IMAGES_DIR_NAME}/')
    cv2.imwrite(save_image_path, hist_image)
