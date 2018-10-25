import os
import cv2
import tqdm
import numpy as np

from src.config import MEAN_HIST_PATH, TRAIN_DIR, TEST_DIR

img_train_dir = os.path.join(TRAIN_DIR, 'images')
img_test_dir = os.path.join(TEST_DIR, 'images')

images_names = os.listdir(img_train_dir)
images_names += os.listdir(img_test_dir)

n_samples = len(images_names)

images_paths = []
for img_name in images_names:
    train_path = os.path.join(img_train_dir, img_name)
    test_path = os.path.join(img_test_dir, img_name)
    if os.path.exists(train_path):
        images_paths.append(train_path)
    elif os.path.exists(test_path):
        images_paths.append(test_path)

# Check if everything is ok
print("Samples:", n_samples, "Ok:", len(images_paths)==len(images_names))

imgs = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        for img_path in images_paths]

hist = np.zeros(256, dtype=np.int64)

for img in tqdm.tqdm(imgs, desc="Create histograms"):
    if np.sum(img):
        values, counts = np.unique(img, return_counts=True)
        for (v, c) in zip(values, counts):
            hist[v] += c

np.save(MEAN_HIST_PATH, hist/len(images_names))
print("Mean histogram saved!")
