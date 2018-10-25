from os.path import join
import pandas as pd
import numpy as np
import random
import cv2

from src import config


def get_correct_train_ids(train_csv_path, train_dir):
    train_df = pd.read_csv(train_csv_path, index_col='id')
    train_df.fillna('', inplace=True)
    correct_ids = []

    for index, row in train_df.iterrows():
        image_path = join(train_dir, 'images', index + '.png')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        pixel_sum = np.sum(image)
        if pixel_sum:
            correct_ids.append(index)
    return correct_ids


def make_train_folds(train_csv_path, train_dir, depths_path, n_folds):
    depths_df = pd.read_csv(depths_path, index_col='id')
    train_ids = get_correct_train_ids(train_csv_path, train_dir)
    depths_df = depths_df.loc[train_ids]
    depths_df['image_path'] = depths_df.index.map(
        lambda x: join(train_dir, 'images', x + '.png'))
    depths_df['mask_path'] = depths_df.index.map(
        lambda x: join(train_dir, 'masks', x + '.png'))
    depths_df.sort_values('z', inplace=True)
    depths_df['fold'] = [i % n_folds for i in range(depths_df.shape[0])]
    depths_df.sort_index(inplace=True)
    return depths_df


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    train_folds_df = make_train_folds(config.TRAIN_CSV_PATH,
                                      config.TRAIN_DIR,
                                      config.DEPTHS_PATH,
                                      config.N_FOLDS)
    train_folds_df.to_csv(config.TRAIN_FOLDS_PATH)
    print(f"Folds saved to {config.TRAIN_FOLDS_PATH}")
