from os.path import join
import numpy as np
import pandas as pd
import cv2

from src import config
from src.utils import RLenc, make_dir

PREDICTION_DIRS = [
    '/workdir/data/predictions/mos-fpn-lovasz-se-resnext50-001/'
]
FOLDS = [0, 1, 2, 3, 4, 5]
FOLD_DIRS = [join(p, 'fold_%d'%f) for p in PREDICTION_DIRS for f in FOLDS]

PREDICTION_DIRS = [
    '/workdir/data/predictions/fpn-lovasz-se-resnext50-006-after-001/',
]

FOLDS = [0, 1, 2, 3, 4]
FOLD_DIRS += [join(p, 'fold_%d'%f) for p in PREDICTION_DIRS for f in FOLDS]

segm_thresh = 0.4
prob_thresh = 0.5

SAVE_NAME = 'mean-005-0.4'
MEAN_PREDICTION_DIR = f'/workdir/data/predictions/{SAVE_NAME}'

make_dir(join(MEAN_PREDICTION_DIR, 'masks'))


def get_mean_probs_df():
    probs_df_lst = []
    for fold_dir in FOLD_DIRS:
        probs_df = pd.read_csv(join(fold_dir, 'test', 'probs.csv'), index_col='id')
        probs_df_lst.append(probs_df)

    mean_probs_df = probs_df_lst[0].copy()
    for probs_df in probs_df_lst[1:]:
        mean_probs_df.prob += probs_df.prob
    mean_probs_df.prob /= len(probs_df_lst)

    return mean_probs_df


if __name__ == "__main__":
    print('Make average submission', FOLD_DIRS)

    mean_probs_df = get_mean_probs_df()
    sample_submition = pd.read_csv(config.SAMPLE_SUBM_PATH)

    for i, row in sample_submition.iterrows():
        pred_name = row.id + '.png'
        pred_lst = []
        for fold_dir in FOLD_DIRS:
            pred_path = join(fold_dir, 'test', pred_name)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            pred = pred / 255
            pred_lst.append(pred)

        mean_pred = np.mean(pred_lst, axis=0)
        prob = mean_probs_df.loc[row.id].prob

        pred = mean_pred > segm_thresh
        prob = int(prob > prob_thresh)
        pred = (pred * prob).astype(np.uint8)

        if np.all(pred == 1):
            pred[:] = 0
            print('Full mask to empty', pred_name)

        rle_mask = RLenc(pred)
        cv2.imwrite(join(MEAN_PREDICTION_DIR, 'masks', pred_name), pred * 255)
        row.rle_mask = rle_mask

    sample_submition.to_csv(join(MEAN_PREDICTION_DIR, f'{SAVE_NAME}.csv'), index=False)
