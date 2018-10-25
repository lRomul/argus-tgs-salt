import cv2
import numpy as np
import pandas as pd
from os.path import join

from src import config
from src.utils import RLenc, make_dir
from pipeline.mean_submission import MEAN_PREDICTION_DIR
from mosaic.postprocess import postprocess

mean_path = join(MEAN_PREDICTION_DIR, 'masks')
postprocess_path = join(MEAN_PREDICTION_DIR, 'postprocessed')

make_dir(postprocess_path)


if __name__ == "__main__":
    print('Make postprocessed submission')
    
    postprocess(mean_path, postprocess_path)

    sample_submition = pd.read_csv(config.SAMPLE_SUBM_PATH)
    for i, row in sample_submition.iterrows():
        pred_name = row.id+'.png'
        pred = cv2.imread(join(postprocess_path, pred_name),
                          cv2.IMREAD_GRAYSCALE)
        pred = pred > 0
        rle_mask = RLenc(pred.astype(np.uint8))
        row.rle_mask = rle_mask

    sample_submition.to_csv(join(MEAN_PREDICTION_DIR,
                            'submission.csv'), index=False)
