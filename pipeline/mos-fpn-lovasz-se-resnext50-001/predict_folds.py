import os
from os.path import join
import re
import cv2
import numpy as np
import pandas as pd

from argus import load_model

import torch

from src.transforms import SimpleDepthTransform, SaltTransform, CenterCrop
from src.argus_models import SaltMetaModel
from src.transforms import HorizontalFlip
from src.utils import RLenc, make_dir
from src import config

EXPERIMENT_NAME = 'mos-fpn-lovasz-se-resnext50-001'
FOLDS = list(range(config.N_FOLDS))
ORIG_IMAGE_SIZE = (101, 101)
PRED_IMAGE_SIZE = (128, 128)
TRANSFORM_MODE = 'crop'
TRAIN_FOLDS_PATH = '/workdir/data/train_folds_148_mos_emb_1.csv'
IMAGES_NAME = '148'
FOLDS_DIR = f'/workdir/data/experiments/{EXPERIMENT_NAME}'
PREDICTION_DIR = f'/workdir/data/predictions/{EXPERIMENT_NAME}'
make_dir(PREDICTION_DIR)

SEGM_THRESH = 0.5
PROB_THRESH = 0.5


class Predictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.model.nn_module.final = torch.nn.Sigmoid()
        self.model.nn_module.eval()

        self.depth_trns = SimpleDepthTransform()
        self.crop_trns = CenterCrop(ORIG_IMAGE_SIZE)
        self.trns = SaltTransform(PRED_IMAGE_SIZE, False, TRANSFORM_MODE)

    def __call__(self, image):
        tensor = self.depth_trns(image, 0)
        tensor = self.trns(tensor)
        tensor = tensor.unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            segm, prob = self.model.nn_module(tensor)
            segm = segm.cpu().numpy()[0][0]
            segm = self.crop_trns(segm)

            segm = (segm * 255).astype(np.uint8)
            prob = prob.item()

            return segm, prob


def pred_val_fold(model_path, fold):
    predictor = Predictor(model_path)
    folds_df = pd.read_csv(TRAIN_FOLDS_PATH)
    fold_df = folds_df[folds_df.fold == fold]
    fold_prediction_dir = join(PREDICTION_DIR, f'fold_{fold}', 'val')
    make_dir(fold_prediction_dir)

    prob_dict = {'id': [], 'prob': []}
    for i, row in fold_df.iterrows():
        image = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
        segm, prob = predictor(image)
        segm_save_path = join(fold_prediction_dir, row.id + '.png')
        cv2.imwrite(segm_save_path, segm)

        prob_dict['id'].append(row.id)
        prob_dict['prob'].append(prob)

    prob_df = pd.DataFrame(prob_dict)
    prob_df.to_csv(join(fold_prediction_dir, 'probs.csv'), index=False)


def pred_test_fold(model_path, fold):
    predictor = Predictor(model_path)
    prob_df = pd.read_csv(config.SAMPLE_SUBM_PATH)
    prob_df.rename(columns={'rle_mask': 'prob'}, inplace=True)

    fold_prediction_dir = join(PREDICTION_DIR, f'fold_{fold}', 'test')
    make_dir(fold_prediction_dir)

    for i, row in prob_df.iterrows():
        image_path = join(config.TEST_DIR, 'images'+IMAGES_NAME, row.id + '.png')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        segm, prob = predictor(image)
        row.prob = prob
        segm_save_path = join(fold_prediction_dir, row.id + '.png')
        cv2.imwrite(segm_save_path, segm)

    prob_df.to_csv(join(fold_prediction_dir, 'probs.csv'), index=False)


def get_mean_probs_df(pred_dir):
    probs_df_lst = []
    for i in range(len(FOLDS)):
        fold_dir = os.path.join(pred_dir, f'fold_{FOLDS[i]}')
        probs_df = pd.read_csv(join(fold_dir, 'test', 'probs.csv'), index_col='id')
        probs_df_lst.append(probs_df)

    mean_probs_df = probs_df_lst[0].copy()
    for probs_df in probs_df_lst[1:]:
        mean_probs_df.prob += probs_df.prob
    mean_probs_df.prob /= len(probs_df_lst)

    return mean_probs_df


def make_mean_submission():
    mean_probs_df = get_mean_probs_df(PREDICTION_DIR)

    sample_submition = pd.read_csv(config.SAMPLE_SUBM_PATH)

    for i, row in sample_submition.iterrows():
        pred_name = row.id + '.png'
        segm_lst = []
        for i in range(len(FOLDS)):
            fold_dir = os.path.join(PREDICTION_DIR, f'fold_{FOLDS[i]}')
            pred_path = join(fold_dir, 'test', pred_name)
            segm = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            segm = segm.astype(np.float32) / 255
            segm_lst.append(segm)

        mean_segm = np.mean(segm_lst, axis=0)
        prob = mean_probs_df.loc[row.id].prob

        pred = mean_segm > SEGM_THRESH
        prob = int(prob > PROB_THRESH)
        pred = (pred * prob).astype(np.uint8)

        rle_mask = RLenc(pred)
        row.rle_mask = rle_mask

    sample_submition.to_csv(join(PREDICTION_DIR, f'{EXPERIMENT_NAME}-mean-subm.csv'), index=False)


def get_best_model_path(dir_path):
    model_scores = []
    for model_name in os.listdir(dir_path):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', model_name)
        if score is not None:
            score = score.group(0)[1:-4]
            model_scores.append((model_name, score))
    model_score = sorted(model_scores, key=lambda x: x[1])
    best_model_name = model_score[-1][0]
    best_model_path = os.path.join(dir_path, best_model_name)
    return best_model_path


if __name__ == "__main__":
    for i in range(len(FOLDS)):
        print("Predict fold", FOLDS[i])
        fold_dir = os.path.join(FOLDS_DIR, f'fold_{FOLDS[i]}')
        best_model_path = get_best_model_path(fold_dir)
        print("Model path", best_model_path)
        print("Val predict")
        pred_val_fold(best_model_path, FOLDS[i])
        print("Test predict")
        pred_test_fold(best_model_path, FOLDS[i])

    print("Mean submission")
    make_mean_submission()
