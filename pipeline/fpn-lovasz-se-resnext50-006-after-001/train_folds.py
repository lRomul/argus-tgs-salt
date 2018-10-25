import os
from os.path import join
import re
import math
import torch

import argus
from argus.callbacks import MonitorCheckpoint, LoggingToFile
from argus import load_model

from torch.utils.data import DataLoader

from src.dataset import SaltDataset
from src.transforms import SimpleDepthTransform, SaltTransform
from src.argus_models import SaltMetaModel
from src.losses import LovaszProbLoss
from src import config


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


BASE_EXPERIMENT_NAME = 'fpn-lovasz-se-resnext50-006'
EXPERIMENT_NAME = 'fpn-lovasz-se-resnext50-006-after-001'
N_FOLDS = 5
FOLDS = list(range(N_FOLDS))
BATCH_SIZE = 16
IMAGE_SIZE = (128, 128)
OUTPUT_SIZE = (101, 101)
TRAIN_FOLDS_PATH = '/workdir/data/train_folds_148.csv'
LR = 0.005
SAVE_DIR = f'/workdir/data/experiments/{EXPERIMENT_NAME}'


class CosineAnnealingLR:
    def __init__(self, base_lr, T_max, eta_min=0.):
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = base_lr

    def __call__(self, epoch):
        return self.eta_min + (self.base_lr - self.eta_min) \
               * (1 + math.cos(math.pi * (epoch % self.T_max) / self.T_max)) / 2


cos_ann = CosineAnnealingLR(LR, 50, eta_min=LR*0.1)


@argus.callbacks.on_epoch_start
def update_lr(state: argus.engine.State):
    lr = cos_ann(state.epoch)
    state.model.set_lr(lr)
    state.logger.info(f"Set lr: {lr}")


def train_fold(save_dir, train_folds, val_folds, model_path):
    depth_trns = SimpleDepthTransform()
    train_trns = SaltTransform(IMAGE_SIZE, True, 'crop')
    val_trns = SaltTransform(IMAGE_SIZE, False, 'crop')
    train_dataset = SaltDataset(TRAIN_FOLDS_PATH, train_folds, train_trns, depth_trns)
    val_dataset = SaltDataset(TRAIN_FOLDS_PATH, val_folds, val_trns, depth_trns)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = load_model(model_path)
    model.nn_module.final = None
    model.prediction_transform.segm_thresh = 0
    model.loss = LovaszProbLoss(lovasz_weight=0.5, prob_weight=0.5)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_crop_iout', max_saves=3, copy_last=False),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
        update_lr
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=300,
              callbacks=callbacks,
              metrics=['crop_iout'])


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(os.path.join(SAVE_DIR, 'source.py'), 'w') as outfile:
        outfile.write(open(__file__).read())

    for i in range(len(FOLDS)):
        val_folds = [FOLDS[i]]
        train_folds = FOLDS[:i] + FOLDS[i + 1:]
        save_fold_dir = os.path.join(SAVE_DIR, f'fold_{FOLDS[i]}')
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        model_path = get_best_model_path(join('/workdir/data/experiments', BASE_EXPERIMENT_NAME, 'fold_%d' % i))
        print(f'Base model path: {model_path}')
        train_fold(save_fold_dir, train_folds, val_folds, model_path)
