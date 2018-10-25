import os
import json

import argus
from argus.callbacks import MonitorCheckpoint, EarlyStopping, LoggingToFile

from torch.utils.data import DataLoader

from src.dataset import SaltDataset
from src.transforms import SimpleDepthTransform, SaltTransform
from src.lr_scheduler import ReduceLROnPlateau
from src.argus_models import SaltMetaModel
from src.losses import LovaszProbLoss
from src import config

from src.nick_zoo.resnet_blocks import resnet34


EXPERIMENT_NAME = 'fpn-lovasz-se-resnext50-006'
BATCH_SIZE = 24
IMAGE_SIZE = (128, 128)
OUTPUT_SIZE = (101, 101)
TRAIN_FOLDS_PATH = '/workdir/data/train_folds_148.csv'
SAVE_DIR = f'/workdir/data/experiments/{EXPERIMENT_NAME}'
N_FOLDS = 5
FOLDS = list(range(N_FOLDS))
PARAMS = {
    'nn_module': ('SeResnextFPNProb50', {
        'num_classes': 1,
        'num_channels': 3,
        'final': 'logits',
        'dropout_2d': 0.2,
        'skip_dropout': True,
        'fpn_layers': [8, 16, 32, 64, 128]
    }),
    'loss': ('LovaszProbLoss', {
        'lovasz_weight': 0.75,
        'prob_weight': 0.25,
    }),
    'prediction_transform': ('ProbOutputTransform', {
        'segm_thresh': 0.0,
        'prob_thresh': 0.5,
    }),
    'optimizer': ('SGD', {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001}),
    'device': 'cuda'
}


def train_fold(save_dir, train_folds, val_folds):
    depth_trns = SimpleDepthTransform()
    train_trns = SaltTransform(IMAGE_SIZE, True, 'crop')
    val_trns = SaltTransform(IMAGE_SIZE, False, 'crop')
    train_dataset = SaltDataset(TRAIN_FOLDS_PATH, train_folds, train_trns, depth_trns)
    val_dataset = SaltDataset(TRAIN_FOLDS_PATH, val_folds, val_trns, depth_trns)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SaltMetaModel(PARAMS)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_crop_iout', max_saves=3, copy_last=False),
        EarlyStopping(monitor='val_crop_iout', patience=100),
        ReduceLROnPlateau(monitor='val_crop_iout', patience=30, factor=0.64, min_lr=1e-8),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=700,
              callbacks=callbacks,
              metrics=['crop_iout'])


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(os.path.join(SAVE_DIR, 'source.py'), 'w') as outfile:
        outfile.write(open(__file__).read())

    with open(os.path.join(SAVE_DIR, 'params.json'), 'w') as outfile:
        json.dump(PARAMS, outfile)

    for i in range(len(FOLDS)):
        val_folds = [FOLDS[i]]
        train_folds = FOLDS[:i] + FOLDS[i + 1:]
        save_fold_dir = os.path.join(SAVE_DIR, f'fold_{FOLDS[i]}')
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds)
