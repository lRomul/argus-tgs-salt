import os
import json

import argus
from argus.callbacks import MonitorCheckpoint, EarlyStopping, LoggingToFile

from torch.utils.data import DataLoader

from src.dataset import SaltDataset, SaltTestDataset
from src.transforms import SimpleDepthTransform, SaltTransform
from src.lr_scheduler import ReduceLROnPlateau
from src.argus_models import SaltMeanTeacherModel
from src import config

from src.nick_zoo.resnet_blocks import resnet152


EXPERIMENT_NAME = 'flex-fpn-mt-resnet152-001'
TRAIN_BATCH_SIZE = 10
VAL_BATCH_SIZE = 10
UNLABELED_BATCH = 6
IMAGE_SIZE = (128, 128)
OUTPUT_SIZE = (101, 101)
TRAIN_FOLDS_PATH = '/workdir/data/train_folds_148.csv'
TEST_DIR = '/workdir/data/test/images148'
SAVE_DIR = f'/workdir/data/experiments/{EXPERIMENT_NAME}'
FOLDS = list(range(config.N_FOLDS))
PARAMS = {
    'nn_module': ('UNetFPNFlexProb', {
        'num_classes': 1,
        'num_channels': 3,
        'blocks': resnet152,
        'final': 'sigmoid',
        'dropout_2d': 0.2,
        'is_deconv': True,
        'deflation': 4,
        'use_first_pool': False,
        'skip_dropout': True,
        'pretrain': 'resnet152',
        'pretrain_layers': [True for _ in range(5)],
        'fpn_layers': [16, 32, 64, 128]
    }),
    'loss': ('FbBceProbLoss', {
        'fb_weight': 0.95,
        'fb_beta': 2,
        'bce_weight': 0.9,
        'prob_weight': 0.85
    }),
    'prediction_transform': ('ProbOutputTransform', {
        'segm_thresh': 0.5,
        'prob_thresh': 0.5,
    }),
    'mean_teacher': {
        'alpha': 0.99,
        'rampup_length': 10,
        'unlabeled_batch': UNLABELED_BATCH,
        'consistency_segm_weight': 0.3,
        'consistency_prob_weight': 0.3
    },
    'optimizer': ('Adam', {'lr': 0.0001}),
    'device': 'cuda'
}


@argus.callbacks.on_epoch_start
def update_model_epoch(state: argus.engine.State):
    state.model.epoch = state.epoch


def train_fold(save_dir, train_folds, val_folds):
    depth_trns = SimpleDepthTransform()
    train_trns = SaltTransform(IMAGE_SIZE, True, 'crop')
    val_trns = SaltTransform(IMAGE_SIZE, False, 'crop')
    train_dataset = SaltDataset(TRAIN_FOLDS_PATH, train_folds, train_trns, depth_trns)
    val_dataset = SaltDataset(TRAIN_FOLDS_PATH, val_folds, val_trns, depth_trns)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4)
    test_dataset = SaltTestDataset(test_dir=TEST_DIR, transform=val_trns, depth_transform=depth_trns)

    model = SaltMeanTeacherModel(PARAMS)
    model.test_dataset = test_dataset

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_crop_iout', max_saves=3, copy_last=False),
        EarlyStopping(monitor='val_crop_iout', patience=100),
        ReduceLROnPlateau(monitor='val_crop_iout', patience=25, factor=0.72, min_lr=1e-8),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
        update_model_epoch
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
