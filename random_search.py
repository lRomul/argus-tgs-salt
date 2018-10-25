import os
import json
import random
import numpy as np
from pprint import pprint

from argus.callbacks import MonitorCheckpoint, EarlyStopping, LoggingToFile

from torch.utils.data import DataLoader

from src.dataset import SaltDataset
from src.transforms import SimpleDepthTransform, SaltTransform
from src.lr_scheduler import ReduceLROnPlateau
from src.argus_models import SaltMetaModel

from src.nick_zoo.resnet_blocks import resnet34


SAVE_DIR = 'random-search-flex-fpn-resnet34-001'
VAL_FOLDS = [0]
TRAIN_FOLDS = [1, 2, 3, 4]
START_FROM = 0
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
OUTPUT_SIZE = (101, 101)
TRAIN_FOLDS_PATH = '/workdir/data/train_folds_148.csv'


if __name__ == "__main__":
    for i in range(START_FROM, 1000):
        experiment_dir = f'/workdir/data/experiments/{SAVE_DIR}/{i:03}'
        np.random.seed(i)
        random.seed(i)
        random_params = {
            'dropout': float(np.random.uniform(0.0, 1.0)),
            'fb_weight': float(np.random.uniform(0.5, 1.0)),
            'fb_beta': float(np.random.choice([0.5, 1, 2])),
            'bce_weight': float(np.random.uniform(0.5, 1.0)),
            'prob_weight': float(np.random.uniform(0.5, 1.0))
        }

        params = {
            'nn_module': ('UNetFPNFlexProb', {
                'num_classes': 1,
                'num_channels': 3,
                'blocks': resnet34,
                'final': 'sigmoid',
                'dropout_2d': random_params['dropout'],
                'is_deconv': True,
                'deflation': 4,
                'use_first_pool': False,
                'skip_dropout': True,
                'pretrain': 'resnet34',
                'pretrain_layers': [True for _ in range(5)],
                'fpn_layers': [16, 32, 64, 128]
            }),
            'loss': ('FbBceProbLoss', {
                'fb_weight': random_params['fb_weight'],
                'fb_beta': random_params['fb_beta'],
                'bce_weight': random_params['bce_weight'],
                'prob_weight': random_params['prob_weight']
            }),
            'prediction_transform': ('ProbOutputTransform', {
                'segm_thresh': 0.5,
                'prob_thresh': 0.5,
            }),
            'optimizer': ('Adam', {'lr': 0.0001}),
            'device': 'cuda'
        }
        pprint(params)

        depth_trns = SimpleDepthTransform()
        train_trns = SaltTransform(IMAGE_SIZE, True, 'crop')
        val_trns = SaltTransform(IMAGE_SIZE, False, 'crop')
        train_dataset = SaltDataset(TRAIN_FOLDS_PATH, TRAIN_FOLDS, train_trns, depth_trns)
        val_dataset = SaltDataset(TRAIN_FOLDS_PATH, VAL_FOLDS, val_trns, depth_trns)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = SaltMetaModel(params)

        callbacks = [
            MonitorCheckpoint(experiment_dir, monitor='val_crop_iout', max_saves=1, copy_last=False),
            EarlyStopping(monitor='val_crop_iout', patience=100),
            ReduceLROnPlateau(monitor='val_crop_iout', patience=30, factor=0.7, min_lr=1e-8),
            LoggingToFile(os.path.join(experiment_dir, 'log.txt'))
        ]

        with open(os.path.join(experiment_dir, 'random_params.json'), 'w') as outfile:
            json.dump(random_params, outfile)

        model.fit(train_loader,
                  val_loader=val_loader,
                  max_epochs=600,
                  callbacks=callbacks,
                  metrics=['crop_iout'])
