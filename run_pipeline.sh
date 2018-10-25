#!/usr/bin/env bash

# make biharmonic inpaint
python make_inpaint_images.py

# train fpn-lovasz-se-resnext50-006-after-001
python pipeline/fpn-lovasz-se-resnext50-006/train_folds.py
python pipeline/fpn-lovasz-se-resnext50-006-after-001/train_folds.py
python pipeline/fpn-lovasz-se-resnext50-006-after-001/predict_folds.py

# train mos-fpn-lovasz-se-resnext50-001
python pipeline/mos-fpn-lovasz-se-resnext50-001/train_folds.py
python pipeline/mos-fpn-lovasz-se-resnext50-001/predict_folds.py

# make mean submission
python pipeline/mean_submission.py

# make postprocessed submission
python pipeline/postprocessed_submission.py
