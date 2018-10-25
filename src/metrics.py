# From https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/metrics.py

import numpy as np
from pycocotools import mask as cocomask

from argus.metrics import Metric

from src.transforms import CenterCrop


def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)


def binary_from_rle(rle):
    return cocomask.decode(rle)


def get_segmentations(labeled):
    nr_true = labeled.max()
    segmentations = []
    for i in range(1, nr_true + 1):
        msk = labeled == i
        segmentation = rle_from_binary(msk.astype('uint8'))
        segmentation['counts'] = segmentation['counts'].decode("UTF-8")
        segmentations.append(segmentation)
    return segmentations


def iou(gt, pred):
    gt[gt > 0] = 1.
    pred[pred > 0] = 1.
    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = 1e-09
    return intersection / union


def compute_ious(gt, predictions):
    gt_ = get_segmentations(gt)
    predictions_ = get_segmentations(predictions)

    if len(gt_) == 0 and len(predictions_) == 0:
        return np.ones((1, 1))
    elif len(gt_) != 0 and len(predictions_) == 0:
        return np.zeros((1, 1))
    else:
        iscrowd = [0 for _ in predictions_]
        ious = cocomask.iou(gt_, predictions_, iscrowd)
        if not np.array(ious).size:
            ious = np.zeros((1, 1))
        return ious


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)


class SaltIOUT(Metric):
    name = 'iout'
    better = 'max'

    def reset(self):
        self.iout_sum = 0
        self.count = 0

    def update(self, step_output: dict):
        preds = step_output['prediction'].cpu().numpy().astype(np.uint8)
        trgs = step_output['target'].cpu().numpy().astype(np.uint8)

        for i in range(trgs.shape[0]):
            pred = preds[i][0]
            trg = trgs[i][0]
            self.iout_sum += compute_eval_metric(trg, pred)
            self.count += 1

    def compute(self):
        return self.iout_sum / self.count


class SaltCropIOUT(Metric):
    name = 'crop_iout'
    better = 'max'

    def __init__(self, size=(101, 101)):
        super().__init__()
        self.crop_trns = CenterCrop(size)

    def reset(self):
        self.iout_sum = 0
        self.count = 0

    def update(self, step_output: dict):
        preds = step_output['prediction'].cpu().numpy().astype(np.uint8)
        trgs = step_output['target'].cpu().numpy().astype(np.uint8)

        for i in range(trgs.shape[0]):
            pred = preds[i][0]
            trg = trgs[i][0]
            pred, trg = self.crop_trns(pred, trg)
            self.iout_sum += compute_eval_metric(trg, pred)
            self.count += 1

    def compute(self):
        return self.iout_sum / self.count
