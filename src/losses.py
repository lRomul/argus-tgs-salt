import torch
from torch import nn

from src.lovasz import lovasz_hinge


# Source https://github.com/NikolasEnt/Lyft-Perception-Challenge/blob/master/loss.py
def fb_loss(preds, trues, beta):
    smooth = 1e-4
    beta2 = beta*beta
    batch = preds.size(0)
    classes = preds.size(1)
    preds = preds.view(batch, classes, -1)
    trues = trues.view(batch, classes, -1)
    weights = torch.clamp(trues.sum(-1), 0., 1.)
    TP = (preds * trues).sum(2)
    FP = (preds * (1-trues)).sum(2)
    FN = ((1-preds) * trues).sum(2)
    Fb = ((1+beta2) * TP + smooth)/((1+beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = Fb.sum() / (weights.sum() + smooth)
    return torch.clamp(score, 0., 1.)


class FBLoss(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        return 1 - fb_loss(output, target, self.beta)


class FbBceProbLoss(nn.Module):
    def __init__(self, fb_weight=0.33, fb_beta=1, bce_weight=0.33, prob_weight=0.33, min_mask_pixels=0):
        super().__init__()
        self.fb_weight = fb_weight
        self.bce_weight = bce_weight
        self.prob_weight = prob_weight
        self.min_mask_pixels = min_mask_pixels

        self.fb_loss = FBLoss(beta=fb_beta)
        self.bce_loss = nn.BCELoss()
        self.prob_loss = nn.BCELoss()

    def forward(self, output, target):
        segm, prob_pred = output
        if self.fb_weight > 0:
            fb = self.fb_loss(segm, target) * self.fb_weight
        else:
            fb = 0

        if self.bce_weight > 0:
            bce = self.bce_loss(segm, target) * self.bce_weight
        else:
            bce = 0

        prob_trg = target.view(target.size(0), -1).sum(dim=1) > self.min_mask_pixels
        prob_trg = prob_trg.to(torch.float32)
        if self.prob_weight > 0:
            prob = self.prob_loss(prob_pred, prob_trg) * self.prob_weight
        else:
            prob = 0

        return fb + bce + prob


class ConsistencyLoss(nn.Module):
    def __init__(self, segm_weight=0.33, prob_weight=0.33):
        super().__init__()
        self.segm_weight = segm_weight
        self.prob_weight = prob_weight

        self.segm_loss = nn.BCELoss()
        self.prob_loss = nn.BCELoss()

    def forward(self, student_pred, teacher_pred):
        student_segm, student_prob = student_pred
        teacher_segm, teacher_prob = teacher_pred

        if self.segm_weight > 0:
            segm = self.segm_loss(student_segm, teacher_segm) * self.segm_weight
        else:
            segm = 0

        if self.prob_weight > 0:
            prob = self.prob_loss(student_prob, teacher_prob) * self.prob_weight
        else:
            prob = 0

        return segm + prob


class LovaszProbLoss(nn.Module):
    def __init__(self, lovasz_weight=0.5, prob_weight=0.5, min_mask_pixels=0):
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.prob_weight = prob_weight
        self.min_mask_pixels = min_mask_pixels
        self.prob_loss = nn.BCELoss()

    def forward(self, output, target):
        segm, prob_pred = output

        prob_trg = target.view(target.size(0), -1).sum(dim=1) > self.min_mask_pixels
        prob_trg = prob_trg.to(torch.float32)
        if self.prob_weight > 0:
            prob = self.prob_loss(prob_pred, prob_trg) * self.prob_weight
        else:
            prob = 0

        if self.lovasz_weight > 0:
            lovasz = lovasz_hinge(segm.squeeze(1), target.squeeze(1)) \
                     * self.lovasz_weight
        else:
            lovasz = 0

        return prob + lovasz
