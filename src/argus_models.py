import torch
import numpy as np

from argus import Model
from argus.utils import detach_tensors, to_device

from src.nn_modules import UNetProbResNet
from src.romul_zoo.senet import SeResnextProb50
from src.nick_zoo.unet_flex import UNetFlexProb
from src.nick_zoo.unet_flex_fpn import UNetFPNFlexProb
from src.nick_zoo.senet_fpn import SeResnextFPNProb50
from src.losses import FbBceProbLoss, ConsistencyLoss, LovaszProbLoss
from src.transforms import ProbOutputTransform
from src.metrics import SaltIOUT, SaltCropIOUT
from src.utils import sigmoid_rampup


class SaltProbModel(Model):
    nn_module = UNetProbResNet
    loss = FbBceProbLoss
    prediction_transform = ProbOutputTransform


class SaltMetaModel(Model):
    nn_module = {
        'UNetProbResNet': UNetProbResNet,
        'SeResnextProb50': SeResnextProb50,
        'UNetFlexProb': UNetFlexProb,
        'UNetFPNFlexProb': UNetFPNFlexProb,
        'SeResnextFPNProb50': SeResnextFPNProb50
    }
    loss = {
        'FbBceProbLoss': FbBceProbLoss,
        'LovaszProbLoss': LovaszProbLoss
    }
    prediction_transform = {
        'ProbOutputTransform': ProbOutputTransform
    }


class SaltMeanTeacherModel(Model):
    nn_module = {
        'UNetProbResNet': UNetProbResNet,
        'SeResnextProb50': SeResnextProb50,
        'UNetFlexProb': UNetFlexProb,
        'UNetFPNFlexProb': UNetFPNFlexProb,
        'SeResnextFPNProb50': SeResnextFPNProb50
    }
    loss = {
        'FbBceProbLoss': FbBceProbLoss,
        'LovaszProbLoss': LovaszProbLoss
    }
    prediction_transform = {
        'ProbOutputTransform': ProbOutputTransform
    }

    def __init__(self, params):
        super().__init__(params)
        self.alpha = params['mean_teacher']['alpha']
        self.rampup_length = params['mean_teacher']['rampup_length']
        self.unlabeled_batch = params['mean_teacher']['unlabeled_batch']
        self.test_dataset = None
        self.consistency_loss = ConsistencyLoss(
            params['mean_teacher']['consistency_segm_weight'],
            params['mean_teacher']['consistency_prob_weight'],
        )
        self.epoch = 0
        self.teacher = self._build_nn_module(self.params)
        self.teacher.to(self.device)
        self.teacher.train()
        for param in self.teacher.parameters():
            param.detach_()

    def update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(),
                                    self.nn_module.parameters()):
            t_param.data.mul_(self.alpha).add_(1 - self.alpha, s_param.data)

    def sample_unlabeled_input(self):
        assert self.test_dataset is not None
        indices = np.random.randint(0, len(self.test_dataset), size=self.unlabeled_batch)
        samples = [self.test_dataset[idx] for idx in indices]
        return torch.stack(samples, dim=0)

    def prepare_unlabeled_batch(self, batch, device):
        input, trg = batch
        unlabeled_input = self.sample_unlabeled_input()
        input = torch.cat([input, unlabeled_input], dim=0)
        return to_device(input, device), to_device(trg, device)

    def train_step(self, batch) -> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()

        input, target = self.prepare_unlabeled_batch(batch, self.device)
        student_pred = self.nn_module(input)
        with torch.no_grad():
            teacher_pred = self.teacher(input)

        consistency_weight = sigmoid_rampup(self.epoch, self.rampup_length)
        loss = consistency_weight * self.consistency_loss(student_pred, teacher_pred)
        student_pred = [pred[:target.size(0)] for pred in student_pred]

        loss += self.loss(student_pred, target)
        loss.backward()
        self.optimizer.step()
        self.update_teacher()
        prediction = detach_tensors(student_pred)
        target = detach_tensors(target)
        return {
            'prediction': self.prediction_transform(prediction),
            'target': target,
            'loss': loss.item()
        }
