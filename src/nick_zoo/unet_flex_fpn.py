import torch
import torch.nn as nn

from src.nick_zoo.unet_flex import UNetFlexProb
from src.nick_zoo.resnet_blocks import resnet34
from src.nick_zoo.layers import FPNBlock

nonlinearity = nn.ReLU


class UNetFPNFlexProb(UNetFlexProb):
    def __init__(self, num_classes, num_channels=3, blocks=resnet34,
                 final='softmax', dropout_2d=0.5, deflation=4,
                 is_deconv=True, first_last_conv=(64, 32),
                 skip_dropout=False, xavier=False, pretrain=None,
                 pretrain_layers=None, use_first_pool=True, fpn_layers=None):
        super(UNetFPNFlexProb, self).__init__(num_classes, num_channels,
                                              blocks, final, dropout_2d, deflation,
                                              is_deconv, first_last_conv, skip_dropout,
                                              xavier, pretrain, pretrain_layers,
                                              use_first_pool)

        if fpn_layers is None:
            fpn_layers = [128 for _ in range(len(self.encoders_n))]
        else:
            assert len(self.encoders_n) == len(fpn_layers), \
                "Incorrect number of FPN blocks"

        n_blocks = len(fpn_layers)
        c_scale = 1

        self.fpn_blocks = nn.ModuleList([])
        for i in range(n_blocks):
            self.fpn_blocks.append(FPNBlock(self.encoders_n[i],
                                            fpn_layers[i], c_scale))
            c_scale = c_scale * 2

        fpn_filters = sum(fpn_layers)
        # Final classifier
        if self.use_first_pool:
            if is_deconv:
                self.finalupscale = nn.ConvTranspose2d(fpn_filters,
                                                       first_last_conv[1], 3,
                                                       stride=2)
                self.finalconv2 = nn.Conv2d(first_last_conv[1], first_last_conv[1], 3)
            else:
                self.finalupscale = nn.Upsample(scale_factor=2, mode='bilinear',
                                                align_corners=False)
                self.finalconv2 = nn.Conv2d(fpn_filters, first_last_conv[1], 3)
        else:
            self.finalconv2 = nn.Conv2d(fpn_filters, first_last_conv[1], 3, padding=1)

        self.prob_bn = nn.BatchNorm2d(self.encoders_n[-1])

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        if self.use_first_pool:
            x = self.firstmaxpool(x)

        enc = []
        for encoder in self.encoders:
            x = encoder(x)
            enc.append(x)

        # Decoder with Skip Connections
        dec = []
        for i in range(len(self.decoders) - 1, 0, -1):
            x = self.decoders[i](x)
            if self.skip_dropouts is not None:
                x = torch.cat([x, self.skip_dropouts[i - 1](enc[i - 1])], dim=1)
            else:
                x = torch.cat([x, enc[i - 1]], dim=1)
            dec.append(self.fpn_blocks[i](x))

        x = self.decoders[0](x)
        dec.append(self.fpn_blocks[0](x))
        # Concat the FPN
        x = torch.cat(dec, dim=1)

        # Final Classification
        if self.use_first_pool:
            x = self.finalupscale(x)
            x = self.finalrelu1(x)
        if self.finaldropout is not None:
            x = self.finaldropout(x)
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
        x = self.finalconv3(x)
        if self.final is not None:
            x = self.final(x)

        prob = self.prob_bn(enc[-1])
        prob = self.prob_avg_pool(prob)
        prob = prob.view(prob.size(0), -1)
        prob = self.prob_fc(prob)
        prob = self.sigmoid(prob)
        prob = prob.view(-1)
        return x, prob
