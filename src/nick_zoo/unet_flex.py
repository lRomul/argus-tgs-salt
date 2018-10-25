import torch
import torch.nn as nn

from src.nick_zoo.resnet_blocks import resnet34
from src.nick_zoo.layers import DecoderBlock, BasicBlock, Bottleneck
from src.nick_zoo.utils import class_for_name

nonlinearity = nn.ReLU


class UNetFlexProb(nn.Module):
    block_dict = {
        'basic': BasicBlock,
        'bottleneck': Bottleneck
    }

    def __init__(self, num_classes, num_channels=3, blocks=resnet34,
                 final='softmax', dropout_2d=0.5, deflation=4,
                 is_deconv=True, first_last_conv=[64, 32],
                 skip_dropout=False, xavier=False, pretrain=None,
                 pretrain_layers=None, use_first_pool=True,
                 return_enc=False):
        super(UNetFlexProb, self).__init__()
        assert num_channels > 0, "Incorrect num channels"
        assert final in ['softmax', 'sigmoid'],\
                         "Incorrect output type. Should be 'softmax' or 'sigmoid'"
        assert len(first_last_conv)==2, "'first_last_conv' - list with number of the first and the last convs"
        self.inplanes = first_last_conv[0]
        self.dropout_2d = dropout_2d
        self.use_first_pool = use_first_pool
        self.return_enc = return_enc

        # Initial convolutions
        self.firstconv = nn.Conv2d(num_channels, self.inplanes, kernel_size=(7, 7),
                                   stride=(2, 2), padding=(3, 3), bias=False)
        self.firstbn = nn.BatchNorm2d(self.inplanes)
        self.firstrelu = nonlinearity(inplace=True)
        self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoders
        self.encoders = nn.ModuleList([])
        encoders_n = []
        for block in blocks:
            self.encoders.append(self._make_encoder(self.block_dict[block['type']],
                                                    block['n_blocks'],
                                                    block['n_filters'],
                                                    block['stride']))
            encoders_n.append(self.inplanes)

        # Decoders
        self.decoders = nn.ModuleList([DecoderBlock(2*encoders_n[0], encoders_n[0],
                                                    is_deconv, deflation=deflation)])
        for i in range(1, len(encoders_n)-1):
            self.decoders.append(DecoderBlock(2*encoders_n[i], encoders_n[i-1],
                                              is_deconv, deflation=deflation))
        self.decoders.append(DecoderBlock(encoders_n[-1], encoders_n[-2],
                                          is_deconv, deflation=deflation))
        print("Actual number of filters:", encoders_n)
        self.encoders_n = encoders_n
        if skip_dropout and dropout_2d is not None:
            self.skip_dropouts = nn.ModuleList([nn.Dropout2d(p=dropout_2d)
                                for _ in range(len(self.decoders)-1)])
        else:
            self.skip_dropouts = None

        # Final classifier
        if self.use_first_pool:
            if is_deconv:
                self.finalupscale = nn.ConvTranspose2d(encoders_n[0],
                                                       first_last_conv[1], 3,
                                                       stride=2)
                self.finalconv2 = nn.Conv2d(first_last_conv[1], first_last_conv[1], 3)
            else:
                self.finalupscale = nn.Upsample(scale_factor=2, mode='bilinear',
                                                align_corners=False)
                self.finalconv2 = nn.Conv2d(encoders_n[0], first_last_conv[1], 3)
        else:
            self.finalconv2 = nn.Conv2d(encoders_n[0], first_last_conv[1],  3, padding=1)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finaldropout = nn.Dropout2d(p=dropout_2d) if dropout_2d\
                            is not None else None
        self.finalrelu2 = nonlinearity(inplace=True)
        if self.use_first_pool:
            self.finalconv3 = nn.Conv2d(first_last_conv[1], num_classes, 2, padding=1)
        else:
            self.finalconv3 = nn.Conv2d(first_last_conv[1], num_classes, 3, padding=1)
        # Prob branch
        self.pool = nn.MaxPool2d(2, 2)
        self.prob_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.prob_fc = nn.Linear(encoders_n[-1], 1)
        self.sigmoid = nn.Sigmoid()

        if final=='softmax':
            self.final = nn.Softmax(dim=1)
        else:
            self.final = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if xavier:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # Default resnet initialization from torchvision
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrain is not None and pretrain_layers is not None:
            use_pretrain_resnet(self, pretrain, pretrain_layers)


    def _make_encoder(self, block, n_blocks, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
        for i in range(len(self.decoders)-1, 0, -1):
            x = self.decoders[i](x)
            if self.skip_dropouts is not None:
                x = torch.cat([x, self.skip_dropouts[i-1](enc[i-1])], dim=1)
            else:
                x = torch.cat([x, enc[i-1]], dim=1)

        x = self.decoders[0](x)

        # Final Classification
        if self.use_first_pool:
            x = self.finalupscale(x)
            x = self.finalrelu1(x)
        if self.finaldropout is not None:
            x = self.finaldropout(x)
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
        x = self.finalconv3(x)
        x = self.final(x)

        prob = self.prob_avg_pool(enc[-1])
        prob = prob.view(prob.size(0), -1)
        prob = self.prob_fc(prob)
        prob = self.sigmoid(prob)
        prob = prob.view(-1)
        if self.return_enc:
            return x, prob, enc
        else:
            return x, prob


def use_pretrain_resnet(model, encoder, layers):
    # Unsafe function, use wisely!
    assert encoder in ['resnet18', 'resnet34', 'resnet50',\
                       'resnet101', 'resnet152'],\
                       "Incorrect encoder type"
    resnet = class_for_name("torchvision.models", encoder)\
        (pretrained=True)
    resnet_layers = [resnet.layer1, resnet.layer2,
                     resnet.layer3, resnet.layer4]
    if layers[0]:
        model.firstconv = resnet.conv1
        model.firstbn = resnet.bn1
        model.firstrelu = resnet.relu
        model.firstmaxpool = resnet.maxpool

    if len(layers) > 1:
        for i, layer in enumerate(layers[1:]):
            if layer:
                model.encoders[i] = resnet_layers[i]
