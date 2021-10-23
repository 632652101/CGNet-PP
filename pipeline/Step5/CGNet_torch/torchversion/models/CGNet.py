import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Any

__all__ = ['CGNet', 'cgnet']


class CGNet(nn.Module):
    def __init__(self, classes=19, m=3, n=21, dropOutFlag=False):
        super(CGNet, self).__init__()
        # Stage 1
        self.level1_0 = ConvBNPReLU(3, 32, 3, 2)
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)  # feature map size divided 2, 1/2

        self.sample1 = InputInjection(1)  # down sample feature map size divided 2, 1/2
        self.sample2 = InputInjection(2)  # down sample feature map size divided 4, 1/4

        self.b1 = BNPReLU(32 + 3)

        # Stage 2
        self.level2_0 = ContextGuidedBlockDown(32 + 3, 64, dilation_rate=2, reduction=8)  # CG block
        self.level2 = nn.ModuleList()
        for i in range(0, m - 1):
            self.level2.append(ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8))  # CG block
        self.bn_prelu_2 = BNPReLU(128 + 3)

        # Stage 3
        self.level3_0 = ContextGuidedBlockDown(128 + 3, 128, dilation_rate=4, reduction=16)
        self.level3 = nn.ModuleList()
        for i in range(0, n - 1):
            self.level3.append(ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16))  # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropOutFlag:
            print("have droput layer")
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), Conv(256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))

        # init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d') != -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(x)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(x)
        inp2 = self.sample2(x)

        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        global output1, output2

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))

        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        out = f.upsample(
            classifier,
            x.size()[2:],
            mode='bilinear',
            align_corners=False
        )  # Upsample score map, factor=8
        return out


class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=(stride,),
            padding=(padding, padding),
            bias=False
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        output = self.bn(output)
        output = self.act(output)
        return output


class InputInjection(nn.Module):
    def __init__(self, down_sampling_ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, down_sampling_ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for pool in self.pool:
            x = pool(x)
        return x


class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
           input: input feature map
           return: normalized and activated feature map
        """
        output = self.bn(x)
        output = self.act(output)
        return output


class ContextGuidedBlockDown(nn.Module):
    """
        the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)  # size/2, channel: nIn--->nOut

        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)

        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = Conv(2 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv1x1(x)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  # the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=(stride,),
            padding=(padding, padding),
            groups=nIn,
            bias=False
        )

    def forward(self, x):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(x)
        return output


class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=(stride,),
            padding=(padding, padding),
            groups=nIn,
            bias=False,
            dilation=(d,)
        )

    def forward(self, x):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(x)
        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=(stride,),
            padding=(padding, padding),
            bias=False
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels,
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)  # 1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate)  # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, x):
        output = self.conv1x1(x)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = x + output
        return output


def cgnet(pretrained: bool = False, cuda: bool = False, **kwargs: Any) -> CGNet:
    model = CGNet(**kwargs)
    if pretrained:
        load_dygraph_pretrain(model, cuda, pretrained)
    return model


def load_dygraph_pretrain(model, cuda, path=None):
    '''
    加载作者的自己的网路结构, 这个网络有点问题, 用 CPU 的话只能通过如下的方式进行加载
    '''
    if not (os.path.isdir(path) or os.path.exists(path)):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    if cuda:
        param_state_dict = torch.load(path)
    else:
        param_state_dict = torch.load(path, map_location=torch.device('cpu'))
    dict = convert_state_dict(param_state_dict)["model"]
    model.load_state_dict(dict)
    return


def convert_state_dict(state_dict):
    """
    Converts a state dict saved from a dataParallel module to normal module state_dict inplace
    Args:
        state_dict is the loaded DataParallel model_state
    """
    state_dict_new = OrderedDict()
    state_dict_new["epoch"] = state_dict["epoch"]
    sub_dic = OrderedDict()

    # print(type(state_dict))
    for k, v in state_dict['model'].items():
        # print(k)
        name = k[7:]  # remove the prefix module.
        # My heart is borken, the pytorch have no ability to do with the problem.
        sub_dic[name] = v

    state_dict_new["model"] = sub_dic
    return state_dict_new
