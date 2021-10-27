import os
from typing import Any

import paddle
import paddle.nn as nn
import paddle.nn.functional as f

__all__ = ['CGNet', 'cgnet']


class CGNet(nn.Layer):
    def __init__(self, classes=19, m=3, n=21, dropOutFlag=False):
        super(CGNet, self).__init__()
        # stage 1
        self.level1_0 = ConvBNPReLU(3, 32, 3, 2)
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)

        self.sample1 = InputInjection(1)  # down sample feature map size divided 2, 1/2
        self.sample2 = InputInjection(2)  # down sample feature map size divided 4, 1/4

        self.b1 = BNPReLU(32 + 3)

        # Stage 2
        self.level2_0 = ContextGuidedBlockDown(32 + 3, 64, dilation_rate=2, reduction=8)  # CG block
        self.level2 = []
        for i in range(0, m - 1):
            self.level2.append(ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8))  # CG block
        self.level2 = nn.Sequential(*self.level2)

        self.bn_prelu_2 = BNPReLU(128 + 3)

        # Stage 3
        self.level3_0 = ContextGuidedBlockDown(128 + 3, 128, dilation_rate=4, reduction=16)
        self.level3 = []
        for i in range(0, n - 1):
            self.level3.append(ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16))  # CG block
        self.level3 = nn.Sequential(*self.level3)

        self.bn_prelu_3 = BNPReLU(256)

        if dropOutFlag:
            print("have droput layer")
            self.classifier = nn.Sequential(nn.Dropout2D(0.1), Conv(256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))
        # init model
        

    def forward(self, x):
        # stage 1
        output0 = self.level1_0(x)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)

        inp1 = self.sample1(x)
        inp2 = self.sample2(x)

        # stage 2
        output0_cat = self.b1(paddle.concat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        global output1, output2

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(paddle.concat([output1, output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(paddle.concat([output2_0, output2], 1))

        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        out = f.upsample(
            classifier,
            x.shape[2:],
            mode='bilinear',
            align_corners=False
        )  # Upsample score map, factor=8
        return out


class ConvBNPReLU(nn.Layer):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2D(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias_attr=False
        )
        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        output = self.conv(x)
        output = self.bn(output)
        output = self.act(output)
        return output


class InputInjection(nn.Layer):
    def __init__(self, down_sampling_ratio):
        super().__init__()
        self.pool = []
        for i in range(0, down_sampling_ratio):
            self.pool.append(nn.AvgPool2D(3, stride=2, padding=1))
        self.pool = nn.Sequential(*self.pool)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.pool(x)
        return x


class BNPReLU(nn.Layer):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        args:
           input: input feature map
           return: normalized and activated feature map
        """
        output = self.bn(x)
        output = self.act(output)
        return output


class ContextGuidedBlockDown(nn.Layer):
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

        self.bn = nn.BatchNorm2D(2 * nOut, epsilon=1e-3)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = Conv(2 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        output = self.conv1x1(x)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = paddle.concat([loc, sur], 1)  # the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


class ChannelWiseConv(nn.Layer):
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
        self.conv = nn.Conv2D(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            groups=nIn,
            bias_attr=False
        )

    def forward(self, x):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(x)
        return output


class ChannelWiseDilatedConv(nn.Layer):
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
        self.conv = nn.Conv2D(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            groups=nIn,
            bias_attr=False,
            dilation=d
        )

    def forward(self, x):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(x)
        return output


class Conv(nn.Layer):
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
        self.conv = nn.Conv2D(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias_attr=False
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class FGlo(nn.Layer):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x)
        y = paddle.reshape(y, (b, c))
        y = self.fc(y)
        y = paddle.reshape(y, (b, c, 1, 1))
        return x * y


class ContextGuidedBlock(nn.Layer):
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

        joi_feat = paddle.concat([loc, sur], 1)

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = x + output
        return output


def cgnet(pretrained: bool = False, **kwargs: Any) -> CGNet:
    model = CGNet(**kwargs)
    if pretrained:
        load_dygraph_pretrain(model, pretrained)
    return model


def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path)):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path)
    model.set_dict(param_state_dict)
    return
