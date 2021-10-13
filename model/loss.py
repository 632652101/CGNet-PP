import paddle.nn as nn
import paddle.nn.functional as f


class CrossEntropyLoss2d(nn.Layer):
    """
    This file defines a cross entropy loss for 2D images
    """

    def __init__(self, weight=None, ignore_label=255):
        """
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network.
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        """
        super().__init__()

        self.loss = nn.NLLLoss(weight, ignore_index=ignore_label)

    def forward(self, outputs, targets):
        return self.loss(f.log_softmax(outputs, 1), targets)
