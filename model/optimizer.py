import paddle
from typing import Any

from paddle.optimizer.lr import LRScheduler


def get_optimizer(model: paddle.nn.Layer, **keywords: Any):
    schedule = StepDecay(**keywords)
    return paddle.optimizer.Adam(
        schedule,
        parameters=model.parameters(),
        epsilon=1e-08,
        weight_decay=5e-04
    ), schedule


class StepDecay(LRScheduler):
    def __init__(self,
                 learning_rate,
                 max_epoch=360,
                 last_epoch=-1,
                 verbose=False):
        """
        :param learning_rate: The initial learning rate
        :param max_epoch: max epoch
        :param last_epoch: for resume
        :param verbose: verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``Fa
        lse``
        """
        if not isinstance(max_epoch, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s." %
                type(max_epoch))
        self.max_epoch = max_epoch

        super(StepDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * pow((1 - 1.0 * self.last_epoch / self.max_epoch), 0.9)
