import paddle
from typing import Any


def get_optimizer(model: paddle.nn.Layer, **keywords: Any):
    return paddle.optimizer.Adam(
        StepDecay(**keywords),
        parameters = model.parameters(),
        epsilon=1e-08,
        weight_decay=5e-04
    )


def adjust_learning_rate(last_epoch, max_iter, baselr):
    """
    poly learning stategyt
    lr = baselr*(1-iter/max_iter)^power
    """
    lr = baselr * pow((1 - 1.0 * last_epoch / max_iter), 0.9)

    return lr


from paddle.optimizer.lr import LRScheduler


class StepDecay(LRScheduler):
    def __init__(self,
                 learning_rate,
                 max_iter,
                 last_epoch=-1,
                 verbose=False):
        """
        :param learning_rate: The initial learning rate
        :param step_size: used by get_lr
        :param last_epoch: 上一次完成后的iteration的数目
        :param verbose: verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False``
        """
        if not isinstance(max_iter, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s." %
                type(max_iter))
        self.max_iter = max_iter

        super(StepDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * pow((1 - 1.0 * self.last_epoch / self.max_iter), 0.9)
