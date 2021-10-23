import torch


# torch 需要自己手动调整optimizer中的学习率
def get_optimizer(model, lr):
    return torch.optim.Adam(
        model.parameters(),
        lr,
        (0.9, 0.999),
        eps=1e-08,
        weight_decay=5e-4
    )


def adjust_learning_rate(cur_epoch, max_epoch, curEpoch_iter, perEpoch_iter, baselr):
    """
    poly learning stategyt
    lr = baselr*(1-iter/max_iter)^power
    """
    cur_iter = cur_epoch*perEpoch_iter + curEpoch_iter
    max_iter=max_epoch*perEpoch_iter
    lr = baselr*pow( (1 - 1.0*cur_iter/max_iter), 0.9)

    return lr
