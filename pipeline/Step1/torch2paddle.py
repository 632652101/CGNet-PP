import numpy as np
import torch
import paddle
import pickle

from CGNet_paddle.paddlevision.models.CGNet import cgnet


def transfer():
    '''
    @ Description:
        转化torch的权重为 paddle 的权重.
        1. 必须保证结构是一致的, 该代码只能解决 fc 转置的部分, 其他部分需要自己调节
    :return: none
    '''
    output_fp = "CGNet-PP/pipeline/weights/model_cityscapes_train_on_trainvalset.pdparams"
    paddle_dict = {}
    torch_dict = torch.load("CGNet-PP/pipeline/weights/model_cityscapes_train_on_trainvalset.pth", map_location=torch.device('cpu'))
    torch_dict = torch_dict["model"]

    npy = []
    for k, v in torch_dict.items():
        if v.cpu().numpy().shape != ():
            npy.append(v.cpu().numpy())

    model = cgnet(pretrained=False,
                  classes=19,
                  m=3,
                  n=21)

    i = 0
    for key, value in model.state_dict().items():
        if value.numpy().shape != npy[i].shape:
            paddle_dict[key] = npy[i].T
        else:
            paddle_dict[key] = npy[i]
        i += 1

    paddle.save(paddle_dict, output_fp)


if __name__ == '__main__':
    transfer()
