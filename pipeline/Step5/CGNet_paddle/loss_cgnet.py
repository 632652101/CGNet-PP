import numpy as np
import paddle
import paddle.nn as nn
from paddleversion.models.CGNet import cgnet
from paddleversion.models.loss import CrossEntropyLoss2d
from paddleversion.datasets.preprocess import get_inform_data

from reprod_log import ReprodLogger

if __name__ == "__main__":
    paddle.set_device("cpu")

    reprod_logger = ReprodLogger()
    model = cgnet(pretrained="CGNet-PP/pipeline/weights/model_cityscapes_train_on_trainvalset.pdparams")
    model.eval()

    d = get_inform_data("CGNet-PP/pipeline/weights/cityscapes_inform.pkl")
    weight = paddle.to_tensor(d['classWeights'])
    criterion = CrossEntropyLoss2d(weight)

    # read or gen fake data
    fake_data = np.load("CGNet-PP/pipeline/fack_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    fake_label = np.load("CGNet-PP/pipeline/fack_data/fake_label.npy")
    fake_label = paddle.to_tensor(fake_label, dtype='int64')

    # forward
    out = model(fake_data)
    loss = criterion(out, fake_label)
    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.add("out", out.cpu().detach().numpy())
    reprod_logger.save("CGNet-PP/pipeline/Step3.5/CGNet_paddle/loss_paddle.npy")
