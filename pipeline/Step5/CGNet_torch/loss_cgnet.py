import numpy as np
import torch
import torch.nn as nn

from torchversion.models.CGNet import cgnet
from torchversion.models.loss import CrossEntropyLoss2d
from torchversion.datasets.preprocess import get_inform_data

from reprod_log import ReprodLogger

if __name__ == "__main__":
    reprod_logger = ReprodLogger()

    model = cgnet(pretrained="CGNet-PP/pipeline/weights/model_cityscapes_train_on_trainvalset.pth")
    model.eval()

    d = get_inform_data("CGNet-PP/pipeline/weights/cityscapes_inform.pkl")
    weight = torch.from_numpy(d['classWeights'])
    criterion = CrossEntropyLoss2d(weight)

    # read or gen fake data
    fake_data = np.load("CGNet-PP/pipeline/fack_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)

    fake_label = np.load("CGNet-PP/pipeline/fack_data/fake_label.npy")
    fake_label = torch.from_numpy(fake_label).long()

    # forward
    out = model(fake_data)
    loss = criterion(out, fake_label)
    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.add("out", out.cpu().detach().numpy())
    reprod_logger.save("CGNet-PP/pipeline/Step3.5/CGNet_torch/loss_torch.npy")
