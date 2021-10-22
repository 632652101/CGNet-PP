import numpy as np
import torch

from torchvision.models.CGNet import cgnet

from reprod_log import ReprodLogger

if __name__ == "__main__":
    reprod_logger = ReprodLogger()
    model = cgnet(pretrained="../../weights/model_cityscapes_train_on_trainset.pth",
                  cuda=False,
                  classes=19,
                  m=3,
                  n=21)
    model.eval()

    # read or gen fake data
    fack_data = np.load("../../fack_data/fake_data.npy")
    fack_data = torch.from_numpy(fack_data)
    # forward
    out = model(fack_data)
    print(out.shape)
    # loggers
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_torch.npy")
