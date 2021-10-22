import numpy as np
import paddle

from paddlevision.models.CGNet import cgnet

from reprod_log import ReprodLogger

if __name__ == "__main__":
    # def logger
    paddle.set_device("cpu")
    reprod_logger = ReprodLogger()

    model = cgnet(pretrained="../../weights/torch2paddle.padparams",
                  classes=19,
                  m=3,
                  n=21)
    model.eval()

    # read or gen fake data
    fack_data = np.load("../../fack_data/fake_data.npy")
    fack_data = paddle.to_tensor(fack_data)
    # forward
    out = model(fack_data)
    print(out.shape)
    # logger
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_paddle.npy")
