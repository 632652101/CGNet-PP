import json
import paddle
from mmcv import Config

from utils.dataloader import cityscapes
from model import CGNet
from utils.eval.evaluate import accuracy as acc


def val(args):
    cfg = Config.fromfile(args.configfile)
    print(json.dumps(cfg._cfg_dict, indent=4))

    # data loader
    val_dataset = cityscapes.get_dataset_val(**cfg.data.set.val)
    val_dataloader = cityscapes.get_dataloader(val_dataset, **cfg.data.loader.val)

    # device
    device = paddle.get_device()
    paddle.set_device(device)

    # models
    model = CGNet.cgnet(**cfg.model.backbone)

    # accuracy
    iou_mean, iou_list = acc(loader=val_dataloader, model=model, classes=19, )
    print("mIou: (%.6f)\nIou_list: \n" % iou_mean + str(iou_list))

