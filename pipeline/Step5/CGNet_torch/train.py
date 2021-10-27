import numpy as np
import torch
from reprod_log import ReprodLogger

from torchversion.datasets import preprocess
from torchversion.datasets import cityscapes
from torchversion.models import CGNet, loss, optimizer as opt


def train_some_iters(model,
                     criterion,
                     optimizer,
                     fake_data,
                     fake_label,
                     data_loader,
                     epoch=0,
                     max_iter=2):
    model.eval()
    loss_list = []
    total_batches = len(data_loader)
    for idx in range(max_iter):
        image = torch.from_numpy(fake_data)
        target = torch.from_numpy(fake_label).long()
        output = model(image)
        loss = criterion(output, target)

        loss.backward()

        lr = opt.adjust_learning_rate(cur_epoch=epoch,
                                      max_epoch=args.max_epochs,
                                      curEpoch_iter=idx,
                                      perEpoch_iter=total_batches,
                                      baselr=args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss)

        print(f"{idx + 1}/{max_iter}, loss= {loss.detach().cpu().numpy()}")

    return loss_list


def main(args):
    device = torch.device(args.device)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    print(f"=====> Creating Dataloader")
    datas = preprocess.get_inform_data(args.inform_data_file)

    global data_loader

    # dataset keywords
    dataset_args = {
        "root": args.data_dir,
        "list_path": None,
        "crop_size": input_size,
        "scale": args.random_scale,
        "mirror": args.random_mirror
    }

    if args.dataloader == "train":
        dataset_args["list_path"] = args.dataset_train_list_path
        dataset = cityscapes.get_dataset_train(**dataset_args)
    elif args.dataloader == "val":
        dataset_args.pop("crop_size")
        dataset_args.pop("scale")
        dataset_args.pop("mirror")
        dataset_args["list_path"] = args.dataset_val_list_path
        dataset = cityscapes.get_dataset_val(**dataset_args)
    elif args.dataloader == "test":
        dataset_args["list_path"] = args.dataset_test_list_path
        dataset = cityscapes.get_dataset_test(**dataset_args)
    elif args.dataloader == "trainval":
        dataset_args["list_path"] = args.dataset_trainval_list_path
        dataset = cityscapes.get_dataset_train(**dataset_args)
    else:
        raise Exception("args.dataloader err type")

    dataloader_args = {
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": args.drop_last
    }

    data_loader = cityscapes.get_dataloader(dataset, **dataloader_args)

    print("=====> Dataloader created. ")

    print("=====> Creating model")

    model_args = {
        "classes": args.classes,
        "m": args.m,
        "n": args.n,
        "dropOutFlag": args.dropOutFlag,
        "pretrained": args.pretrained_path,
    }

    model = CGNet.cgnet(**model_args)

    print("=====> Model created.")

    print("=====> Creating criterion.")

    weight = torch.from_numpy(datas['classWeights'])
    criterion_args = {
        "weight": weight,
        "ignore_label": args.ignore_label
    }

    criterion = loss.CrossEntropyLoss2d(
        **criterion_args
    )

    print("=====> Criterion created.")

    # 评估参数
    accuracy_args = {
        "loader": data_loader,
        "model": model,
        "classes": args.classes,
        "cuda": args.cuda,
        "max_iters": args.test_max_iters,
    }

    if args.test_only:
        from torchversion.utils.utils import accuracy
        iou_mean, iou_list = accuracy(**accuracy_args)
        return iou_mean, iou_list

    print("=====> Creating optimizer")
    max_iter = len(data_loader) * args.max_epochs

    opt_args = {
        "lr": args.lr,
    }
    optimize = opt.get_optimizer(model, **opt_args)

    print("=====> Optimizer Created")

    print("=====> Beginning training")

    fake_data = np.load("CGNet-PP/pipeline/fack_data/fake_data.npy")
    fake_label = np.load("CGNet-PP/pipeline/fack_data/fake_label.npy")

    loss_list = train_some_iters(
        model,
        criterion,
        optimize,
        fake_data,
        fake_label,
        data_loader,
        max_iter=2
    )
    print(loss_list)
    return loss_list

    # max_epochs = args.max_epochs
    #
    # for epoch in range(0, max_epochs):
    #     pass
    # lossTr, per_class_iu_tr, mIOU_tr, lr = train(args, dataloader, model, criteria, optimizer, epoch)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Training', add_help=add_help)

    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true", )

    parser.add_argument("--test_max_iters", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=360)

    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        '--train_type',
        type=str,
        default="ontrainval",
        help="ontrain for training on train set, ontrainval for training on train+val set"
    )

    parser.add_argument(
        '--dataset',
        default="cityscapes",
        help="dataset: cityscapes or camvid",
    )

    # dataset args
    parser.add_argument(
        '--data_dir',
        default="CGNet-PP/dataset/Cityscapes",
        help='data directory'
    )

    parser.add_argument(
        '--dataloader',
        default="train",
        help="dataloader, options: train, val, trainval,test"
    )

    parser.add_argument(
        '--dataset_train_list_path',
        default="CGNet-PP/dataset/Cityscapes/list/Cityscapes/cityscapes_train_list.txt",
        help="train set"
    )

    parser.add_argument(
        '--dataset_val_list_path',
        default="CGNet-PP/dataset/Cityscapes/list/Cityscapes/cityscapes_val_list.txt",
        help="val set"
    )

    parser.add_argument(
        '--dataset_trainval_list_path',
        default="CGNet-PP/dataset/Cityscapes/list/Cityscapes/cityscapes_trainval_list.txt",
        help="trainval set"
    )

    parser.add_argument(
        '--dataset_test_list_path',
        default="CGNet-PP/dataset/Cityscapes/list/Cityscapes/cityscapes_test_list.txt",
        help="test set"
    )

    '''
    dataloader_args = {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 1,
        "pin_memory": True,
        "drop_last": True
    }
    '''
    # dataloader args
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="the batch size is set to 16 for 2 GPUs"
    )

    parser.add_argument(
        '--shuffle',
        type=bool,
        default=True,
        help="the batch size is set to 16 for 2 GPUs"
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help=" the number of parallel threads"
    )

    parser.add_argument(
        '--pin_memory',
        type=bool,
        default=True,
    )

    parser.add_argument(
        '--drop_last',
        type=bool,
        default=True,
    )

    '''
        model_args = {
            "classes": 19,
            "m": 3,
            "n": 21,
            "dropOutFlag": False,
            "pretrained": False,
            "cuda": False
        }
    '''

    parser.add_argument(
        '--classes',
        type=int,
        default=19,
    )

    parser.add_argument(
        '--m',
        type=int,
        default=3,
    )

    parser.add_argument(
        '--n',
        type=int,
        default=21,
    )

    parser.add_argument(
        '--dropOutFlag',
        type=bool,
        default=False,
    )

    parser.add_argument(
        '--pretrained_path',
        type=str,
        default="CGNet-PP/pipeline/weights/model_cityscapes_train_on_trainvalset.pth",
    )

    parser.add_argument(
        '--cuda',
        type=bool,
        default=False,
    )

    # criterion_args
    '''
        criterion_args = {
            "weight" : None,
            "ignore_label" : 255
        }
    '''
    parser.add_argument('--ignore_label', type=int, default=255, )

    # ---------------

    # opt args
    parser.add_argument(
        '--last_iter',
        type=int,
        default=-1,
        help="resume last_iter")

    parser.add_argument("--print_lr", dest="print_lr", help="print learning rate", action="store_true", )

    parser.add_argument(
        '--scaleIn',
        type=int,
        default=1,
        help="for input image, default is 1, keep fixed size")

    parser.add_argument(
        '--input_size',
        type=str,
        default="512,1024", help="input size of model")

    parser.add_argument(
        '--random_mirror',
        type=bool,
        default=True,
        help="input image random mirror")

    parser.add_argument(
        '--random_scale',
        type=bool,
        default=True,
        help="input image resize 0.5 to 2")

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help="initial learning rate")

    parser.add_argument(
        '--savedir',
        default="CGNet-PP/checkpoint",
        help="directory to save the model snapshot")

    parser.add_argument(
        '--resume',
        type=str,
        default="CGNet-PP/pipeline/weights/model_cityscapes_train_on_trainvalset.pth",
        help="use this file to load last checkpoint for continuing training")

    parser.add_argument(
        '--inform_data_file',
        default="CGNet-PP/dataset/Cityscapes/cityscapes_inform.pkl",
        help="saving statistic information of the dataset (train+val set), classes weigtht, mean and std")

    parser.add_argument(
        '--M',
        type=int,
        default=3,
        help="the number of blocks in stage 2")

    parser.add_argument(
        '--N',
        type=int,
        default=21,
        help="the number of blocks in stage 3")

    parser.add_argument(
        '--logFile',
        default="CGNet-PP/res/log.txt",
        help="storing the training and validation logs")

    parser.add_argument(
        '--gpus',
        type=str,
        default="0,1",
        help="default GPU devices (0,1)")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    loss_list = main(args)
    reprod_logger = ReprodLogger()
    for idx, loss in enumerate(loss_list):
        reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    reprod_logger.save("CGNet-PP/pipeline/Step4/CGNet_torch/bp_align_paddle.npy")
