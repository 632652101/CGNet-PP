import os
import sys
import time
import json
import paddle
import numpy as np
import os.path as osp
from mmcv import Config

from utils.dataloader import preprocess, cityscapes
from model import CGNet, loss, optimizer as opt
from utils.eval.evaluate import get_iou, accuracy as acc


def train_one_epoch(model,
                    criterion,
                    optimizer: paddle.optimizer.Optimizer,
                    train_loader,
                    epoch,
                    cfg):
    global lr

    model.train()
    epoch_loss = []
    data_list = []
    total_batches = len(train_loader)

    for iteration, batch in enumerate(train_loader, 0):
        lr = optimizer.get_lr()
        start_time = time.time()

        images, labels, _, _ = batch
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels).astype('int64')
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        epoch_loss.append(loss.cpu().detach().numpy())
        time_taken = time.time() - start_time

        gt = np.asarray(labels.cpu().detach().numpy()[0], dtype=np.uint8)
        output = output.cpu().detach().numpy()[0]
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        data_list.append([gt.flatten(), output.flatten()])

        output_log = 'Epoch: [%d | %d] iter: (%d/%d) \t cur_lr: %.6f loss: %.3f time:%.2f' % (epoch + 1, cfg.train.max_epochs,
                                                                                            iteration + 1,
                                                                                            total_batches, lr,
                                                                                            loss.cpu().detach().numpy(),
                                                                                            time_taken)
        print(output_log)
        sys.stdout.flush()

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    mean_iou, per_class_iou = get_iou(data_list)
    output_log = 'mIOU: %.6f, \nper_class_iou: ' % mean_iou + str(per_class_iou)
    print(output_log)
    sys.stdout.flush()

    return average_epoch_loss_train, per_class_iou, mean_iou, lr


def train(args):
    cfg = Config.fromfile(args.configfile)
    print(json.dumps(cfg._cfg_dict, indent=4))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = "weights/checkpoints"

    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    print("Checkpoint path: %s." % checkpoint_path)
    sys.stdout.flush()

    # data loader
    if args.trainval:
        dataset = cityscapes.get_dataset_train(**cfg.data.set.trainval)
    else:
        dataset = cityscapes.get_dataset_train(**cfg.data.set.train)

    dataloader = cityscapes.get_dataloader(dataset, **cfg.data.loader.train)

    # device
    device = paddle.get_device()
    paddle.set_device(device)

    # model
    model = CGNet.cgnet(**cfg.model.backbone)

    # criterion
    x = preprocess.get_inform_data(args.inform_data_file)
    weight = paddle.to_tensor(x['classWeights'])
    criterion = loss.CrossEntropyLoss2d(weight=weight, ignore_label=255)

    # optimizer
    optimize, schedule = opt.get_optimizer(model, **cfg.train.opt)

    max_epochs = cfg.train.max_epochs
    start_epochs = cfg.train.resume.last_epoch + 1

    for epoch in range(start_epochs, max_epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, max_epochs))
        # training
        train_loss, _, mean_iou, lr = train_one_epoch(model,
                                                      criterion,
                                                      optimize,
                                                      dataloader,
                                                      epoch,
                                                      cfg)
        schedule.step()

        # save the model
        model_file_name = checkpoint_path + '/model_' + str(epoch + 1) + '_mean_iou_' + "%.4f" % (
            mean_iou) + "_lr_%.6f_" % (lr) + '.pdparams'
        state = model.state_dict()
        paddle.save(state, model_file_name)
