import time

import numpy as np
from PIL import Image
import paddle

from .colorize_mask import cityscapes_colorize_mask


def accuracy(loader,
             model,
             classes=19,
             png_save_dir="images",
             iou_save_dir="logs/mIOU.txt"):
    model.eval()
    data_list = []
    total_len = len(loader)
    for i, (x, label, size, name) in enumerate(loader):
        print(f"{i+1}/{total_len}", end='\r')
        # 防止反向传播的计算
        # input_var = Variable(input, volatile=True)
        input_var = paddle.to_tensor(x, stop_gradient=True)

        # print(input_var.shape)
        output = model(input_var)
        # save seg image
        output = output.cpu().numpy()[0]  # 1xCxHxW ---> CxHxW
        gt = np.asarray(label.numpy()[0], dtype=np.uint8)
        output = output.transpose(1, 2, 0)  # CxHxW --> HxWxC
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

        output_color = cityscapes_colorize_mask(output)
        output = Image.fromarray(output)
        output.save('%s/%s.png' % (png_save_dir, name[0]))
        output_color.save('%s/%s_color.png' % (png_save_dir, name[0]))

    iou_mean, iou_list = _get_iou(data_list, classes, save_path=iou_save_dir)
    print("mIou result saved at " + iou_save_dir)
    return iou_mean, iou_list


def get_iou(data_list, classes=19, save_path=None):
    return _get_iou(data_list, classes, save_path)


def _get_iou(data_list, classes=19, save_path=None):
    from multiprocessing import Pool

    ConfM = ConfusionMatrix(classes)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    if save_path:
        with open(save_path, 'a') as f:
            f.write('\n' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')
    return aveJ, j_list


class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None, ignore_label=255):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))
        self.ignore_label = ignore_label

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == self.ignore_label:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall / self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:  # and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m
