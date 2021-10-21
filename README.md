# CGNet

Date: October 13, 2021

# 一、简介

Context Guided Network（CGNet）是由中国科学院计算所Tianyi Wu、Sheng Tang等人提出，通过使用CG block成功训练出了神经网络，网络参数少，准确率在同网络参数个数数量级上效果突出。

**论文：**[CGNet: A Light-weight Context Guided Network for Semantic Segmentation](https://arxiv.org/pdf/1811.08201.pdf)

# 二、复现精度

| 实现       | Backbone | Crop Size  | mIoU  |
| ---------- | -------- | ---------- | ----- |
| CGNet 指标 | M3N21    | 512 * 1024 | 68.27 |
| CGNet 复现 | M3N21    | 512 * 1024 | 72.04 |

# 三、数据集

使用的数据集是 [Cityscapes](https://www.cityscapes-dataset.com/) 

### 下载方法一

[Cityscapes](https://www.cityscapes-dataset.com/) 上下载gtCoarse、gtFine、leftImg8bit文件，解压该文件，将该数据集根据官方repo转化为[19](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py) 个类型。处理完成后对应的文件夹应该具有如下的结构：

```
├── cityscapes_test_list.txt
├── cityscapes_train_list.txt
├── cityscapes_trainval_list.txt
├── cityscapes_val_list.txt
├── cityscapes_val.txt
├── gtCoarse
│   ├── train
│   ├── train_extra
│   └── val
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
├── license.txt
```

后将解压的文件放入 `data/Cityscapes` 文件夹下。

### 下载方法二

在百度aistudio上下载处理好的[数据集](https://aistudio.baidu.com/aistudio/datasetdetail/111446)，直接解压将解压后Cityscapes文件夹中的内容，放入 `data/Cityscapes` 文件夹下，如有文件重名可以进行覆盖。

# 四、环境依赖

- 硬件：CPU、GPU
- 框架：PaddlePaddle ≥ 2.0.0

# 五、快速开始

### step1: clone

```
# clone this repo
git clone https://github.com/632652101/CGNet-PP.git
cd CGNet-PP
```

### step2: 安装依赖

```
pip install -r requirements.txt
```

### step3: 训练

```
python main.py --train
```

### step4: 在val集上测试

```
python main.py --val
```

# 六、其他

### 复现日志文件

复现的日志文件存放在logs/log_reprod文件夹下。

### 权重文件

`model_cityscapes_train_on_trainset.pdparams` 是源作者repo中`model_cityscapes_train_on_trainset.pth`对应的权重文件。

`model_cityscapes_train_on_trainvalset.pdparams` 是源作者repo中`model_cityscapes_train_on_trainvalset.pth`对应的权重文件。

