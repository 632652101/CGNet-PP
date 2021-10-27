# CGNet

Date: October 13, 2021

# 一、简介

Context Guided Network（CGNet）是由中国科学院计算所Tianyi Wu、Sheng Tang等人提出，通过使用CG block成功训练出了神经网络，网络参数少，准确率在同网络参数个数数量级上效果突出。

**论文：**[CGNet: A Light-weight Context Guided Network for Semantic Segmentation](https://arxiv.org/pdf/1811.08201.pdf)

# 二、复现精度

| 实现       | Backbone | Crop Size  | mIoU  |
| ---------- | -------- | ---------- | ----- |
| CGNet 指标 | M3N21    | 512 * 1024 | 68.27 |
| CGNet 复现 | M3N21    | 512 * 1024 | 68.89 |

# 三、数据集

使用的数据集是 [Cityscapes](https://www.cityscapes-dataset.com/) 

### 下载方法一

[Cityscapes](https://www.cityscapes-dataset.com/) 上下载gtCoarse、gtFine、leftImg8bit文件，解压该文件，将该数据集根据官方repo转化为[19](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py) 个类型。处理完成后对应的文件夹应该具有如下的结构：

```
├── cityscapes_test_list.txt
├── cityscapes_train_list.txt
├── cityscapes_trainval_list.txt
├── cityscapes_val_list.txt
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

# 六、代码结构与详细说明

### 6.1 代码结构

```
├─config                          # 配置
│   └── cgnet
│       └── M3N21_512x1024.py     # 配置权重文件，训练超参数,epoch,lr等
├─data                            # 数据集 
├─eval                            # 评估脚本
├─model                           # 模型
├─images                          # 结果图片
├─utils                           # 工具代码
├─logs                            # 日志文件，复现的日志文件存放在logs/log_reprod文件夹下。
│  val.py                         # 评估
│  README.md                      # readme
│  requirements.txt               # 依赖
│  train.py                       # 训练
│  main.py                        # 主函数

```

### 6.2 参数说明

可以在config/cgnet/M3N21_512x1024.py 中设置训练与评估相关参数，具体如下：

| 参数                      | 默认值                             | 说明                 | 注意事项                                        |
| ------------------------- | ---------------------------------- | -------------------- | ----------------------------------------------- |
| model.backbone.pretrained | M3N21_512x1024_top2.pdparams       | 预训练模型参数路径   | 可以替换成weights/checkpoints里面训练得到的参数 |
| data                      | 默认参考Cityscapes数据集进行配置。 | 数据加载参数         | 通常不需要修改                                  |
| train                     | lr=0.001 max_epoch=360             | 训练参数，优化器参数 | resume设置上一轮从哪里开始                      |

### 6.3 训练与测试流程

#### 训练

```
# 数据集解压
mkdir data
mkdir images
cp /home/aistudio/data/data111446/dataset.zip /home/aistudio/work/CGNet-PP-main/data 
unzip -q /home/aistudio/work/CGNet-PP-main/data/dataset.zip -d /home/aistudio/work/CGNet-PP-main/data

# 安装依赖
pip install mmcv

# 开始训练
python main.py --train
```

训练过程中，会显示每一轮`batch`训练，将会打印当前epoch、step以及loss值。最终训练得到的参数会保存在weights/checkpoints里，例子：

```
Epoch: [1 | 360] iter: (1/371) 	 cur_lr: 0.001000 loss: 3.670 time:0.47
mIOU: 0.233958, 
per_class_iou: [0.7918437084344949, ...]
```

#### 恢复训练

1. 修改config文件中的 `model.backbone.pretrained` 参数为上次保存的模型参数路径（默认在weights/checkpoints/ 文件夹下）。
2. 修改config文件中的 `train.opt.last_epoch` 为上次结束的epoch数`train.resume.last_epoch`也为上次结束的epoch数。

#### 测试

```
# 数据集解压
mkdir data
mkdir images
cp /home/aistudio/data/data111446/dataset.zip /home/aistudio/work/CGNet-PP-main/data 
unzip -q /home/aistudio/work/CGNet-PP-main/data/dataset.zip -d /home/aistudio/work/CGNet-PP-main/data

# 安装依赖
pip install mmcv

# 开始在val集上测试
python main.py --val
```

测试结果生成的日志文件会自动保存在 logs/mIoU.txt 里面，例子：

```
2021-10-13 22:08:50
meanIOU: 0.6889543175845664
[0.9702754299143364, 0.7920537139625758, ...]
```



# 七、模型信息

| 信息     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| 发布者   | Qijing yuan、yuxing wang                                     |
| 更新时间 | 2021.10                                                      |
| 框架版本 | Paddle 2.1.2                                                 |
| 应用场景 | 语义分割                                                     |
| 支持硬件 | CPU,GPU                                                      |
| 在线运行 | https://aistudio.baidu.com/aistudio/projectdetail/2526211?shared=1 |

