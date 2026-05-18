---
title: CNN 的破局之道：感受野与参数共享的艺术
description: 李宏毅深度学习2022课程笔记 - 为什么图像处理要用CNN？从全连接层的退化讲起
---

# 图像的视觉基石：从全连接到 CNN

!!! abstract "核心导读"
    在处理图像任务时，我们为什么不直接用全连接神经网络（Fully Connected Network）？   
    一张普通的 $100 \times 100$ 彩色图片，展开后有 $100 \times 100 \times 3 = 30000$ 个特征。如果隐藏层有 1000 个神经元，仅这一层的权重（Weight）就高达 **3000万** 个！参数量爆炸不仅会导致计算极其缓慢，更容易引发严重的**过拟合（Overfitting）**。    
    李宏毅老师指出：**CNN 并不是什么神奇的新物种，它其实是一个被“阉割”过的全连接层。** 因为图像具备某些特定的物理特性（Spatial Prior），我们故意去掉了全连接层中很多不必要的连接，从而诞生了高效率的 CNN。

---

## 1. 为什么 CNN 敢于“阉割”网络？图像的三个先验假设

当我们人类在看一张图片并判断“这是一只鸟”时，我们并不是在看图片的每一个像素点。CNN 的设计正是基于这种人类视觉的直觉。

### 假设一：局部特征足够判断（Receptive Field）
要识别一只鸟，神经元不需要看整张图片，它只要看到“鸟嘴”或者“爪子”这一个局部区域，就能做出判断。
因此，CNN 引入了 **感受野（Receptive Field）** 的概念。每一个神经元（或者说卷积核的一个通道）只连接到输入图像的一个局部小窗口（比如 $3 \times 3$ 的矩阵），而不是连接所有像素。



### 假设二：同样的特征会出现在不同位置（Parameter Sharing）
“鸟嘴”可能出现在图片的左上角，也可能出现在右下角。探测“鸟嘴”的那个神经元，无论在图片的哪个位置，它的工作原理是一样的。
所以，CNN 引入了 **参数共享（Parameter Sharing）**。同一个过滤器（Filter/Kernel）会在整张图片上滑动（Sliding），无论滑到哪里，这个过滤器的权重参数都是完全一样的。这极大地减少了模型所需的参数量！



### 假设三：缩放不改变物体本质（Subsampling）
把一张高分辨率的猫的图片，缩小成原来的一半，人眼依然能认出那是一只猫。
基于此，CNN 引入了 **池化层（Pooling）**。通过下采样（比如保留 $2 \times 2$ 区域内的最大值），在保留核心特征的同时，大幅减少特征图（Feature Map）的尺寸，进一步降低运算量。

[Image demonstrating max pooling operation in CNN reducing a 4x4 matrix to a 2x2 matrix]

---

## 2. CNN 的标准流水线架构

将上述三个概念组合起来，就构成了我们在代码中最常写的 CNN 架构流水线：

1. **Convolution（卷积层）：** 提取局部特征（对应感受野与参数共享）。
2. **Activation（激活函数）：** 通常使用 ReLU，加入非线性表达能力。
3. **Pooling（池化层）：** 降维，提取主要特征，增加平移不变性。
4. **Flatten（展平）：** 将三维的特征图拉平变成一维向量。
5. **Fully Connected（全连接层）：** 综合所有提取到的高级特征，输出最终的分类概率。

!!! info "数学速查：特征图尺寸计算"
    在手写网络时，最痛苦的莫过于算不准卷积后的尺寸。假设输入尺寸为 $N$，卷积核大小为 $F$，步长（Stride）为 $S$，边缘填充（Padding）为 $P$，那么输出尺寸 $O$ 的计算公式为：

$$ O = \frac{N - F + 2P}{S} + 1 $$

---

## 3. PyTorch 实战：手写一个标准 CNN Block

理解了原理，落实到代码上就非常自然了。下面是一个标准的包含卷积、激活、池化的 CNN 模块：

```python title="cnn_block.py"
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv2d 参数：输入通道数, 输出通道数(Filter数量), 卷积核大小, 步长, 填充
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 尺寸缩小一半
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 假设输入图片为 32x32，经过两次 MaxPool 后尺寸变为 8x8
        self.fc = nn.Linear(32 * 8 * 8, 10) # 展平后接入全连接层进行10分类

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1) # Flatten 操作，相当于 nn.Flatten()
        x = self.fc(x)
        return x