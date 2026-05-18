# Optimization & Generalization

在训练深度学习模型时，最常遇到的问题就是：**为什么我的 Loss 降不下去？** 或者 **为什么训练集 Loss 很低，一到测试集就拉胯？**

训练模型并不只是一味地加深网络，而是要先分清是 **Optimization（优化）** 出了问题，还是 **Generalization（泛化）** 出了问题。

<!-- more -->

## 1. 核心判断逻辑：看 Loss 

模型在测试集上表现不好，并不一定就是“过拟合（Overfitting）了”，也可能是模型太弱（model bias）的问题

> **黄金法则：永远先看 Training Data 的 Loss！**

* **A：Training Loss 就很高** $\rightarrow$ 这是 **Optimization（优化失败）** 或 **Model Bias（模型太弱）** 的问题。
* **B：Training Loss 很低，但 Testing Loss 很高** $\rightarrow$ 这才是真正的 **Overfitting（过拟合）** 或 **Mismatch（数据分布不一致）**。

## 2. Optimization：为什么 Training Loss 降不下去？

如果在训练集上 Loss 就卡住了，主要有两个可能：

### 2.1 Model Bias（模型偏差）

“大海捞针，但针根本不在海里。”
模型复杂度太低（比如层数太少，或者只是个线性模型），导致它所能表示的 Function Space 里，根本就不存在能够把 Loss 降到最低的那个完美的 Function。

* **方法：** 把模型做大！增加网络的层数或宽度。

### 2.2 Optimization Issue（优化失败）

“针在海里，但你捞不到。”
模型其实足够复杂，Function Space 包含了那个最优解，但是梯度下降（Gradient Descent）算法卡在了半路，找不到它。

这时候，往往是遇到了 **Critical Point（临界点）**，即梯度为 0 的地方。临界点分为两种：

1. **Local Minima（局部最小值）：** 走到谷底了，周围没有更低的地方。
2. **Saddle Point（鞍点）：** 梯度为 0，但并不是真正的最低点，从某些方向看是低谷，从某些方向看是高峰。

#### 解析：如何区分鞍点和局部最小值？

我们需要借助高数中的泰勒展开（Taylor Series）和 **Hessian 矩阵 ($H$)**。在临界点处，一阶导（Gradient）为 0，Loss 的变化完全由二次偏导构成的 Hessian 矩阵决定：

$$ L(\theta) \approx L(\theta') + \frac{1}{2}(\theta - \theta')^T H (\theta - \theta') $$

* **如果 $H$ 的所有特征值 (Eigenvalues) 都大于 0（正定）：** 这是一个 **Local Minima**。无路可走。
* **如果 $H$ 的特征值有正有负：** 这是一个 **Saddle Point**！
* 在高维空间（深度学习中参数动辄几百万）中，几乎不可能所有方向的特征值都大于 0。**所以在深度学习中，大多数卡住的地方其实是鞍点，而不是局部最小值。**

**如何走出鞍点：**
算出 Hessian 矩阵负特征值对应的**特征向量（Eigenvector）**，顺着这个向量的方向更新参数，就能继续让 Loss 下降！（当然，实际工程中算 Hessian 矩阵计算量太大，我们通常用 Momentum 等动量优化器来冲出鞍点）。

---

## 3. Generalization：如何拯救 Overfitting？

如果 Training Loss 已经压得很低了，但测试集表现惨不忍睹，这才是真正的 **Overfitting（过拟合）**。

对付过拟合，有两种方法：

### 3.1 从“数据”下手（最有效）

* **More Training Data：** 数据越多，模型越难死记硬背。
* **Data Augmentation（数据增强）：** 用已有的数据创造“新”数据。比如图像识别中的翻转、裁剪、改变颜色；NLP 中的同义词替换。

### 3.2 从“模型”下手（限制模型的自由度）

不要给模型太大的弹性，让它只能学到共性的规律。

* **减少参数量：** 强行缩减网络层数或神经元个数。
* **Early Stopping（早停）：** 看着 Validation Loss，只要它开始反弹上升，立刻停止训练。
* **Regularization（正则化）：** 在 Loss 函数里加上 L1 或 L2 惩罚项，逼迫模型使用更小、更平滑的权重。
* **Dropout：** 训练时随机“拔掉”一些神经元，强迫剩下的神经元不能过度依赖彼此，练出独立判断的能力。

```python
# PyTorch 中的 Dropout 示例
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.5) # 50% 概率失活
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) # 在全连接层后加入 Dropout
        x = self.fc2(x)
        return x
```