# BP神经网络

### 核心原理

#### 模型构建

在逻辑回归中，既然一个神经元只能画一条直线，那我们就把多个神经元并排放在一起，构成一个隐藏层 。

假设输入特征矩阵是 $X$，我们引入一个隐藏层和一个输出层：

1. **输入层** 接收原始数据。
2. **隐藏层** 提取特征，输出 $A^{[1]}$。
3. **输出层** 根据隐藏层的特征进行最终分类，输出 $\hat{Y}$（或 $A^{[2]}$）。

但如果只做矩阵乘法 $W^{[2]}(W^{[1]}X)$，根据矩阵乘法的结合律，它等效于 $(W^{[2]}W^{[1]})X$。这说明如果不加激活函数，无论叠多少层，本质上依然是一个单层线性变换。

因此，我们需要在隐藏层引入非线性激活函数 Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

#### 前向传播

假设我们有 $m$ 个样本，每个样本 $n_0$ 个特征。隐藏层有 $n_1$ 个神经元，输出层有 $n_2$ 个神经元。
（大写字母代表包含所有 $m$ 个样本的矩阵）

**1. 隐藏层计算：**

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$

$$A^{[1]} = \sigma(Z^{[1]})$$

* $X$ 的维度：$(n_0, m)$
* $W^{[1]}$ 的维度：$(n_1, n_0)$
* $Z^{[1]}$ 和 $A^{[1]}$ 的维度：$(n_1, m)$

这里当输入 $x$ 是一个很大的负数时，$e^{-x}$ 会变得极其庞大，导致指数爆炸。在代码中可以添加clip限制

**2. 输出层计算：**

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$

$$A^{[2]} = \sigma(Z^{[2]})$$

* $W^{[2]}$ 的维度：$(n_2, n_1)$
* $A^{[2]}$（即预测结果 $\hat{Y}$）的维度：$(n_2, m)$

运用多次矩阵乘法即可完成前向传播

#### 反向传播 (BP) 

前向传播只是算出一个结果，而神经网络真正学会知识的过程，在于**反向传播**——通过计算损失函数对每个参数的偏导数（梯度），来指导参数的更新。

假设我们有 $m$ 个样本，采用交叉熵损失函数 $L = -\sum y \ln(a)$。

1. 输出层误差与梯度：

    $$dZ^{[2]} = A^{[2]} - Y$$

    $$dW^{[2]} = \frac{1}{m} (A^{[1]})^T dZ^{[2]}$$

    $$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

2. 隐藏层误差与梯度：误差传回隐藏层，需要乘上当前层激活函数的导数：

    $$dZ^{[1]} = (dZ^{[2]} (W^{[2]})^T) * \sigma'(Z^{[1]})$$
  
    $$dW^{[1]} = \frac{1}{m} X^T dZ^{[1]}$$

    $$db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$

#### $dZ$的推导

交叉熵函数为：

$$L = -[y \ln(a) + (1 - y) \ln(1 - a)]$$

对 $a$ 求导：

$$\frac{\partial L}{\partial a} = -\left[ y \cdot \frac{1}{a} + (1 - y) \cdot \frac{1}{1 - a} \cdot (-1) \right]$$

$$\frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1 - y}{1 - a}$$

通分：

$$\frac{\partial L}{\partial a} = \frac{-y(1 - a) + a(1 - y)}{a(1 - a)} = \frac{-y + ay + a - ay}{a(1 - a)} = \frac{a - y}{a(1 - a)}$$

激活函数 $a$ 对 $z$ 求偏导：

$$a = (1 + e^{-z})^{-1}$$

$$\frac{\partial a}{\partial z} = -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

化简：

$$\frac{\partial a}{\partial z} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = a \cdot (1 - a)$$

$$ dz = \frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} = \frac{a - y}{a(1 - a)} \cdot a(1 - a)$$

最终结果极其简洁：

$$ dz = a - y$$

矩阵化后：  

$$dZ^{[2]} = A^{[2]} - Y$$

#### Kaiming 初始化

如果把所有权重初始化为 0，那么隐藏层的所有神经元将会进行完全相同的计算，计算出相同的梯度，并在每次迭代后保持相同的值。这就破坏了神经网络的非对称性。

所以通常用微小的随机数进行初始化。但是，普通的随机初始化很容易导致信号在多层传播后急剧衰减（梯度消失）或爆炸。

它的核心思想是让每一层输入和输出的方差保持一致。通过严格的数学推导，当权重初始化满足均值为 0，方差为 $\frac{2}{n_{in}}$ 时，网络表现最好

（我的概率论与数理统计论文内容就为Kaiming初始化）

---

### 代码实现

#### 前向传播

```python
# Sigmoid 阈值函数
    def sigmod(self, x):
        # 为了防止指数爆炸，使用 np.clip 限制输入范围
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # 前向传播
    def forword(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmod(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmod(self.z2)
        return self.a2
```

#### 反向传播

```python
    def backword(self, X, y):
        m = X.shape[0]
        # 1. 输出层误差
        dz2 = self.a2 - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 2. 隐藏层误差回传
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.d_sigmod(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 3. 参数更新
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
```

#### 训练过程

```python
epochs = 10
    batch_size = 100
    # 训练循环
    for epoch in range(epochs):
        # 打乱数据集，防止模型产生顺序依赖
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # 前向计算与反向更新
            bpnn.forword(X_batch)
            bpnn.backword(X_batch, y_batch)
```

#### pytorch实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BPNeuralNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=120, output_size=10):
        super(BPNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

def train_pytorch_model():
    #训练集导入省略
    model = BPNeuralNetwork(input_size=784, hidden_size=120, output_size=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epochs = 10
    history_data = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0 
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        history_data['loss'].append(avg_loss)
```