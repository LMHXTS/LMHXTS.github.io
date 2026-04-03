# 贝叶斯分类算法：从概率论到代码实现

### 核心原理

贝叶斯分类器的灵魂在于**贝叶斯定理（Bayes' Theorem）**，它提供了一种通过先验概率和似然估计来计算后验概率的机制。在机器学习的语境下，就是在已知特征 $X$ 的前提下，反推该样本属于类别 $Y$ 的概率。

而**“朴素”（Naive）**二字，源于一个极其强势且理想化的假设：**条件独立性假设**。即假设在给定目标值时，各个特征之间是相互独立的。

---

### 数学推导

给定特征向量 $X = [x_1, x_2, ..., x_n]$ 和类别 $Y = c$，根据贝叶斯定理，后验概率计算如下：

$$P(Y=c|X) = \frac{P(X|Y=c)P(Y=c)}{P(X)}$$

由于对于所有的类别 $c$，分母 $P(X)$ 都是相同的，我们在比较大小时可以将其忽略，只求分子最大化：

$$Y_{predict} = \arg\max_{c} P(X|Y=c)P(Y=c)$$

引入**朴素假设**（特征条件独立），联合似然可以拆解为各个特征似然的连乘：

$$P(X|Y=c) = \prod_{i=1}^{n} P(x_i|Y=c)$$

我们假设特征服从**高斯分布（正态分布）**。对于类别 $c$ 中的第 $i$ 个特征，其概率密度函数为：

$$P(x_i|Y=c) = \frac{1}{\sqrt{2\pi\sigma_{c,i}^2}} \exp\left(-\frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2}\right)$$

为了防止多个极小的概率值连乘导致**计算机浮点数下溢（Underflow）**，我们通常对两边取自然对数，将连乘转化为连加：

$$\log P(Y=c|X) \propto \log P(Y=c) + \sum_{i=1}^{n} \log P(x_i|Y=c)$$

这就是在代码中实际优化的数学目标。

---

### 代码实现

纯手工用 `NumPy` 搭建一个高斯朴素贝叶斯分类器。

```python
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        """
        训练模型：计算每个类别的先验概率、特征均值和方差
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # 初始化均值、方差和先验概率矩阵
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            # 加上一个极小的epsilon防止方差为0除以0
            self._var[idx, :] = X_c.var(axis=0) + 1e-9 
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx, x):
        """
        计算高斯概率密度函数 (PDF)
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        """预测样本类别"""
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """计算单个样本属于各类的后验概率，返回最大概率对应的类别"""
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            # 将连乘转换为对数相加
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # 返回具有最大后验概率的类别
        return self._classes[np.argmax(posteriors)]
