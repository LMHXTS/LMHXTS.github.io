# 决策树算法

### 核心原理

决策树是一种树形结构的监督学习模型，可用于分类与回归任务。它通过特征选择 → 树生成 → 剪枝三个步骤构建模型，本文主要进行前两个步骤，决策树的核心在于根据特征分类，找到一个特征，使得根据该特征分类后，子集的‘纯度’最高。

决策树的训练复杂度大致为 $O(N \cdot M \cdot d)$，其中 $N$ 是样本数，$M$ 是特征数。虽然训练快，但如果 $d$ 过大，模型的方差会剧增。

ID3 (Iterative Dichotomiser 3)：使用 信息增益 (Information Gain) 作为选择特征的标准。它的核心思想是：选择那个能让数据集信息熵下降最快的特征。

C4.5：是对 ID3 的改进版。它改用 信息增益率 (Gain Ratio)，旨在解决 ID3 倾向于选择取值较多的特征的缺陷。

---

### 数学推导

1. 信息熵 (Entropy)：  
首先，我们需要量化数据集 $D$ 的混乱程度。假设 $D$ 中有 $K$ 个类别，第 $k$ 类样本所占比例为 $p_k$，则熵定义为：
$$H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

2. 信息增益 (Information Gain)：  
   假设我们选择特征 $A$来划分 $D$，特征 $A$ 有 $V$ 个可能的取值。根据 $A$ 的取值将 $D$ 划分为 $V$ 个子集 $\{D^1, D^2, \dots, D^V\}$。特征 $A$ 对数据集 $D$ 的信息增益定义为：
   $$g(D, A) = H(D) - \sum_{v=1}^{V} \frac{|D^v|}{|D|} H(D^v)$$
   这里 $\sum_{v=1}^{V} \frac{|D^v|}{|D|} H(D^v)$ 是在特征 $A$ 给定的条件下，数据集 $D$ 的条件熵。增益越大，说明使用 $A$ 进行分类获得的“知识”越多。

3. 信息增益率 (Gain Ratio)：  
   ID3 的局限性：如果一个特征中每个可能的取值都不同，那么每个子集只包含一个样本，熵全部为 0。ID3 会认为这个特征的信息增益极大，但这种分类毫无泛化能力。  
    所以C4.5算法中引入了特征熵，也叫分裂信息：
    $$IV(A) = -\sum_{v=1}^{V} \frac{|D^v|}{|D|} \log_2 \frac{|D^v|}{|D|}$$
    进而定义增益率为：
    $$g_R(D, A) = \frac{g(D, A)}{IV(A)}$$
    通过除以 $IV(A)$，特征取值越多，$IV(A)$ 越大，增益率就会被相应惩罚。


---

### 代码实现

#### ID3: 

```python
   # 计算数据集的总体信息熵  dataset: 数据集
def cal_total_entropy(dataset):
    count = dataset[TARGET].value_counts()
    prob = count / len(dataset[TARGET])
    total_entropy = np.sum(-prob * np.log2(prob))
    return total_entropy

#计算基于特定特征的条件信息熵（加权平均熵）  feature：特征
def cal_conditional_entropy(dataset, feature):
    conditional_entropy = 0
    for value in dataset[feature].unique():
        subset = dataset[dataset[feature] == value]
        count = subset[TARGET].value_counts()
        prob = count / len(subset)
        # 加权求和：(子集大小/总数据集大小) * 子集熵
        conditional_entropy += (len(subset) / len(dataset)) * np.sum(-prob * np.log2(prob))
    return conditional_entropy

#计算基于特定特征的信息增益
def cal_gain(dataset, feature):
    total_entropy = cal_total_entropy(dataset)
    conditional_entropy = cal_conditional_entropy(dataset, feature)
    information_gain = total_entropy - conditional_entropy
    return information_gain
```

#### C4.5:

```python
#计算分裂信息（Split Information）
def cal_split_information(dataset, feature):  
    split_info = 0
    for value in dataset[feature].unique():
        subset = dataset[dataset[feature] == value]
        proportion = len(subset) / len(dataset)
        if proportion > 0:  # 避免log(0)的问题
            split_info -= proportion * np.log2(proportion)
    return split_info

#计算信息增益率 
def cal_gain_ratio(dataset, feature):
    information_gain = cal_gain(dataset, feature)
    split_information = cal_split_information(dataset, feature)
    # 避免分母为0
    if split_information == 0:
        return 0
    gain_ratio = information_gain / split_information
    return gain_ratio
```
#### 决策树搭建:

```python 
def build_tree(dataset,features):
    majority = dataset[TARGET].mode()[0]
    node = Treenode(majority)
    if len(dataset[TARGET].unique()) == 1:
        node.leaf = True
        node.label = dataset[TARGET].iloc[0]
        return node
    elif len(features) == 0:
        node.leaf = True
        return node
    else:
        gains = [cal_gain(dataset,f) for f in features]   
        #C4.5:   gains = [cal_gain_ratio(dataset,f) for f in features]
        max_feature = features[np.argmax(gains)]
        node.feature = max_feature
        for val in dataset[max_feature].unique():
            subset = dataset[dataset[max_feature] == val]
            if len(subset) == 0:
                child = Treenode(majority)
                child.leaf = True
                node.children[val] = child
            else:
                delete_f = [f for f in features if f != max_feature]
                node.children[val] = build_tree(subset,delete_f)
        return node 
```
---
### 算法分析

决策树的优势在于他的可解释性，可以根据训练结果直接绘制出相应的决策树；数据也无需预处理，不需要进行归一化等操作，它只根据特征进行分类排序。

但决策树也具有很明显的劣势，极容易针对数据过拟合，训练集中每一个杂志的扰动都会对模型训练造成影响，这就需要进行后剪枝或升级为cart算法（现代工业界XGBoost, LightGBM 等集成算法中最流行的基学习器，后续blog中会提到）；  
其次决策树只能对数据进行水平或竖直的切割，若真实边界是斜线（如X - Y > 100），决策树需要无数个‘台阶’去逼近它，导致效率极低（可以通过随机森林（Random Forest）或梯度提升树（XGBoost/LightGBM）来解决，后续blog中会提到）。