# K-近邻算法

### 核心原理

K-近邻算法是机器学习中最简单的一类算法，是一种基本的分类与回归方法。  
主要思想：假定给定一个训练数据集，其中实例标签已定，当输入新的实例时，可以根据其最近的k个训练实例的标签，预测新实例对应的标注信息。  
三要素：距离度量（一般采用欧式距离）、 K值选择 、 分类决策规则（如多数表决）  
在实际实现中，我们不可能对每一个实例都去计算其到每个点的欧氏距离，因此，引入了kd树对k维空间中的实例点进行存储以便对其进行快速检索。本质上是使用二叉树的思路对k维空间进行划分。详情见BV1No4y1o7ac，

---

### 数学推导

切分维度的选择：虽然简单的 kd树会按坐标轴顺序（即 $1, 2, ..., k, 1, 2...$）轮流切分，但更科学的做法是每次选择方差最大的维度进行切分。设当前节点包含 $m$ 个样本，对于第 $j$ 个维度，其方差为：

$$S_j = \frac{1}{m}\sum_{i=1}^{m}(x_{i}^{(j)} - \bar{x}^{(j)})^2$$

其中 $\bar{x}^{(j)}$ 是该维度上的均值。选择最大的 $S_j$ 对应的维度进行切分，可以让空间划分得更加均匀。

回溯的几何判定条件： 假设我们正在寻找目标点 $x_{target}$ 的最近邻，当前找到的最短距离为 $d_{best}$。此时我们回溯到了某个父节点 $x_{node}$，该父节点在第 $j$ 维度的坐标为 $x_{node}^{(j)}$，它代表了一个切分超平面。目标点 $x_{target}$ 到这个切分超平面的垂直距离为：

$$d_{plane} = |x_{target}^{(j)} - x_{node}^{(j)}|$$


如果 $d_{plane} < d_{best}$，说明以目标点为圆心、$d_{best}$ 为半径的超球体，跨越了该父节点构成的超平面。因此，我们必须进入该父节点的另一个子树去进行搜索。如果 $d_{plane} \ge d_{best}$，说明另一侧空间绝不可能存在更近的点，我们可以直接剪枝（Pruning），放弃搜索那一半的树。

---

### 代码实现

```python
import numpy as np
import heapq

#用KD树实现KNN算法
#point : 划分点  axis ：划分轴
class KDnode:
    def __init__(self, point, left=None, right=None, axis=0):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis

#构建KD树
def build_kd_tree(points, depth=0):
    if len(points) == 0:
        return None
    k = len(points[0])
    axis = depth % k
    #根据当前轴对点进行排序
    sorted_points = sorted(points, key=lambda x: x[axis])
    mid = len(sorted_points) // 2
    #递归构建KD树
    return KDnode(
        point = sorted_points[mid],
        left = build_kd_tree(sorted_points[:mid], depth+1),
        right = build_kd_tree(sorted_points[mid+1:], depth+1),
        axis = axis
    )

#计算两点之间的欧氏距离
def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

#搜索KD树
def knn_search(node, target, k, current_best=None):

    if current_best is None:
        current_best = []
    
    if node is None:
        return current_best
    
    axis = node.axis
    if target[axis] < node.point[axis]:
        next_branch = node.left
        opposite_branch = node.right
    else:
        next_branch = node.right
        opposite_branch = node.left
    # 先搜索接近的分支
    knn_search(next_branch, target, k, current_best)
    # 回溯
    dist = distance(node.point, target)
    current_best.append((dist, node.point))
    current_best.sort(key=lambda x: x[0])
    # 只保留k个最近的点
    if len(current_best) > k:
        current_best.pop()
    # 检查是否需要搜索对面分支
    if len(current_best) < k or abs(target[axis] - node.point[axis]) < current_best[-1][0]:
        knn_search(opposite_branch, target, k, current_best)
    
    return current_best

#查询
def knn_query(tree, target, k):
    results = knn_search(tree, target, k, [])
    return results
```

---

### 算法分析

从K-NN原理上可以看出，k-近邻算法更适合解决特征维度较少、且特征与任务强相关的分类或回归问题。其在 2 维或 3 维数据下表现优异（接近 $O(\log N)$）。但一旦当数据的维度非常高，kd树的性能会急剧退化，甚至变得比暴力搜索还要慢。  
若只是特征少，K-NN也未必能完美应对，K-NN本身没有任何的权重筛选能力，因此对特征极其敏感，无法承受噪声干扰，这也是为什么会引入决策树的原因。  
虽然K-NN算法早已被扫进了ai历史博物馆，但他的思路却为大模型提供了启发，现在的语言模型虽然聪明，但它们会“幻觉”，且不知道企业内部的私有数据。工业界目前的标准解法是 RAG（检索增强生成）。它就是将将所有的企业文档、维基百科通过 Embedding 模型转化为高维向量。当用户提问时，也转化为高维向量。然后，在海量文档库中寻找距离用户问题“最近的 k 个文档片段”，喂给大模型作为上下文。这就是彻头彻尾的 k-NN 算法， 只不过，面对几十亿的向量，我们不再追求“绝对精确”的kd树，而是使用近似最近邻算法ANN，牺牲 1% 的精度，换取万倍的搜索速度提升。