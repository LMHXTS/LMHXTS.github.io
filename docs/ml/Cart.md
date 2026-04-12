# CART算法
 
### 核心原理

CART（Classification and Regression Trees），全称分类与回归树，从名称中就能看出来，它也是一种基于决策树的算法，只不过在ID3与C4.5的基础上，能够同时处理分类问题与回归问题。  
CART 的本质是一个二叉树。无论面对的是分类还是回归问题，CART 都会在特征空间中不断地进行二分，每次分裂，算法都会遍历所有特征的的所有可能切分点，寻找一个“最优切分点”，将当前数据集划分为两部分（左子树和右子树）。这个过程是递归的，直到满足停止条件（如达到最大深度，或节点内样本过少）。  
**对于分类问题**：模型希望切分后的两个子集内部的类别尽可能“纯粹”。叶子节点的预测值为该节点内出现次数最多的类别，这时就要用到**基尼指数**。  
**对于回归问题**：模型希望切分后的两个子集内部的预测误差尽可能小。叶子节点的预测值为该节点内所有样本目标值的平均数，用**平方误差**来解决。

---

### 数学推导

#### 分类树：基尼指数（Gini）
在ID3与C4.5算法中，都使用了信息熵作为衡量信息“纯度”的标准，但其中使用的对数运算在计算机中计算成本较高，在CART树中，使用基尼指数来评判。  
假设数据集 $D$ 中有 $K$ 个类别，第 $k$ 个类别在数据集中出现的概率为 $p_k$，则数据集 $D$ 的基尼不纯度定义为：

$$Gini(D) = \sum_{k=1}^{K} p_k (1 - p_k) = 1 - \sum_{k=1}^{K} p_k^2$$

如果我们在特征 $A$ 的某个切分值 $a$ 处将数据集 $D$ 分裂为 $D_1$（满足 $A \le a$）和 $D_2$（满足 $A > a$），那么分裂后的基尼指数（条件基尼指数）为： 

$$Gini(D, A) = \frac{|D_1|}{|D|} Gini(D_1) + \frac{|D_2|}{|D|} Gini(D_2)$$

优化目标：寻找特征 $A$ 和切分点 $a$，使得 $Gini(D, A)$ 达到最小。

#### 回归树：平方误差

回归问题处理的是连续值，我们无法计算概率和基尼指数。CART 回归树采用启发式的方法，目标是最小化平方误差。假设我们选择第 $j$ 个特征 $x^{(j)}$ 和它的取值 $s$ 作为切分变量和切分点，将特征空间划分为两个区域： 

$$R_1(j, s) = \{x | x^{(j)} \le s\}$$

$$R_2(j, s) = \{x | x^{(j)} > s\}$$

我们希望找到两个区域的固定预测值 $c_1$ 和 $c_2$，使得每个区域内的样本到这个预测值的均方误差最小。根据微积分知识，使得 $\sum (y_i - c)^2$ 最小的 $c$ 就是该区域内所有 $y_i$ 的均值。因此，优化目标变为寻找最优的 $j$ 和 $s$，求解：

$$\min_{j, s} \left[ \min_{c_1} \sum_{x_i \in R_1(j,s)} (y_i - c_1)^2 + \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2 \right]$$

其中 $\hat{c}_1 = \text{mean}(y_i | x_i \in R_1(j,s))$，$\hat{c}_2 = \text{mean}(y_i | x_i \in R_2(j,s))$。

#### CART树的剪枝

为了衡量树的好坏，CART 定义了代价复杂度损失函数：

$$C_\alpha(T) = C(T) + \alpha |T|$$

$T$：代表任意一棵子树。  
$C(T)$：代表这棵树在训练集上的预测误差（分类中的基尼指数，回归中的平方误差和）。  
$|T|$：代表这棵子树的叶子节点总数。  
$\alpha \ge 0$：复杂度惩罚系数。它决定了我们对“臃肿”的容忍度。  
现在，我们把目光聚焦到树中的任意一个内部节点 $t$。如果不剪枝，以 $t$ 为根节点的子树记为 $T_t$。它的损失为：

$$C_\alpha(T_t) = C(T_t) + \alpha |T_t|$$

如果进行剪枝，把 $t$ 变成一个叶子节点。它的损失为：

$$C_\alpha(t) = C(t) + \alpha \cdot 1$$

显然，如果不加惩罚（$\alpha = 0$），子树的误差肯定比单个叶子节点小（$C(T_t) < C(t)$）。但随着 $\alpha$ 不断增大，复杂的子树受到的惩罚 $\alpha |T_t|$ 会越来越重。当 $\alpha$ 大到某个临界值时，剪枝和不剪枝的损失会相等：

$$C(T_t) + \alpha |T_t| = C(t) + \alpha$$

解这个方程，我们能求出这个节点的剪枝临界值 $g(t)$：

$$g(t) = \frac{C(t) - C(T_t)}{|T_t| - 1}$$

$g(t)$ 越小，说明这个分支为了减少那么一丁点误差，却长出了非常多的叶子，它就应该被最先剪掉。

---

### 代码实现

#### CART树构建
CART树构建与ID3，C4.5类似，只不过需要限定最小分类样本数与最大深度防止过拟合
```python

def __init__(self, min_samples_split=2, max_depth=float("inf")):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

def split_dataset(self, X, y, feature_i, threshold):
        """根据特征和阈值将数据集一分为二"""
        split_func = lambda x: x[feature_i] <= threshold
        left_indices = np.array([split_func(x) for x in X])
        
        return X[left_indices], y[left_indices], X[~left_indices], y[~left_indices]

    def build_tree(self, X, y, current_depth=0):
        """递归构建二叉树"""
        n_samples, n_features = np.shape(X)
        best_criteria = None
        best_sets = None
        
        # 判断是否满足分裂条件
        if n_samples >= self.min_samples_split and current_depth < self.max_depth:
            best_impurity_reduction = 0
            
            # 遍历所有特征和对应特征的所有唯一取值
            for feature_i in range(n_features):
                unique_values = np.unique(X[:, feature_i])
                for threshold in unique_values:
                    # 尝试切分
                    X_left, y_left, X_right, y_right = self.split_dataset(X, y, feature_i, threshold)
                    
                    if len(X_left) > 0 and len(X_right) > 0:
                        # 计算不纯度的减少量（信息增益的等价概念）
                        impurity_reduction = self.calculate_impurity_reduction(y, y_left, y_right)
                        
                        # 寻找最大不纯度减少量（即最小化分裂后的损失）
                        if impurity_reduction > best_impurity_reduction:
                            best_impurity_reduction = impurity_reduction
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "left_X": X_left, "left_y": y_left,
                                "right_X": X_right, "right_y": y_right
                            }
            # 如果找到了有效的最佳分裂
        if best_impurity_reduction > 0:
            left_branch = self.build_tree(best_sets["left_X"], best_sets["left_y"], current_depth + 1)
            right_branch = self.build_tree(best_sets["right_X"], best_sets["right_y"], current_depth + 1)
            return Node(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"],
                        left_branch=left_branch, right_branch=right_branch)
        
        # 否则，成为叶子节点，计算叶子节点的预测值
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)
```

#### 分类树

```python
class CARTClassifier(CARTBase):
    def calculate_gini(self, y):      """计算基尼指数"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def calculate_impurity_reduction(self, y, y_left, y_right):       """计算基尼指数的减少量"""
        p = len(y_left) / len(y)
        return self.calculate_gini(y) - (p * self.calculate_gini(y_left) + (1 - p) * self.calculate_gini(y_right))

    def calculate_leaf_value(self, y):         """计算叶子节点的预测值（多数类）"""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
```

#### 回归树

```python
class CARTRegressor(CARTBase):
    def calculate_variance(self, y):         """计算方差"""
        return np.var(y)

    def calculate_impurity_reduction(self, y, y_left, y_right):      """计算方差的减少量 """
        p = len(y_left) / len(y)
        return self.calculate_variance(y) - (p * self.calculate_variance(y_left) + (1 - p) * self.calculate_variance(y_right))

    def calculate_leaf_value(self, y):        """计算叶子节点的预测值（均值）"""
        return np.mean(y)
```

剪枝的代码就不在这里给出了

---

### 算法分析

CART树相较于ID3和C4.5算法来说，具有本质上的突破，可以同时处理分类和回归问题。而且ID3与C4.5在面对离散特征时往往会形成多叉树，数据切分过快，非常容易过拟合；而CART树永远是一颗二叉树，这种严格的二叉树结构保证了数据在树中的流动更加平缓，增强了模型的稳定性和泛化能力。同时，CART引入了代价复杂度剪枝（Cost-Complexity Pruning, CCP）。它在损失函数中显式地加入了对树节点数量的惩罚项，剪枝策略更加严谨。  
但是虽然二叉分裂缓解了数据碎片化，但如果一个分类特征真的具有很强的多态独立性，强制将其二分其实会破坏特征的内在逻辑，导致树的深度被迫增加；而且单棵 CART 树极其容易陷入局部最优，这就需要引入随机森林（后续blog）来解决这个问题。

### 补充
（来源自gemini）
为什么用基尼指数取代信息熵：
ID3 和 C4.5 计算不纯度依赖于信息熵，而 CART 分类树换成了基尼指数。这个改变的本质，是为了在保证优化方向正确的前提下，追求极致的计算速度。让我们用泰勒展开来揭示这两者在数学上的底层联系：对于一个包含 $K$ 个类别的系统，第 $k$ 类的概率为 $p_k$，信息熵的定义为：

$$Entropy = - \sum_{k=1}^{K} p_k \ln p_k$$

(注：这里为了方便求导，底数取自然对数 $e$。)  
信息熵中包含了大量的对数运算 $\ln$，在底层代码实现中，对数函数的泰勒级数展开计算极其消耗 CPU 资源。现在，我们把函数 $f(x) = \ln x$ 在 $x=1$ 处进行一阶泰勒展开：

$$\ln x \approx \ln(1) + (x - 1) \cdot \frac{d(\ln x)}{dx}\Big|_{x=1} = 0 + (x - 1) \cdot 1 = x - 1$$

由于概率 $p_k$ 的值在 $[0, 1]$ 之间，我们将 $\ln p_k \approx p_k - 1$ 代入信息熵的公式中：

$$Entropy \approx - \sum_{k=1}^{K} p_k (p_k - 1) = \sum_{k=1}^{K} (p_k - p_k^2) = \sum_{k=1}^{K} p_k - \sum_{k=1}^{K} p_k^2$$

因为所有类别的概率之和 $\sum_{k=1}^{K} p_k = 1$，所以：

$$Entropy \approx 1 - \sum_{k=1}^{K} p_k^2$$

而这正是基尼指数
