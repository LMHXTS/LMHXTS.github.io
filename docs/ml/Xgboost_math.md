# XGBoost算法数学推导

### 简要介绍

XGBoost全称（Extreme Gradient Boosting）极端梯度提升，他是梯度提升决策树的一种改进，通过继承多个弱学习器（通常为决策树）来构建强大的预测模型。  

#### boosting集成学习

集成学习就是将多个基学习器组合起来，以获得更好预测性能的机器学习方法。而XGBoost的集成学习发展自boosting方法，通过分步迭代的方式构建模型，即： 

1. 每一步都训练一个新的弱学习器
2. 新学习器专注于纠正已有模型的错误
3. 最终将所有弱学习器组合形成强大的预测模型  

#### 梯度提升决策树（GBDT）

梯度提升决策树（GBDT）是一种特殊的提升算法，它使用决策树作为基学习器，并通过梯度下降方法优化损失函数。基本流程如下：

1. 初始化模型，通常是一个常数值
2. 计算当前模型的残差（实际值与预测值的差）
3. 训练一个新的决策树来拟合这些残差
4. 将新树添加到现有模型中
5. 重复步骤2-4直到满足停止条件

XGBoost在GBDT的基础上优化了目标函数，将其分解为损失函数和正则化项（控制模型复杂度）两部分，且采用贪心算法来确定最佳的树结构。  

下面进行数学推导；

---

### 数学推导

#### 目标函数定义

假设现在有n个样本，一共需要训练K棵树。在第t步时，我们需要训练一棵新树$f_t$， 此时的目标函数可以写为：

$$Obj^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \Omega(f_t) + constant$$

其中：  
$l$ 是可导的凸损失函数（如均方误差、对数损失）。  
$\hat{y}_i^{(t)}$ 是前 $t$ 棵树对样本 $i$ 的累积预测值。  
$f_t(x_i)$ 是我们当前要学习的第 $t$ 棵树。  
$\Omega(f_t)$ 是这棵树的正则化项（复杂度）  

可以知道 $\hat{y}_i^{(t)} = \sum_{j=1}^t(f_j(x_i)) = \hat{y}_i^{(t-1)} + f_t(x_i)$ ，
于是上式可进一步写为：

$$Obj^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + constant$$

在第t棵树训练时，$\hat{y}_i^{(t-1)}$ 为已知常数。

#### 二阶泰勒展开近似

为寻找最优$f_t$，我们将损失函数在 $\hat{y}_i^{(t-1)}$ 处进行二阶泰勒展开, （ 这里$f_t(x_i)$相当于$\delta x$ ）：

$$Obj^{(t)} \simeq \sum_{i=1}^n  \left[ l(y_i, \hat{y}_i^{(t-1)}) +  \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}  f_t(x_i) + \frac{1}{2}  \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}  f_t^2(x_i) \right] + \Omega(f_t)$$

将常数系数化简表示后：

$$Obj^{(t)} \simeq \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$

其中：  
$g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$ $~~~~~$ $h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$ 

#### 定义树的结构与正则化项

一棵决策树本质上是将样本映射到某个叶子节点并赋予一个权重。我们设树 $f_t$ 共有 $T$ 个叶子节点，每个叶子的权重为 $w_j$。正则化项定义为：

$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

权重项系数定义为$\frac{1}{2} \lambda$有利于后续化简。

#### 转换遍历方式

这步是本人认为最精髓的一步，它将求目标函数时的遍历方式统一为遍历叶子节点。  
定义 $I_j$ 为被划分到第 $j$ 个叶子节点的样本集合,即$I_j = \{ x_i | f_t(x_i) \in w_j \}$ 于是目标函数可化为：

$$Obj^{(t)} \simeq \sum_{j=1}^T \left[ (\sum_{i \in I_j} g_i) w_j + \frac{1}{2} (\sum_{i \in I_j} h_i + \lambda) w_j^2 \right] + \gamma T$$

令 $G_j = \sum_{i \in I_j} g_i$ ，$H_j = \sum_{i \in I_j} h_i$ 。目标函数变成了一个关于 $w_j$ 的极简一元二次函数：

$$Obj^{(t)} \simeq \sum_{j=1}^T \left[ G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2 \right] + \gamma T$$

#### 求解最优结构

对 $w_j$ 求导并令其为 0，可以直接得到第 $j$ 个叶子节点的最优权重：

$$w_j^* = - \frac{G_j}{H_j + \lambda}$$

将最优权重代回目标函数，得到这棵树的极小值（结构分数）：

$$Obj^* = - \frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T$$

#### 节点分裂的收益

有了结构分数，我们在建树分裂节点时，就可以通过计算分裂前后的分数差直接判断一个特征点分割的优劣：

$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma$$

---

### 简要调参分析

XGBoost模型中参数主要如下：  
• eta（学习率）：每次迭代后对叶子节点权重的缩减系数，控制学习速度，范围[0,1]  
• $\gamma$（最小分裂损失）：节点分裂所需的最小损失减少量，越大越保守  
• max_depth（最大树深）：单棵树的最大深度，越大模型越复杂  
• min_child_weight（最小子节点权重）：子节点所需的最小样本权重和，用于控制过拟合  
• subsample（样本采样比例）：构建每棵树时使用的训练样本比例，范围(0,1]  
• colsample_bytree（特征采样比例）：构建每棵树时使用的特征比例，范围(0,1]  
• $\lambda$（正则化系数）：叶子权重的正则化项系数  
• n_estimators（树的数量）：总共训练的树的数量

调参策略：

1. 控制过拟合的参数：
• 降低学习率（eta）并增加树的数量（n_estimators）  
• 增加min_child_weight和$\gamma$  
• 使用subsample和colsample_bytree进行采样  
• 增加正则化参数$\lambda$和alpha  

2. 提高性能的参数：  
• 增加max_depth以捕获更复杂的模式  
• 减少min_child_weight以允许更细粒度的分裂  
• 调整学习率和树的数量以平衡精度和训练时间  

3. 调参顺序：   
• 首先确定学习率和树的数量  
• 然后调整树的结构参数（max_depth, min_child_weight）  
• 接着调整采样参数（subsample, colsample_bytree）  
• 最后调整正则化参数（$\lambda$, alpha）  
