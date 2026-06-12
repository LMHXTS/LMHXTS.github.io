# 电动力学第二章：静电场 复习笔记

---

## 一、静电场的基本方程与边界条件

静电场的特点是电荷静止且场量不随时间变化。因为静电场是无旋场（$\nabla\times\vec{E}=0$），可引入标势 $\varphi$ 使得 $\vec{E}=-\nabla\varphi$。

- **泊松方程 (Poisson's Equation)**：在有电荷分布的区域，电势满足 $\nabla^{2}\varphi=-\dfrac{\rho}{\epsilon}$。
- **拉普拉斯方程 (Laplace's Equation)**：在无自由电荷分布的区域（$\rho=0$），方程退化为 $\nabla^{2}\varphi=0$。
- **边值关系**：在两种不同介质的分界面上，电势必须满足连续性 $\varphi_1 = \varphi_2$。法向导数满足 $\epsilon_2\dfrac{\partial\varphi_2}{\partial n}\Big|_S - \epsilon_1\dfrac{\partial\varphi_1}{\partial n}\Big|_S = -\sigma$。
- **静电场能量**：总能量公式为 $W = \dfrac{1}{2}\displaystyle\int_{\infty}\rho\varphi\,d\tau$。考试常考概念：**不能**把 $\rho\varphi$ 看成是电场能量密度，真实的静电能是以密度为 $\vec{E}\cdot\vec{D}$ 的形式连续分布在空间中。

---

## 二、核心理论：唯一性定理 (Uniqueness Theorem)

唯一性定理是后续所有数学求解方法（尤其是"猜"解法如镜像法）的合法性基础。

- **定理内容**：设区域内给定自由电荷分布 $\rho(x)$，只要在边界 $S$ 上给定**电势 $\varphi|_S$**（第一类边值条件）或**电势的法向导数 $\dfrac{\partial\varphi}{\partial n}\Big|_S$**（第二类边值条件），则区域内的电场被唯一确定。
- **物理意义**：只要我们找到一个解，且这个解既满足泊松/拉普拉斯方程，又符合给定的边界条件，那它就是唯一正确的物理真实解。

---

## 三、三大核心求解方法（重点应用考点）

### 1. 分离变量法 (Method of Separation of Variables)

- **适用条件**：求解区域内无电荷（满足拉普拉斯方程），且边界必须是简单的规则几何面。
- **球坐标系通解（轴对称情况）**：当电势与方位角 $\phi$ 无关时，通解可展开为勒让德多项式：

$$\varphi(r,\theta)=\sum_{n=0}^{\infty}\left(A_{n}r^{n}+\frac{B_{n}}{r^{n+1}}\right)P_{n}(\cos\theta)$$

求解步骤在于根据有限远、无穷远以及交界面条件确定常数 $A_n$ 和 $B_n$。

### 2. 镜像法 (Method of Images)

基于唯一性定理，用假想的"像电荷"代替真实的导体感应电荷或介质极化电荷对场点的作用。

- **两大限制铁律**：像电荷**必须放在所研究的场域外**；边界必须是简单的规则面。
- **三大经典模型**：
    1. **无限大接地导体平面**：距板 $a$ 处的点电荷 $Q$，像电荷为 $-Q$，位于关于平面对称的距离 $a$ 处。
    2. **接地导体球面（半径 $R_0$）**：距球心 $a$ 处的点电荷 $Q$，像电荷大小为 $Q' = -\dfrac{R_0}{a}Q$，放置在球内距离球心 $b = \dfrac{R_0^2}{a}$ 处。
    3. **相交接地平面（劈形，夹角 $\alpha$）**：只有当 $2\pi/\alpha$ 为**偶数**时，才能用镜像法求解。包括原电荷在内，总共存在 $2\pi/\alpha$ 个点电荷。

### 3. 格林函数法 (Method of Green Function)

- **格林函数物理意义**：$G(\vec{x},\vec{x}')$ 表示位于 $\vec{x}'$ 处的**单位点电荷**在特定边界下于观察点 $\vec{x}$ 处激发的电势。
- **数学定义**：满足 $\nabla^{2}G(\vec{x},\vec{x}^{\prime})=-\dfrac{1}{\epsilon_{0}}\delta(\vec{x}-\vec{x}^{\prime})$。
- **无界空间格林函数**：$G(\vec{x},\vec{x}^{\prime})=\dfrac{1}{4\pi\epsilon_{0}}\dfrac{1}{|\vec{x}-\vec{x}^{\prime}|}$。
- **解的形式**：通过格林第二公式，可将任意电势分布表示为体电荷的体积分和边界条件的面积分之和。
