## 1.1 Mathematical Optimization

**Optimization problem**

$$
\begin{aligned}
&\text{minimize} \; f_0(x) \\
&\text{subject to} \; f_i(x) \leq b_i, \quad i = 1, \ldots, m
\end{aligned}
$$

- optimization variable: vector $x = (x_1, \ldots, x_n)$
- objective function: $f_0: \mathbb{R}^n \rightarrow \mathbb{R}$
- (inequality) constraint function: $f_i: \mathbb{R}^n \rightarrow \mathbb{R}, \; i = 1, \ldots, m$
- constraint limits: $b_1, \ldots, b_m$

제약 조건을 만족하는 모든 벡터들 중에서 objective value가 가장 작은 vector $x^*$를 optimal(solution)이라고 한다.

for any $z$ with $f_1(z) \leq b_1, \ldots, f_m(z) \leq b_m$에 대해 $f_0(z) \geq f_0(x^*)$

**Linear program**

$$
f_i(\alpha x + \beta y) = \alpha f_i(x) + \beta f_i(y)
$$

for all $x, y \in \mathbb{R}^n$ and all $\alpha, \beta \in \mathbb{R}$.


**Convex optimization problem**

$$
f_i(\alpha x + \beta y) \leq \alpha f_i(x) + \beta f_i(y)
$$

for all $x, y \in \mathbb{R}^n$ and all $\alpha, \beta \in \mathbb{R}$ with $\alpha + \beta = 1, \alpha \geq 0, \beta \geq 0$.


> cf) linear programming의 경우 모든 alpha, beta에 대해서 성립해야 하기 때문에 더욱 엄격한 조건이다.
> 
> 나는 convex optimization에서 오히려 $\alpha + \beta = 1, \alpha \geq 0, \beta \geq 0$ 조건이 생겼기 때문에 더 엄격해졌다고 생각했는데, 그게 아니라 오히려 저 조건 내에서만 성립하면 되기 때문에 오히려 요구 조건이 더 약해진 것이라고 한다.
> 
> 게다가 등식에서 부등식으로 완화돼서 convex optimization이 linear programming을 포함한다고 한다.

### 1.1.2 Solving optimization problems

몇몇 특정 유형의 문제만(problem classes) 수백 ~ 수천개에 이르는 변수와 제약 조건을 포함하더라도 신뢰성 있게(reliably) 해결할 수 있는 효율적인 알고리즘이 존재한다.

책에서는 두 가지 예시로 `least squares problem`, `linear program`을 소개한다.


## 1.2 Least-squares and linear programming

### 1.2.1 Least-squares problems

least-squares problem은 제약 조건이 없고, 다음과 같이 $a^T_i x - b_i$ term들의 제곱을 모두 더한 형태로 objective function이 정의돼 있다.

$$
\text{minimize} \; f_0(x) = \Vert Ax - b \Vert^2_2 = \sum^k_{i = 1}(a^T_i x - b_i)^2
$$

$a^T_i$는 $A \in \mathbb{R}^{k \times n}$  (with $k \geq n$)의 row이고, $x \in \mathbb{R}^n$는 optimization variable이다.

> cf) 2-norm
> $$
> \Vert x \Vert_2 = \sqrt{x^2_1 + x^2_2 + \ldots x^2_n} = x^T x
> $$

**Solving least-squares problems**

$$
f_0(x) = \Vert Ax - b \Vert^2_2 = (Ax - b)^T(Ax - b) = x^T A^T Ax - 2b^T Ax + b^T b
$$

최소값을 찾기 위해서는 미분해서 0이 되는 벡터를 찾으면 된다.

$$
\nabla_x f_0(x) = 2A^T Ax - 2A^Tb = 0
$$

따라서 위와 같이 analytical solution을 구할 수 있다.


### 1.2.2 Linear programming

linear programming은 objective function과 constraint functions이 모두 선형(linear)인 최적화 문제를 의미한다.

$$
\begin{aligned}
&\text{minimize} \; c^Tx \\
&\text{subject to} \; a^T_i x \leq b_i, \quad i = 1, \ldots, m
\end{aligned}
$$

$c, a_1, \ldots, a_m \in \mathbb{R}^n$는 vector이고 $b_1, \ldots, b_m \in \mathbb{R}$은 scalar이다.


**Solving linear programs**

linear programming은 analytic solution을 구할 수 없지만, interior-point method를 사용하여 주어진 정확도로 linear program 문제를 해결하는데 필요한 연산 횟수에 대한 엄밀한 상한(rigorous bounds)을 설정할 수 있다.

하지만, least-squares와 linear programs을 해결하는 기술은 이미 충분히 검증되고 안정적이며 실용화된 분야(mature techonology)이다.