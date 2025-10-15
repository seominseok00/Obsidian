## 1.1 Mathematical Optimization

##### Optimization problem

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

**Linear function**

$$
f_i(\alpha x + \beta y) = \alpha f_i(x) + \beta f_i(y)
$$

for all $x, y \in \mathbb{R}^n$ and all $\alpha, \beta \in \mathbb{R}$.


**Convex function**

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


## 1.3 Convex Optimization

##### Convex optimization problem

$$
\begin{aligned}
&\text{minimize} \; f_0(x) \\
&\text{subject to} \; f_i(x) \leq b_i, \quad i = 1, \ldots, m
\end{aligned}
$$

where the functions $f_0, \ldots, f_m: \mathbb{R}^n \rightarrow \mathbb{R}$ are convex, i.e., satisfy


$$
f_i(\alpha x + \beta y) \leq \alpha f_i(x) + \beta f_i(y)
$$

for all $x, y \in \mathbb{R}^n$ and all $\alpha, \beta \in \mathbb{R}$ with $\alpha + \beta = 1, \alpha \geq 0, \beta \geq 0$.


### 1.3.1 Solving convex optimization problems

convex optimization problems은 일반적인 analytical solution은 존재하지 않지만, 해를 구하기 위한 효과적인 방법들이 존재한다. (Interior-point methods는 실제로 매우 잘 동작)


### 1.3.2 Using convex optimization

만약 어떤 문제를 convex optimization problem으로 formulate 할 수 있다면, (약간 과장해서) 해당 문제를 풀었다고도 할 수 있다.


## 1.4 Nonlinear optimization

Nonlinear optimization은 objective 또는 constraint function이 선형이지 않고, convex한지도 알려져 있지 않은 문제를 나타낸다.

안타깝게도, nonlinear programming problem을 효과적으로 풀 수 있는 방법이 존재하지 않는다.


### 1.4.1 Local optimization

Local optimization은 근처의 feasible points 내에서 objective function을 최소화하는 점을 찾는다.

따라서 모든 feasible points 중에서 가장 작은 목적함수 값을 갖는다는 보장은 없다.

Local optimization은 빠르고, 큰 스케일의 문제를 다룰 수 있고, 오직 objective, constraint functions이 미분 가능하기만 하면 된다는 장점이 있지만,

initial guess가 매우 중요하며, 종종 알고리즘의 파라미터 값에 민감하다는 단점이 있다.


> Local optimization methods for nonlinear programming vs Convex optimization
> 
> Local optimization methods
> - objective, constraint function이 미분 가능하기만 하면 사용 가능
> - 알고리즘 선택, 알고리즘 파라미터 조정, 좋은 initial guess를 찾기 어려움
> 
> Convex optimization
> - 문제 formulation이 어려움
> - solution을 찾는 것은 쉬움 (mature technology)


### 1.4.2 Global Optimization

Global optimization은 variables의 수가 적고, computing time이 중요하지 않으며 실제로 global solution을 찾는 가치가 큰 경우에 사용한다.

ex) worst-case analysis, verification safety-critical system

>cf) local optimization은 시스템의 신뢰성(reliable)을 보장할 수 없다.
>단순히 특정 지역 내에서만 나쁜 파라미터를 찾지 못한 것 뿐이니까. 반면에 global optimization은 파라미터의 absolute worst values를 찾기 때문에 시스템의 안정성을 보장할 수 있음.


### 1.4.3 Role of convex optimization in nonconvex problems

**Initialization for local optimization**

1. 원래 문제를 convex optimization problem으로 근사
2. 근사한 문제의 solution을 구함
3. 구한 solution을 원래 문제의 initial guess로 사용


**Convex heuristic for nonconvex optimization**

randomized algorithms은 nonconvex problem의 근사 해를 구하기 위해 몇몇 확률 분포로부터 candidates를 뽑아 그 중에서 가장 좋은 값을 approximate solution으로 선택하는 알고리즘을 의미하는데, 

이때 확률 분포를 정하는 것이 convex problem이라고 한다. (항상 그런건 아니고 때때로)


**Bounds for global optimization**

global optimization 기법들은 nonconvex problem의 최적값에 대한 lower bound를 필요로 하는데(계산이 쉬운 cheaply computable), 이러한 lower bound를 구하는 대표적인 두 가지 방법은 모두 convex optimization에 기반한다.

ex) relaxation: nonconvex constraint를 convex한 더 느슨한 제약 조건으로 대체하는 방식 (Lagrangian relaxation)