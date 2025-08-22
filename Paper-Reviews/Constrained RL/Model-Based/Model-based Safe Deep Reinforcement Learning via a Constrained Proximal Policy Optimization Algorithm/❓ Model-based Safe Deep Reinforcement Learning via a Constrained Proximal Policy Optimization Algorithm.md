### 3. Background

#### 3.1 Constrained Markov Decision Process (CMDP)

By a policy $\pi = \{\pi_0, \pi_1, \ldots \}$, we mean a decision rule for selecting actions.

For any $k \geq 0$ and $s \in S, \pi_k(s) \in \mathbb{P}(s)$ is the probability distribution $\pi_k(s) \triangleq (\pi_k(s, a), a \in A(s))$ where $\pi_k(s, a)$ is the probability of picking action $a$ in state $s$ at instant $k$ under policy $\pi$.

A stationary policy is a randomized policy as above except with $\pi_k = \pi_l, \forall k \neq l$.

Thus, a stationary policy selects actions according to a given distribution regardless of the instant when an action is chosen according to the given policy.

**Stationary Policy**

논문의 $\pi_k$에서 밑첨자 $k$는 에피소드 안에서의 time step을 의미

즉, 에피소드 동안 같은 분포를 쓰는 정책을 stationary policy라고 한다.

학습 알고리즘이 파라미터를 갱신하면서 정책이 바뀌는 것은 $\theta^{(0)}, \theta^{(1)}, \ldots$ 다른 시간 축에서 일어남

#### 3.2 Lagrangian Relaxation based Proximal Policy Optimization

The Lagrangian of the constrained optimization problem (3) can be written as follows:
$$
L(\theta, \lambda) = J^R(\pi_\theta) - \lambda (J^C(\pi_\theta) - d)
\tag{4}
$$
where $\lambda \in \mathbb{R}^+$ is the Lagrange multiplier and is a positive real number.

In terms of the Lagrangian, the goal is to find a tuple $(\theta^*, \lambda^*)$ of the policy and Lagrange multiplier such that
$$
L(\theta^*, \lambda^*) = \max_\theta \min_\lambda L(\theta, \lambda)
\tag{5}
$$

Solving the max-min problem as above is equivalent to finding a global optimal saddle point $(\theta^*, \lambda^*)$ such that $\forall(\theta, \lambda)$, the following holds:
$$
L(\theta^*, \lambda) \geq L(\theta^*, \lambda^*) \leq L(\theta, \lambda^*)
\tag{6}
$$

We assume that $\theta$ refers to the parameter of a Deep Neural Network, hence finding such a globally optimal saddle point is computationally hard.

So our aim is to find a locally optimal saddle point which satisfies (6) in a local neighbourhood $H_{\epsilon_1, \epsilon_2}$ which is defined as follows:
$$
H_{\epsilon_1, \epsilon_2} \triangleq \{(\theta, \lambda) | \| \theta - \theta^* \| \leq \epsilon_1, \| \lambda - \lambda^* \| \leq \epsilon_2 \}
$$
for some $\epsilon_1, \epsilon_2 > 0$.

Assuming that $L(\theta, \lambda)$ is known for every $(\theta, \lambda)$ tuple, a gradient search procedure for finding a local $(\theta^*, \lambda^*)$ tuple would be the following:
$$
\begin{aligned}
\theta_{n + 1} 
&= \theta_n - \eta_1(n) \nabla_{\theta_n} (-L(\theta_n, \lambda_n)), \\
&= \theta_n + \eta_1(n) [\nabla_{\theta_n} J^R(\pi_\theta) - \lambda_n \nabla_{\theta_n} J^C(\pi_\theta)] \\
\lambda_{n + 1} 
&= [\lambda_n + \eta_2(n) \nabla_{\lambda_n}(-L(\theta_n, \lambda_n))]_+, \\
&= [\lambda_n - \eta_2(n) (J^C(\pi_\theta) - d)]_+
\end{aligned}
$$
Here $[x]_+$ denotes $\max(0, x)$.


**Regular step-size conditions**

$i = 1, 2, \sum_k \eta_i(n) = \infty, \sum_k \eta^2_i (n) < \infty$.

충분히 많이 학습해야 하며, 학습률이 발산하지 않고 점점 작아져서 수렴해야 함.

### 4. Challenges in Environment Model Learning

#### Handling aleatoric and epistemic uncertainties

Each neural network's output parameterizes a multivariate normal distribution with diagonal covariance matrix.

**네트워크의 출력은 하나의 평균, 분산(보통 가우시안 분포를 모데링하므로)을 출력하는거 아닌지?**

환경의 다음 상태는 보통 다차원 벡터이므로, multivariate normal distribution을 모델링하는 듯


#### Aggregation of Error

In order to tackle this problem, most of the model-based RL approches use shorter (or truncated) horizon during the policy optimization phase.

**Shorter horizon이랑 Truncated horizon이랑 무슨 차이?**

truncated하면 마찬가지로 짧아지는거 아닌지?

GPT의 답변:

- Shorter horizon: planning 할 때, 짧은 horizon에 대해서만 계산하고 이를 기준으로 정책을 최적화
- Truncated horizon: horizon $T$는 그대로 두고, rollout 중간에 이를 잘라서($T$ 이전 스텝에), 이후 스텝에 대해서는 value function으로 대체하는 것

In  [safe-LOOP](https://arxiv.org/abs/2008.10066), approximation of the value function is used to provide long-term reasoning instead of using a truncated horizon.

safe-LOOP에서는 truncated horizon 대신 value function을 사용한다고 하는데, truncated 방식의 설명이 저게 맞는지..?