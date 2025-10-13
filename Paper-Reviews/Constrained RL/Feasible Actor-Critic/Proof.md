---
title: Feasible Actor-Critic
author: Haitong Ma, Yang Guan, Shengbo Eben Li, Xiangteng Zhang, Sifa Zheng, Jianyu Chen
published: Archive
created: 2025-09-30 16:33
status:
category: Constrained RL
tags:
  - Constrained-RL
  - Statewise-Constrained-RL
pdf: file:///Users/seominseok/Documents/Bookends/Attachments/Feasible%20Actor-Critic.pdf
understanding: "4"
rating: ★★★
---
### Proof of Corollary 6

상태 $s$에서 feasible solution이 존재하지 않는다는 것은, 다음 부등식이 항상 성립한다는 것을 의미한다.

$$
v^{\pi}_{C}(s) > d
$$

multiplier network의 gradient에 따르면 $\hat{\nabla} J_{\lambda}$는 항상 0보다 크기 때문에, multiplier $\lambda(s)$는 무한대로 발산한다.


### Proof of Theorem 7

infeasible region에서는 안전한 정책을 찾을 수 없으므로, feasible region $\mathcal{S}_{F}$만 고려한다.

$$
\mathcal{L}_{\text{stw}}(\pi, \lambda) = \mathbb{E}_{s \sim d_{0}(s)} \{ -v^{\pi}(s) + \lambda(s) (v^{\pi}_{C}(s) - d) \} \tag{SL}
$$

^3a1708

[[#^3a1708]]은 다음과 같이 풀어서 쓸 수 있다.

$$
\begin{equation}
	\begin{aligned}
		J_{\mathcal{L}}(\theta, \lambda) 
		&= \mathbb{E}_{s \sim d^{0}(s)} \left\{ v^{\pi(s)} + \lambda(s) \left[ v^{\pi}_{C}(s) - d \right] \right\} \\
		&= \mathbb{E}_{s \sim d^{0}(s)} \{ v^{\pi}(s) \} + \sum_{s \in \mathcal{S}_{F}} d^{0}(s) \lambda(s) \left[ v^{\pi}_{C}(s) - d \right] \\
		&= \mathbb{E}_{s \sim d^{0}(s)} \{ v^{\pi}(s) \} + \sum_{s \in \mathcal{S}_{F}} \lambda(s) \left\{ d^{0}(s) [v^{\pi}_{C}(s) - d] \right\}
	\end{aligned}
\end{equation}
$$

마지막 식은 다음과 같은 최적화 문제에 대응된다.

$$
\begin{equation}
	\begin{aligned}
		&\min_{\theta} \mathbb{E}_{s \sim d^{0}(s)} v^{\pi}(s) \\
		&\text{s.t.} \; d^{0}(s) [v^{\pi}_{C}(s) - d] \leq 0, \; \forall s \in \mathcal{S}_{F}
	\end{aligned}
	\tag{A-SP}
\end{equation} 
$$

^68a82f

식 [[#^68a82f]]에서 제약 조건 $d^{0}(s) [v^{\pi}_{C}(s) - d]$은 $v^{\pi}_{C}(s) - d \leq 0, \forall s \in \mathcal{I}_{F}$와 동일하다.

$d^{0}(s)$는 initial state distribution이기 때문에(*논문에서는 visiting probability라고 언급*) $d^{0}(s) \geq 0$.

- 만약, 상태 $s$가 initial state가 아닐 경우($s \notin \mathcal{I}$) $d^{0}(s) = 0$이므로 고려 대상이 아니다.
- $d^{0}(s) > 0$인 경우, 부등식 방향은 그대로 유지하고, 스케일만 달라지므로 $v^{\pi}_{C}(s) - d \leq 0, \forall s \in \mathcal{I}_{F}$가 **실질적인 제약 조건이 된다.**

이는 우리가 원래 풀고자 했던 최적화 문제 [[#^b49810]]과 같다.

$$
\mathcal{L}_{\text{ori-stw}}(\pi, \lambda) = - \mathbb{E}_{s \sim d_{0}(s)} v^{\pi}(s) + \sum_{s \in \mathcal{I}_{F}} \lambda(s) \left(v^{\pi}_{C}(s) - d \right) \tag{O-SL}
$$

^b49810


따라서 [[#^68a82f]]는 [[#^3a1708]]과 같다.

근데, [[#^68a82f]]는 [[#^b49810]]과 같으므로, [[#^b49810]]과 [[#^3a1708]]은 같다.


### Proof of Theorem 8

expectation-based constraints는 다음과 같이 재구성 할 수 있다.

$$
\begin{equation}
	\begin{aligned}
		C(\pi) 
		&= \mathbb{E}_{\tau \sim \pi} \{ \sum^{\infty}_{t = 0} \gamma^{t}_{c} c_{t} \} \\
		&= \sum_{s} d_{0}(s) \mathbb{E}_{\tau \sim \pi} \{ \sum^{\infty}_{t = 0} \gamma^{t}_{c} c_{t} | s_{0} = s \} \quad \text{상태별 비용의 가중 평균으로 나타냄} \\
		&= \mathbb{E}_{s \sim d_{0}(s)} \{ v^{\pi}_{C}(s) \}
	\end{aligned}
\end{equation}
$$

Definition 4 [[Paper-Reviews/Constrained RL/Feasible Actor-Critic/Review#^1a3a78]] 처럼, 정책이 feasible($\pi_{f} \in \Pi_{F}$) 하다면, $\forall s \in \mathcal{I} \subseteq \mathcal{S}^{\pi_{f}}_{F}, v^{{\pi}_{f}}_{C}(s) \leq d$

따라서 상태별 비용의 가중 평균으로 나타낸 다음 식도 성립한다.
$$
C(\pi_{f}) = \mathbb{E}_{s \sim d_{0}(s)} \{ v^{{\pi}_{f}}_{C}(s) \} = \sum_{s} d_{0}(s) v^{{\pi}_{f}}_{C}(s) \leq d
$$
*동등(equivalent)이 아니라 포함(inclusion) 관계*  $\Pi_{F} \subseteq \Pi_{C}$

- state-wise constraint를 만족하면 expectation-based constraint도 만족
- 하지만 expectation-based constraint를 만족한다고, state-wise constraint를 만족하는 것은 아님


### Proof of Theorem 9

$$
\max_{\pi} J(\pi) = \mathbb{E}_{\tau \sim \pi} \left\{ \sum^{\infty}_{t = 0} \gamma^{t} r_{t} \right\} \; \text{s.t.} \; v^{pi}_{C}(s) \leq d, \forall s \in \mathcal{I}_{F} \tag{SP}
$$

^a26ca7

$$
\mathcal{L}_{\text{ori-stw}}(\pi, \lambda) = - \mathbb{E}_{s \sim d_{0}(s)} v^{\pi}(s) + \sum_{s \in \mathcal{I}_{F}} \lambda(s) \left(v^{\pi}_{C}(s) - d \right) \tag{O-SL}
$$

^89d2ab


문제 [[#^a26ca7]]를 모든 상태에 대한 합으로 모아놓은 형태의 Lagrange function으로 정의한게 문제 [[#^89d2ab]]

상태 $s$를 random variable로 간주하고, 이를 다음과 같은 Lagrange function으로 구성할 수 있다.

$$
L(\pi, \lambda, s) = -v^{\pi}(s) + \lambda(s)[v^{\pi}_{C}(s) - d] \tag{A.1}
$$

^ef9b66

문제 [[#^ef9b66]]의  dual problem은 다음과 같다.

$$
\max_{\lambda} \inf_{\pi} L(\pi, \lambda, s) = \max_{\lambda} \inf_{\pi} \left\{ -v^{\pi}(s) + \lambda[v^{\pi}_{C}(s) - d] \right\} \tag{A.2}
$$

$$
G(\lambda, s) = \inf_{\pi} \left\{ -v^{\pi}(s) + \lambda v^{\pi}_{C}(s) - \lambda d \right\} \tag{A.3, A.5}
$$

^9895e4

expected Lagrangian [[Paper-Reviews/Constrained RL/Feasible Actor-Critic/Review#^fffd43]] 은 [[#^9895e4]] 의 expected solution을 최적화한다.


**Lemma 10: Convex condition for infimum operation**

C가 convex nonempty set이고 함수 f가 (x, y)에 대해 convex하면, y에 대한 infimum

$$
g(x) = \inf_{y \in C} f(x, y) \tag{A.4}
$$

도 convex하다.

**Proposition 11**

따라서 dual problem [[#^9895e4]]은 $S$에서 convex하다.

> **Proof of Proposition 11**

앞서 

- $v^{\pi}$는 $\mathcal{I}_{F}, \Pi$에서 concave
- $v^{\pi}_{C}$는 $\mathcal{I}_{F}, \Pi$에서 convex

하다고 가정했는데, 따라서 식 [[#^9895e4]]의 $-v^{\pi}(s) + \lambda v^{\pi}_{C}(s) - \lambda d$ 이 부분이 convexity의 선형성(linear of convexity)에 의해 $G(\lambda, s)$도 convex하다고 봄.

*$v^{\pi}$는 concave 한데 -를 붙여서 convex function으로 만들었기 때문에*


**Lemma 12: Lower bound on deterministic equivalent (Jensen Inequality)**

stochastic programming problem에서 함수 $f$에 대해 최적화 변수 $x$가 random variable $\omega$에 대해 convex할 경우 다음이 성립한다.

$$
f(x, \mathbb{E} \omega) \leq \mathbb{E} f(x, \omega) \tag{A.6}
$$

따라서 다음 식이 성립한다.

$$
\begin{equation}
	\begin{aligned}
			G^{*}_{\text{exp}} 
			&= \max_{\lambda} \left\{ \inf_{\pi} \left\{ -\mathbb{E}_{s} v^{\pi}(s) + \lambda \mathbb{E}_{s} v^{\pi}_{C} (s) \right \} \right\} \\
			&\geq max_{\lambda} \left\{ \inf_{\pi} \left\{ -v^{\pi}(\mathbb{E}_{s} s) + \lambda v^{\pi}_{C} (\mathbb{E}_{s} s) \right\} \right\} = \max_{\lambda} G(\lambda, \mathbb{E}_{s} s)
	\end{aligned}
	\tag{A.8}
\end{equation}
$$

**Lemma 13: Infinite fitting power of policy and multipliers**

정책 $\pi(\cdot)$가 infinite fitting power를 갖는다면, 다음 식이 성립한다.

$$
\inf_{\pi} \mathbb{E}_{s} \{ \cdot \} = \mathbb{E}_{s} \{ \inf_{\pi} (\cdot) \}
$$

이에 따라 statewise Lagrangian [[#^3a1708]]은 다음과 같이 표현된다.

$$
\begin{equation}
	\begin{aligned}
		G^{*}_{\text{stw}} 
		&= \max_{\lambda} \inf_{\pi} \mathbb{E}_{s} \left\{ -v^{\pi}(s) + \lambda(s)[v^{\pi}_{C}(s) - d] \right\} \\
		&= \max_{\lambda} \mathbb{E}_{s} \left\{ \inf_{\pi} \left\{ -v^{\pi}(s) + \lambda(s) v^{\pi}_{C}(s) - \lambda(s)d \right\} \right\} \quad \leftarrow \text{Lemma 13} \\
		&= \max_{\lambda} \mathbb{E}_{s} \left\{ G(\lambda(s), s) \right\} \\
		&\leq \mathbb{E}_{s} \max_{\lambda}\{G(\lambda(s), s)\}
	\end{aligned}
	\tag{A.9}
\end{equation}
$$

**Proposition 14**

dual problem [[#^9895e4]]의 optimal solution $\max_{\lambda} G(\lambda, s)$는 $\mathcal{S}$에 대해 concave하다.

>**Proof of Proposition 14**

$$
\begin{equation}
	\begin{aligned}
		G^{*}_{\text{exp}} 
		&\geq \max_{\lambda} G(\lambda, \mathbb{E}_{s} s) \quad \leftarrow \text{Lemma 12} \\
		&\geq \mathbb{E}_{s} \max_{\lambda}[G(\lambda, s)] \quad \leftarrow \text{부등호 반대가 맞는거 아닌가?} \\
		&\geq \mathbb{E}_{s} \max_{\lambda(s)} [G(\lambda(s), s)] \\
		&\geq G^{*}_{\text{stw}}
	\end{aligned}
\end{equation}
$$