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
## The Purpose of This Study
Constrained Markov Decision Process (CMDP)

$$
\max_{\pi} J(\pi) = \mathbb{E}_{\tau \sim \pi} \left\{ \sum^{\infty}_{t = 0} \gamma^{t} r_{t} \right\} \; \text{s.t.} \; C(\pi) = \mathbb{E}_{\tau \sim \pi} \left\{ \sum^{\infty}_{t = 0} \gamma^{t}_{c} c_{t} \right\} \leq d
$$
$$
\Pi_{C} = \{ \pi | C(\pi) \leq d \}
$$

^e36422

기존 CMDP에서는 [[#^e36422]]과  같이 feasible policy set을 정의했다.

CMDP의 feasible policy set에서는 어떤 상태가 feasible 한지에 대해서는 다루지 않는다.


## Preliminaries

`Definition 1: Feasible state under the policy`

정책 $\pi$가 주어졌을 때, 상태 $s$에서 시작하여 정책 $\pi$를 따라 움직였을 때 받는 expected return of cost $c_{t}$가 다음 조건을 만족하면 상태 $s$가 feasible(safe)하다고 한다.

$$
v^{\pi}_{C}(s) = \mathbb{E}_{\tau \sim \pi} \left\{ \sum^{\infty}_{t = 0} \gamma^{t}_{c} c_{t} | s_0 = s \right\} \leq d
$$

$\mathbb{E}_{\tau \sim \pi} \{ \cdot | s_0 = s \}$는 초기 상태 $s$에서 시작해서 정책 $\pi$를 따라 움직였을 때 생성된 trajectory의 기댓값을 의미한다.


`Definition 2: Infeasible region`

어떤 정책을 선택하더라도 안전하지 않은 상태들을 infeasible state로 정의하며, infeasible region $\mathcal{S}_{I}$는 다음과 같이 정의된다.

$$
\mathcal{S}_{I} = \{ s | v^{\pi}_{C}(s) > d, \forall \pi \in \Pi \}
$$

`Definition 3: Feasible region`

$$
\mathcal{S}_{F} = \complement_{\mathcal{S}} \mathcal{S}_{I}
$$

feasible region과 infeasible region은 정책과는 무관하나, 정책이 충분히 좋지 않으면 feasible state $s \in \mathcal{S}_{F}$에서도 infeasible 할 수 있음.

따라서 정책 $\pi$하에서 feasible region은 다음과 같이 정의

$$
\mathcal{S}^{\pi}_{F} = \{ s | v^{\pi}_{C}(s) \leq d\}
$$

실제 문제에서는 initial state $\mathcal{I}$에 infeasible state가 있을 수 있으므로, 논문에서는 $\mathcal{I}_{F} = \mathcal{I} \cup \mathcal{S}_{F}$만 고려한다. (infeasible state에 대해 최적화하는 것은 성능을 저하시키기 때문)


`Definition 4: Statewise safety constraint`

statewise safety constraint는 다음과 같이 정의된다.

$$
v^{\pi}_{C}(s) \leq d, \forall s \in \mathcal{I}_{F}
$$

^1a3a78

따라서 statewise safety constraint 하에서 feasible policy set은 다음과 같다.

$$
\Pi_{F} = \{ \pi | v^{\pi}_{C}(s) \leq d, \forall s \in \mathcal{I}_{F} \}
$$
최적화 문제는 다음과 같이 정의된다.

$$
\max_{\pi} J(\pi) = \mathbb{E}_{\tau \sim \pi} \left\{ \sum^{\infty}_{t = 0} \gamma^{t} r_{t} \right\} \; \text{s.t.} \; v^{pi}_{C}(s) \leq d, \forall s \in \mathcal{I}_{F} \tag{SP}
$$

^d17e65


논문에서는 [[#^d17e65]]를 풀기 위해 Lagrangian-based approach를 사용한다.

제약 조건이 $\mathcal{I}_{F}$에 속하는 모든 상태 $s$에 대해 정의돼 있기 때문에, 상태별 Lagrange multiplier(statewise multiplier) $\lambda(s)$가 필요하다.

이를 통해 식 [[#^d17e65]]는 아래와 같이 Lagrange function으로 표현할 수 있다.

$$
\mathcal{L}_{\text{ori-stw}}(\pi, \lambda) = - \mathbb{E}_{s \sim d_{0}(s)} v^{\pi}(s) + \sum_{s \in \mathcal{I}_{F}} \lambda(s) \left(v^{\pi}_{C}(s) - d \right) \tag{O-SL}
$$

^830830

Lagrange multiplier는 해의 feasibility에 대한 물리적 의미를 갖는다. 이를 statewise complementary slackness condition을 통해 설명한다.

`Proposition 5: Statewise complementary slackness condition`

문제 [[#^d17e65]]에서 상태 $s$에 대한 optimal multiplier $\lambda^{*}(s)$와 optimal safety critic $v^{*}_{C}(s)$라고 하면, 다음 조건이 성립한다.

$$
\lambda^{*}(s) = 0, \; v^{\pi^{*}}_{C}(s) < d, \; \text{or} \; \lambda^{*}(s) > 0, \; v^{\pi^{*}}_{C}(s) = d
$$

이 명제는 문제 [[#^d17e65]]에 대한 Karush-Kuhn-Tucker (KKT) necessary condition으로부터 비롯된다.

만약 $v^{\pi^{*}}_{C}(s) = d$라면 제약 조건이 활성화(active)되어서 목적 함수가 더 이상 최적화 될 수 없도록 방해하고 있음을 의미하며, 더 최적화할 경우 infeasible region으로 들어가게 된다. (즉, safety constraint가 활성화 되지 않은 경우 feasible region 안에 있어야 한다)

이러한 방식으로 optimal multiplier는 상태 $s \in \mathcal{I}_{F}$의 제약 조건이 활성화 되었는지 아닌지를 나타내는데 사용될 수 있다.

그러나 해당 상태가 infeasible state $s \in \mathcal{S}_{I}$인지를 식별하는데는 도움이 되지 않는다. (infeasible region은 모든 정책에 대해 infeasible 하지 않은 상태를 모아놓은 집합이기 때문에)

다음의 따름정리(corollary)를 통해 multiplier를 활용하여 infeasible states를 구분하고, approximate optimal feasible solution을 찾을 수 있다.

`Corollary 6:` 만약 상태 $s$가 infeasible region에 속한다면, primal-dual ascent 과정에서 $\lambda(s) \rightarrow \infty$가 된다.

Proposition 5, Corollary 6에 의해 multiplier와 feasibility 사이의 근사적 관계(approximate relation)를 다음과 같이 정리할 수 있다.

| Multiplier scale $\lambda(s)$ |        Feasibility situation of $s$         |
| :---------------------------: | :-----------------------------------------: |
|             Zero              |      Inactive (inside feasible region)      |
|            Finite             | Active (on the boundary of feasible region) |
|           Infinite            |              Infeasible region              |

논문에서는 infinite 대신 heuristic threshold를 사용했다고 한다.

[[Proof#Proof of Corollary 6]]


## Methods

### Feasible Actor-Critic (FAC)

실제 구현에서는 식 [[#^830830]]의 $\sum_{s} \lambda(s)(v^{\pi}_C(s) - d)$ term 때문에 다룰 수 없는데(intractable)

1. feasible region $\mathcal{I}_{F}$에 접근할 수 없을 뿐더러,
2. $\mathcal{I}_{F}$에 속하는 모든 상태에 대한 합을 계산하는 것이 불가능하기 때문이다.

이 때문에 기존의 constrained RL에서는 이러한 형태의 제약 조건을 고려하지 않았다.

논문에서는 실제 구현을 위해 [[#^830830]]을 샘플 기반 방식으로 변형하며


$$
\mathcal{L}_{\text{stw}}(\pi, \lambda) = \mathbb{E}_{s \sim d_{0}(s)} \{ -v^{\pi}(s) + \lambda(s) (v^{\pi}_{C}(s) - d) \} \tag{SL}
$$

^a6d649

Theorem 7을 통해 [[#^830830]]과 [[#^a6d649]]이 동등하다는 것을 보인다.

**Theorem 7: Equivalence of [[#^830830]] and [[#^a6d649]]**

$\max_{\lambda} \inf_{\pi} \mathcal{L}_{\text{stw}} (\pi, \lambda)$에 대한 optimal policy $\pi^{*}$, optimal Lagrange multiplier $\lambda^{*}$가 존재한다면, 해당 정책은 문제 [[#^d17e65]]의 최적 정책이다.

[[Proof#Proof of Theorem 7]]

### Performance comparison with Expected Lagrangian Methods

제안한 방법(FAC)의 성능의 상한과 하한에 대한 이론적 분석을 제공한다.

**Theorem 8**

모든 가능한 initial states가 feasible 하거나 $\mathcal{I} \subseteq \mathcal{S}_{F}$ 할 경우, feasible policy $\pi_{f}$는 expectation-based constraints 하에서도 반드시 feasible하다.

**즉, 동일한 constraint threshold $d$에 대하여 $\Pi_{F} \subseteq \Pi_{C}$.**

[[Proof#Proof of Theorem 8]]


**Theorem 9**

expectation-based constraints의 Lagrange function은 다음과 같이 정의한다.

$$
\mathcal{L}_{\text{exp}}(\pi, \lambda) \doteq \lambda(C(\pi) - d) - J(\pi) = \lambda(\mathbb{E}_{\tau \sim \pi} \{ \sum^{\infty}_{t = 0} \gamma^{t}_{c} c_t \} - d) - \mathbb{E}_{\tau \sim \pi} \{ \sum^{\infty}_{t = 0} \gamma^{t} r_{t} \} \tag{EL}
$$

^fffd43

Theorem 9에서는 [[#^fffd43]]과 [[#^a6d649]]의 성능을 이론적으로 비교한다.


성능 비교를 위해 policy space, state space가 concave-convex라고 가정

$\mathcal{I}_{F}, \Pi$가 nonempty convex set이라고 가정하자.

- $v^{\pi}$는 $\mathcal{I}_{F}, \Pi$에서 concave하고
- $v^{\pi}_{C}$는 $\mathcal{I}_{F}, \Pi$에서 convex 하다고 하면,

optimal expected Lagrangian [[#^fffd43]] $\mathcal{L}^{*}_{\text{exp}}$는 optimal statewise Lagrangian [[#^a6d649]] $\mathcal{L}^{*}_{\text{stw}}$의 상한이다.

$$
\mathcal{L}^{*}_{\text{stw}} \leq \mathcal{L}^{*}_{\text{exp}}
$$

만약 $\Pi$에서 Slater's condition이 각 상태 $s$에 대해 성립한다면, $\mathcal{L}^{*}_{\diamond} = -J^{*}_{\diamond}, \; \diamond \in \{\text{stw, exp}\}$.

따라서 다음과 같이 하한을 얻을 수 있다.

$$
J^{*}_{\text{stw}} \geq J^{*}_{\text{exp}}
$$

*Slater's condition: 목적 함수가 concave, 제약 조건이 convex 할 때 제약 조건을 엄격하게 만족하는 해가 존재한다.*
*제약 조건을 엄격하게 만족할 경우, Lagrange multiplier가 0이 되므로, primal problem과 dual problem이 같아진다. (strong duality)*


### Practical Implementation

primal dual gradient ascent는 수렴 성능이 좋지 않아서(parctically), 정책과 multiplier 업데이트를 서로 다른 스케줄링으로 업데이트

또한 안정성을 향상시키기 위해, cost value function이 constraint threshold에 가까워지면 multiplier network 학습을 시작

## Results & Discussion


## Critique