---
title: SafeDreamer
author: Weidong Huang, Jiaming Ji, Chunhe Xia, Borong Zhang, Yaodong Yang
published: ICLR 2024
created: 2025-10-13 15:30
status:
category: Constrained RL
tags:
  - Constrained-RL
  - Model-Based
  - ICLR
pdf: file:///Users/seominseok/Documents/Bookends/Attachments/SafeDreamer.pdf
url: https://sites.google.com/view/safedreamer
github: https://github.com/PKU-Alignment/SafeDreamer
understanding:
rating:
---
## The Purpose of This Study

CMDP에서는 cost function으로 위험한 행동을 정량화하며, 에이전트의 목표는 cost를 constraint threshold보다 작게 유지하면서 rewards를 최대화하는 것이다.

기존 Lagrangian 기반 방법에서는 cost를 0에 가깝게 설정할수록 제대로 충족하지 못하는 경우가 자주 발생한다.

하지만 internal dynamics model을 활용하면 높은 보상을 받으며 비용을 거의 0으로 유지하는 action trajectory를 planning 할 수 있다. ([Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method](https://arxiv.org/pdf/2010.07968))

그러나 복잡한 시나리오에서 ground-truth dynamics model을 얻는 것은 현실적으로 불가능하며, long-horizon planning의 높은 계산 비용으로 finite horizon 내에서만 최적화를 수행하게 되어 local optimal 또는 unsafe solution을 유발한다.

따라서 논문에서는 장기적인 보상과 비용을 균형 있게 조절하기 위한 safety-aware world model을 제안한다.


## Methods

### 4.1 Model Components

논문에서는 일반성을 잃지 않기 위해 하나의 제약 조건만 고려한다.

SafeDreamer는 world model, actor-critic model로 구성돼 있다.

매 time step마다 world model은 observation $o_t$, action $a_t$를 입력으로 받는다.

observation encoder로 discrete representation $z_t$로 압축
$$
z_t \sim E_{\phi}(z_t | h_t, o_t)
$$


sequence model은 $z_t$와 $a_t$를 입력으로 받아 다음 representation $\hat{z}_{t + 1}$을 예측
$$
h_t, \hat{z}_t = S_{\phi} (h_{t - 1}, z_{t - 1}, a_{t - 1})
$$

model state $s_t = \{ h_t, z_t \}$로 recurrent state $h_t$와 concatenate 해서 사용한다.

decoder는 $s_t$를 사용하여 observation, reward, cost를 예측한다.
$$
\begin{aligned}
\hat{o}_t &\sim O_{\phi}(\hat{o}_t | s_t) \\
\hat{r}_t &\sim R_{\phi}(\hat{r}_t | s_t) \\
\hat{c}_t &\sim C_{\phi}(\hat{c}_t | s_t)
\end{aligned}
$$

actor-critic에서는 $s_t$를 입력으로 받아 reward value $v_{r_t}$, cost value $v_{c_t}$, action $a_t$를 예측한다.
$$
\begin{aligned}
a_t &\sim \pi_{\theta}(a_t | s_t) \\
\hat{v}_{r_t} &\sim V_{\psi_r}(\hat{v}_{r_t} | s_t) \\
\hat{v}_{c_t} &\sim V_{\psi_c}(\hat{v}_{c_t} | s_t)
\end{aligned}
$$

### 4.2 Online Safety-Reward Planning (OSRP) via World Model

현재 상태 $s_t$로부터 state-action trajectories를 생성

각 trajectory는 critic 모델로 평가되며, optimal safe action trajectory가 실행된다. (Constrained Cross-Entropy method를 적용)

1. $(\mu^0, \sigma^0)_{t:t+ H}$를 초기화 $(\mu^0, \sigma^0 \in \mathbb{R}^{|\mathcal{A}|})$
2. 현재 action distribution $\mathcal{N}(\mu^{j - 1}, \sigma^{j - 1})$에서 $N_{\pi_{\mathcal{N}}}$개만큼 trajectory를 샘플링
3. reward-driven actor로도 $N_{\pi_{\theta}}$개만큼의 trajectory를 샘플링 -> accelerating the convergence of planning
4. world model 안에서 rollout해서 각 trajectory별 reward return $J^R_{\phi}$, cost return $J^C_{\phi}$을 추정
	- cost critic의 에러를 피하기 위해 alternative estimation $J^{C'}_{\phi} = (J^{C, H}_{\phi} / H) L$ 도 계산
	- $J^{C, H}_{\phi} = \sum^{t + H}_t \gamma^t C_{\phi}(s_t)$
	- H: planning horizon, L: episode length
5. $J^{C'}_{\phi} < b$이면 제약 조건을 만족했기 때문에 safe trajectory라고 표시($|A_s|$: 제약 조건을 만족하는 safe trajectory의 수, $N_s$: 우리가 원하는 safe trajectory의 개수)
	1. $|A_s| < N_s$: $-J^{C'}_{\phi}$로 정렬 (safe trajectory가 부족한 경우 -> 일단 더 안전한(비용이 적은) trajectory를 우선시)
		- sampled action trajectories $\{ a_{t: t+H} \}^{N_{\pi_{\mathcal{N}}} + N_{\pi_{\theta}}}$가 candidate action set $A_c$ 
	2. $|A_s| \geq N_s$: $J^R_{\phi}$로 정렬 (safe trajectory가 충분한 경우 -> 보상을 최대로)
		- safe action trajectory set $A_s$가 candidate action set $A_c$
		- OSRP: $J^R_{\phi}$
		- OSRP-Lag: $J^R_{\phi} - \lambda_p J^C_{\phi}$
6. candidate action set $A_c$로부터 sorting key $\Omega$가 높은 상위 $k$개를 elite action set $A_e$로 뽑음.
7. elite action set으로부터 $\mu^j = \frac{1}{k} \sum^k_{i = 1} A^i_e, \sigma^j = \sqrt{\frac{1}{k} \sum^k_{i = 1} (A^i_e - u^j)^2}$ 을 계산해 action distribution을 업데이트 (좋은 trajectory 근처로 분포를 옮김)
8. 2 ~ 7 과정을 J번 반복
9. final distribution $\mathcal{N}(\mu^j, (\sigma^J)^2 \mathrm{I})$에서 trajectory를 하나 샘플링 한 다음, 해당 trajectory의 첫 번째 행동만 실제 환경에서 실행


## Results & Discussion


## Critique