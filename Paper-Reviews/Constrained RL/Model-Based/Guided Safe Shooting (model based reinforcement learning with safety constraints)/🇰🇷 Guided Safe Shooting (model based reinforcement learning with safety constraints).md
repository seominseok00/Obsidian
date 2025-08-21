---
title: Guided Safe Shooting (model based reinforcement learning with safety constraints)
author: Giuseppe Paolo, Jonas Gonzalez-Billandon, Albert Thomas, Balazs Kegl
published: arXiv 2022
created: 2025-08-19 10:03
status:
category:
tags:
  - Constrained-RL
  - Model-Based
  - arXiv
  - Huawei
---

## The Purpose of This Study


## Lit. Review

### 3. Backgorund

#### 3.2 MBRL with decision-time planning

Algorithm 2: MAP-Elites

1. $\Gamma \leftarrow \text{SAMPLE}(\Phi, n)$: $n$개의 정책을 샘플링 (line 4)
2. 샘플링한 정책을 가지고 rollout한 다음(이때 실제 모델 $p_{\text{real}}$을 사용), 생성된 trajectories를 $\mathcal{A}_{\text{ME}}$에 저장. (line 5-8)
   STORE function (Algorithm 3: STORE function of MAP-Elites)
	1. trajectory를 discretized behavior descriptor $\bar{b}$로 변환
	2. $\mathcal{A}_{\text{ME}} = \mathcal{A}_{\text{ME}} \cup (\phi, \bar{b}, r)$: collection of policies $\mathcal{A}_{\text{ME}}$에 저장
3. 아래 과정을 $M$번 반복 ($M$: total evaluated policies)
4. $\Gamma \leftarrow \text{SELECT}(\mathcal{A}_{\text{ME}}, K)$ (line 10): 저장된 정책 중 $K$개를 샘플링
5. parameter에 noise를 추가한 다음 rollout 해서 trajectories를 생성
6. STORE function


## Methods

Algorithm 2, 3은 논문에서 제안한 방법인 Algorithm 4, 5의 vanilla version

Algorithm 4: Safety-aware QD planner

1. $\mathcal{T} \leftarrow \text{SAMPLE}(\phi, n)$: n개의 정책을 샘플링 (line 4)
2. 샘플링한 정책을 가지고 rollout한 다음(이때 학습된 모델 $\hat{p}$을 사용), 생성된 trajectories를 $\mathcal{A}_{\text{ME}}$에 저장 (line 5-8)
   STORE function (Algorithm  5: STORE function of safety-aware planner)
	1. trajectory를 discretized behavior descriptor $\bar{b}$로 변환
	2. $\mathcal{A}_{\text{ME}} = \mathcal{A}_{\text{ME}} \cup (\phi, \bar{b}, r)$: collection of policies $\mathcal{A}_{\text{ME}}$에 저장
	3. 이때 vanilla ME STORE function(Algorithm 3)과는 다르게, 저장시에 cost까지 고려함. (그래서 다음 action을 생성할 때 cost가 낮은 행동을 선택할 수 있음)
3. 아래 과정을 M번 반복 (M: evaluated action sequences) - 여기가 evolutionary generations (line 9-15)
	1. SELECT funtion (Algorithm 6: SELECT function of safety-aware planner): vanilla ME(알고리즘 2)와는 다름
		1. $\Gamma \leftarrow \text{SELECT}(\mathcal{A}_{\text{ME}}, n_{\text{new}},\phi)$: $n_{\text{new}}$개의 정책을 선택
		2. 개수가 부족하면 무작위 정책을 추가로 샘플링 (exploration을 강화)
	2. 선택된 정책($\Gamma$)으로 다시 rollout해서 trajectories를 생성
	3. STORE function (Algorithm 5)
4. $A_{\text{ME}}$에 저장된 행동 중 가장 좋은 행동 $a_t$를 선택하여 실행


## Results & Discussion


## Critique