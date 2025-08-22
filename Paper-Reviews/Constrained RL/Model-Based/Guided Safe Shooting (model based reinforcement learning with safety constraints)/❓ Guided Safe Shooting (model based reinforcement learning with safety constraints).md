### 2. Related Work

**Ensemble Networks**

In our work, we use the MBRL setup and choose to use auto-regressive mixture density net as model which have shown to alleviates error accumulation down the horizon which is an important feature for planning.

**Auto-regressive mixture density net이 뭔지?**

일반적인 NN은 하나의 평균값(point estimate)만 출력

실제 환경은 확률적일 수 있기 때문에 Mixture Density Network는 신경망의 출력이 확률분포의 파라미터(평균, 분산 등)가 되도록 학습하는 네트워크

Auto-regressive는 다차원 벡터를 한 번에 모델링하지 않고, 나눠서 차례대로 예측하는 모델
즉, 첫 번째 상태 차원 $s^0_{t + 1}$을 예측하고, 이를 조건으로 해서 두 번째 차원 $s^1_{t + 1}$을 예측하는 식으로 순차적으로 분포를 학습
$$
p(s_{t + 1}|s_t, a_t) = \Pi^{d_s - 1}_{l = 0} p(s^l_{t + 1} | s^0_{t + 1}, \ldots, s^{l - 1}_{t + 1}, s_t, a_t)
$$
이렇게 할 경우 변수들 간의 의존성(correlation)까지 포착할 수 있음.

### 3. Background

#### 3.1 Reinforcement Learning and safe-RL

To incorporate constraints, we define a cost function $C: S \times A \rightarrow \mathbb{R}$ which, in our case, is a simple indicator for whether the system entered into an unsafe state ($C(s, a) = 1$ if the state $s$ is unsafe and $C(s, a) = 0$ otherwise).

With this definition, the mean cost $\text{MC}(\tau) = \frac{1}{T} \sum^T_{t = 1} C(s_t, a_t)$ is an estimate of the probability of unsafe states visited along one trajectory.

The goal is to find a policy $\pi$ in the policy set $\Pi$ with high expected reward and low safety cost.

While ideally we would like to have zero collisions, in practice we can accept some violations.

Then optimal policy can be rewritten by relaxing the hard constraints on $C_i$:
$$
\pi^ = \arg \min_{C_i} \max \{\text{MR}(\tau)\}
$$
One requirement for this setting is to ensure that all environment are ergodic MDPs which guarantee that any state is reachable from any other state by following a suitable policy.

In this setup, safety violations can be accepted as at any time-step $t$ it is possible to recover a safe state.


##### 5.2 Safe exploration with world model learning

We followed previous work with the exception of SafeCar-Goal, for which we used the original version by [[📝 Benchmarking safe exploration in deep reinforcement learning]] with the position of the unsafe areas randomly resampled at the beginning of each episode.

**Original Safety Gym 환경은 에피소드마다 장애물 위치가 고정돼 있었던건가?!**