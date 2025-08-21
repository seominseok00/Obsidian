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