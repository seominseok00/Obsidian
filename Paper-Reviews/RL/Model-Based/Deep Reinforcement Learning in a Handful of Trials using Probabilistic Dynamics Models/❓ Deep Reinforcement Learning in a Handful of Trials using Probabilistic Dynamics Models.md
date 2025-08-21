### 2. Related work

**Gaussian Process란?**

### 3. Model-based reinforcement learning

**reward function도 같이 학습하는건지?**

### 4. Uncertainty-aware neural network dynamics models

#### Probabilistic neural network

**Bayesian inference, Bayesian nonparametric models는 무슨 차이?**


### 5. Planning and control with learned dynamics

**CEM(Cross Entropy Method)란?**

[The Cross-Entropy Method for Optimization](https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf)

MPC는 현재 상태에서 길이 $T$의 action sequences를 최적화해 첫 번째 행동만 실행하고, 다음 스텝에서 다시 최적화하는 프레임워크

이 최적화 문제를 풀기 위한 optimizer가 CEM

#### 5.1 Our state propagation method: trajectory sampling (TS)

**Trajectory multimodality, Particle separation이 뭔지?**

현재 상태 $s_t$에서 particle 개수 $P$만큼 상태 $s_t$를 복제.

이후 bootstrap model $B$개 중 하나를 매 스텝마다 random sampling해서 다음 particle을 계산하는 것이 TS1, 하나를 고정해놓고 쭉 사용하는 것이 TS$\infty$.

Trajectory multimodality는 예측 trajectory의 cluster 수
Particle Seperation은 각 시점에서의 particle의 퍼짐 정도

![image](fig1.jpg)

**TS1으로 bootstrap model을 매 time step마다 바꿔서 전파한다고 해도, 결국 모든 bootstrap model은 완벽하지 않을 것이고, 그래서 에러가 누적되는건 같을텐데 그러면 trajectory multimodality도 완화시키지 못하는 것 아닌지?**

TS1은 모델별 편향이 시간에 따라 같은 방향으로 누적되는 것을 끊어주기 때문에 trajectory multimodality, particle separation 문제를 완화

남은 multimodality, separation은 실제 dynamics의 noise 때문이며, 이거까지 완전히 제거하지 않기 때문에 soft restriction이라고 함.


#### A.3 One-step predictions of learned models

**holdout data란?**

모델 학습시에 사용하지 않고 따로 떼어둔 평가용 데이터


### 8. Discussion & Conclusion

A promising direction for future work is investigate how policy learning can be incorporated into our framework to amortize the cost of planning at test-time.

**Amortize란?**

금융에서는 빚을 여러 달로 나눠서 갚다 (분할상환)라는 의미인데, 머신러닝에서는 매번 새로 계산해야 하는 작업을 학습된 함수로 대체해서 계산 비용을 줄인다는 의미로 쓰임