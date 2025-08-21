## The Purpose of This Study

### Abstract

Safe exploration is a critical issue in applying RL algorithms in the real world.

We propose an On-policy Model-based Safe Deep RL algorithm in which we learn the transition dynamics of the environment in an online manner as well as find a feasible optimal policy using the Lagrangian Relaxation-based Proximal Policy Optimization.

We use an ensemble of neural networks with different initializations to tackle epistemic and aleatoric uncertainty issues faced during environment model learning.We highlight the issue that arises due to the use of truncated horizon in Constrained RL and suggest a way to incorporate that in our setting.

### 1. Introduction

In our work, we focus on a constrained-based notion of safety.

In Constrained-based RL, the goal is to maximize long-term expected reward returns and keep the expected cost-returns below a prescribed threshold.

Constrained Markov Decision Process (CMDP) provides a framework to keep task performance specifications and safety specifications separate from one another.

The existing model-free algorithms used in the Constrained-RL setting suffer from low sample efficiency in terms of environment interactions, i.e., these algorithms require large number of environment interactions (to converge)

This serves as the motivation for us to use a Model-based approach for Constrained-RL.

We propose a simple and sample efficient model-based approach for Safe Reinforcement Learning which uses Lagrangian relaxation to solve the constrained RL problem.

## Lit. Review

### 2. Related Work

- Model-free RL
In more recent, in [[📚 Constrained Policy Optimization]], a trust region based constrained policy optimization (CPO) framework is proposed, which involved approximation of the problem using surrogate functions for both the objective and the constraints and included a projection step on policy parameters that needed backtracking line search, making it complicated and time-consuming.

In [[📝 Benchmarking safe exploration in deep reinforcement learning]], Lagrangian relaxation of the Constrained RL problem is used and combined with PPO to give a PPO-Lagrangian algorithm.
These algorithms were seen to outperform CPO in terms of constraint satisfaction on several environments in Safety Gym.
Also, these algorithms are simpler to implement and tune.

In [[📚 Safe exploration in continuous action spaces]], the authors formulated a state-wise constrained policy optimization problem where at each transition a constraint needs to be satisfied and an analytical method for correcting the unsafe action using a safety layer trained using random exploration was proposed.

- Model-based RL
In [[🇺🇸 Safe Model-Based Reinforcement Learning with Robust Cross-Entropy Method]], a model based approach is proposed to learn the system dynamics and cost model.
Then roll-outs from the learned model are used to optimize the policy using a modified cross-entropy based method which involves sampling from a distribution of policies, sorting sample policies based on constraint functions and using them to update the policy distribution.

In [[📚 Safe Reinforcement Learning by Imagining the Near Future]], penalized reward functions are used instead of a separate cost function, the model of the environment is learned and the soft-actor critic algorithm is used to optimize the policy.


### 3. Background

#### 3.3 Model-based Constrained RL

We formulate a constrained RL problem using a model-based framework as follows:
$$
\max_{\pi_\theta \in \Pi_\theta} J^R_m(\pi_\theta) \; \text{s.t.} \; J^C_m(\pi_\theta) \leq d
\tag{21}
$$
$$
J^R_m(\pi_\theta) = \mathbb{E} \left[\sum^\infty_{t = 0} \gamma^t R(s_t, a_t, s_{t + 1} | s_0 \sim \mu, s_{t + 1} \sim P_\alpha(\cdot|s_t, a_t), a_t \sim \pi_\theta, \forall t) \right]
\tag{22}
$$
$$
J^C_m(\pi_\theta) = \mathbb{E} \left[\sum^\infty_{t = 0} \gamma^t C(s_t, a_t, s_{t + 1}) |s_0 \sim \mu, s_{t + 1} \sim P_\alpha(\cdot|s_t, a_t), a_t \sim \pi_\theta, \forall t \right]
\tag{23}
$$
In the above, $P_\alpha(\cdot|s_t, a_t)$ is an $\alpha$-parameterized environment model, $d$ is a human prescribed safety threshold for the constraint and $\Pi_\theta$ is the set of all $\theta$-parameterized stationary policies.

We assume the initial state $s_0$ is sampled from the true initial state distribution $\mu$ and then $s_{t + 1} \sim P_\alpha(\cdot|s_t, a_t), \forall > 0$.

We would use approximation of environment $P_\alpha$ to create 'imaginary' roll-outs to estimate the reward and cost returns required for policy optimization algorithms.


### 4. Challenges in Environment Model Learning

#### Handling aleatoric and epistemic uncertainties

`Aleatoric Uncertainty` refers to the notion of natural randomness in the system which leads to variability in outcomes of an experiment.

This uncertainty is irreducible because it is a natural property of the system.

Hence in such case, giving measure of uncertainty in model's prediction is a good practice.


`Epistemic Uncertainty` refers to the notion of lack of sufficient knowledge in the model as a result of which the model does not generalize.


For the learning environment model, we also use an ensemble of $n$ neural networks with random initialization.
Each neural network's output parameterizes a multivariate normal distribution with diagonal covariance matrix.

Suppose the $i$th neural network in the ensemble is parameterized by $\alpha_i$ and the mean and standard deviation outputs are $\mu_{\alpha_i}$ and $\sigma_{\alpha_i}$ respectively.

For the $i$th neural network parameterized by $\alpha_i$, the loss function $L(\alpha_i)$ is given as follows:
$$
L(\alpha_i) = \sum^M_{t = 1}[\mu_{\alpha_i}(s_t, a_t) - s_{t + 1}]^T \sum^{-1}_{\alpha_i}(s_t, a_t)[\mu_{\alpha_i}(s_t, a_t) - s_{t + 1}] + \log |\sum_{\alpha_i}(s_t, a_t)|
$$
where $\mu_{\alpha_i}(s_t, a_t)$ is the mean vector output of the $i$th neural network and $\sum_{\alpha_i}(s_t, a_t)$ is the covariance matrix which is assumed to be a diagonal matrix.

#### Aggregation of Error

In model-based RL, as we move forward along the horizon, the error due to approximation starts aggregating and predictions from the approximated model tend to diverge significantly from the true model.

In order to tackle this problem, most of the model-based RL approches use shorter (or truncated) horizon during the policy optimization phase.

We use truncated horizon in our approach.


#### Implication of using truncated horizon in Constrained RL

When we use truncated horizon in Constrained RL, it leads to underestimation of cost returns (23) under the current policy.

This can lead to constraint violation in the real-environment where the cost objective is based on the infinite horizon cost return.

We propose a hyperparameter-based approach to deal with this problem in the [[#5. Model-based PPO Lagrangian]]
## Methods

### 5. Model-based PPO Lagrangian

![image](Paper-Reviews/Constrained%20RL/Model-Based/Model-based%20Safe%20Deep%20Reinforcement%20Learning%20via%20a%20Constrained%20Proximal%20Policy%20Optimization%20Algorithm/imgs/algorithm.png)

It is difficult to evaluate the policy without interacting with the real environment accurately.

For this we compute the Performance Ratio (PR) metric using ensemble models (see [[📚 Model-ensemble trust-region policy optimization]]) that is defined as follows:
$$
\text{PR} = \frac{1}{n} \sum^n_{i = 1} \mathbb{1}(\zeta^R(\alpha_i, \theta_t) > \zeta^R(\alpha_i, \theta_{t - 1}))
$$
where $\zeta^R(\alpha_i, \theta_t) = \sum^T_{t = 0} \gamma^t R(s_t, a_t, s_{t + 1}), s_0 \sim \mu$ and $\forall t \geq 0 : s_{t + 1} \sim P_{\alpha_i}(\cdot|s_t, a_t), a_t \sim \pi_{\theta_t}(\cdot|s_t)$ 

This measures the ratio of the number of models in which policy is improved to the total number of models in ensemble $(n)$.

If $\text{PR} > \text{PR}_{\text{threshold}}$, we continue training using the same model, if not then we break and re-train our environment model on data collected from the new update policy.

Another challenge that we encounter is the underestimation of $J^C(\pi_\theta)$ resulting from using a truncated horizon (of length $H$ in step 7 of Algorithm 1) to reduce the aggregation of error.

So we need to make the safety threshold $(d)$ stricter.

We make safety threshold stricter using a hyperparameter $0 \leq \beta < 1$ by modifying the Lagrange multiplier update as follows:
$$
\lambda_n = [\lambda_n - \eta_2(n)(J^C(\pi_\theta) - d * \beta)]_+
$$

The variation of expected cost returns and reward returns with respect to $\beta$ is shown in Figure 1.

![image](fig1.png)


### Appendix

#### A. Hyper-parameters and finer experimental details

![image](hyperparameters.png)


## Results & Discussion

### 6. Experimental Details and Results

We test our approach on Safety Gym environments with modified state representations as used in other model-based Safe RL approaches [[🇺🇸 Safe Model-Based Reinforcement Learning with Robust Cross-Entropy Method]].

We increase the difficulty of PointGoal and CarGoal environments by increasing the number of hazards from 10 to 15.

We compare our approach(MBPPO-Lagrangian) with Unconstrained PPO, Constrained Policy Optimization(CPO), PPO-Lagrangian, safe-LOOP(model-based approach)

We use [rliable](https://github.com/google-research/rliable) for plotting.

### 7. Conclusions

Note that we chose the above baselines because they are specifically designed to solve a constrained optimization problem of the same structure as (1)-(3).

In safe-LOOP, approximation of the value function is used to provide long-term reasoning instead of using a truncated horizon.
A limitation of this approach lies in the fact that approximations of reward and cost value functions are used.

The most challenging part of model-based approaches is to learn the environment model.

1. The first issue is of computational resources and the time overhead that is needed. (Our algorithm does much better than safe-LOOP in terms of running time.)
2. In safe exploration settings, agents do not explore as much as an unconstrained agent would do. This adds to the complexity of model learning. (We found this more pronounced in high-dimensional environments like DoggoGoal1.)

**Future works**

- Increasing reward performance of Lagrangian-based approaches.
- Devising better ways for model-learning in high-dimensional state representations in safe RL settings where exploration is limited.
- Adapt off-policy natrual actor-critic algorithms such as in the setting of constrained MDPs and study their performance both theoretically and epirically.


## Critique