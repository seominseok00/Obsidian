## The Purpose of This Study

#### Abstract

Model-based RL algorithms can attain excellent sample efficiency, but often lag behind the best model-free algorithms in terms of asymptotic performance.

This is especially true with high-capacity parametric function approximators, such as deep networks.

In this paper, we study how to bridge this gap, by employing uncertainty-aware dynamics models.


### 1. Introduction

Current model-free reinforcement learning algorithms are quite data-expensive to train.

A promising direction for reducing sample complexity is to explore model-based reinforcement learning (MBRL) methods, which proceed by first acquiring a predictive model of the world, and then using that model to make decisions.

However, the asymptotic performance of MBRL methods on common benchmark tasks generally lags behind model-free methods.

Our approach is based on two observation.

1. Model capacity is a critical ingredient in the success of MBRL methods

While efficient models such as Gaussian processes can learn extremely quickly, they struggle to represent very complex and discontinuous dynamical systems.

By contrast, neural network (NN) models can scale to large datasets with high-dimensional inputs, and can represent such systems more effectively.
However, NNs tend to overfit on small datasets, making poor predictions far into the future.

2. This issue(NNs training issue) can be mitigated by properly incorporating uncertainty into the dynamics model.

**This paper's main contribution**
- Propose method reaches the asymptotic performance of state-of-the-art model-free RL methods on benchmark control tasks.
- Proposed method isolate two distinct classes of uncertainty(aleatoric, epistemic)
- Present a systematic analysis of how incorporating uncertainty into MBRL with NNs affects performance, during both model training and planning.


## Lit. Review

### 2. Related work

Model choice in MBRL is delicate.

Because we desire effective learning in both low-data regimes (at the beginning) and high-data regimes (in the later stages of the learning process).

- Bayesian nonparametric models(e.g., Gaussian processes) are often the model of choice in low-dimensional problems where data efficiency is critical. However, such model introduce additional assumptions(smoothness assumption) on the system.
- Parametric function approximators(e.g., Neural networks) have constant-time inference(unlike Gaussian processes) and tractable training in the large data regime, and have the potential to represent more complex functions, including non smooth dynamics that are often present in robotics. However, most works that use NNs focus on deterministic models, consequently suffering from overfitting in the early stages of learning.

For this reason(overfitting issue in deterministic NN models), our approach is able to achieve even higher data-efficiency than prior deterministic MBRL methods such as [[🇺🇸 Neural network dynamics for model-based deep reinforcement learning with model-free ﬁne-tuning]].


### 3. Model-based reinforcement learning

- state: $s \in \mathbb{R}^{d_s}$
- action: $a \in \mathbb{R}^{d_a}$
- transition function: $f_\theta: \mathbb{R}^{d_s + d_a} \mapsto \mathbb{R}^{d_s}$

Learning forward dynamics is thus the task of fitting an approximation $\tilde{f}$ of the true transition function $f$, given the measurements $\mathcal{D} = \{(s_n, a_n), s_{n + 1}\}^N_{n = 1}$ from the real system.

Once a dynamics model $\tilde{f}$ is learned, we use $\tilde{f}$ to predict the distribution over state-trajectories resulting from applying a sequence of actions.

By computing the expected reward over state-trajectories, we can evaluate multiple candidate action sequences, and select the optimal action sequence to use.


## Methods

### 4. Uncertainty-aware neural network dynamics models

Any MBRL algorithm must select a class of model to predict the dynamics.

This choice is often crucial for an MBRL algorithm, as even small bias can significantly influence the quality of the corresponding controller.

A major challenge is building a model that performs well in low and high data regimes.

To account for uncertainty, we study NNs that model two types of uncertainty.

1. `Aleatoric uncertainty` arises from inherent stochasticities of a system, e.g. observation noise and process noise.
   Aleatoric uncertainty can be captured by outputting the parameters of a parameterized distribution, while still training the network discriminatively.
2. `Epistemic uncertainty` corresponds to subjective uncertainty about the dynamics function, due to a lack of sufficient data to uniquely determine the underlying system exactly.

**We use combinations of probabilistic networks to capture aleatoric uncertainty and 'ensembles' to capture epistemic uncertainty.**


#### Probabilistic neural networks

We define a probabilistic NN as a network whose output neurons simply parameterize a probability distribution function, capturing aleatoric uncertainty, and should not be confused with Bayesian inference.

We use the negative log prediction probability as our loss function.
$$
\text{loss}_p(\theta) = - \sum^N_{n = 1} \log \tilde{f}_\theta(s_{n + 1}|s_n, a_n)
$$
Such network outputs model aleatoric uncertainty. (otherwise known as heteroscedastic noise meaning the output distribution is a function of the input)

However, it does not model epistemic uncertainty, which cannot be captured with purely discriminative training.


#### Deterministic neural networks

We define a deterministic NN as a special case probabilistic network that outputs delta distributions centered around point predictions denoted as
$$
\tilde{f}_\theta(s_t, a_t): \tilde{f}_\theta(s_{t + 1}|s_t, a_t) = \text{Pr}(s_{t + 1}|s_t, a_t) = \delta(s_{t + 1} - \tilde{f}_\theta(s_t, a_t))
$$
trained using the MSE loss: $\text{loss}_D(\theta) = \sum^N_{n = 1} \| s_{n + 1} - \tilde{f}_\theta(s_n, a_n) \|$. (Although MSE can be interpreted as $\text{loss}_p(\theta)$ with a Gaussian model of fixed unit variance, in practice cannot be used for uncertainty-aware propagation, since it does not corresponds to any notion of uncertainty)


#### Ensembles

A principled means to capture epistemic uncertainty is with Bayesian inference.

However, ensembles of bootstrapped models are even simpler: no additional hyperparameters need be tuned, whilst still providing reasonable uncertainty estimates.

We consider ensembles of $B$-many bootstrapped models, using $\theta_b$ to refer to the parameters of our $b$th model $\tilde{f}_{\theta_b}$. (ensembles can be composed of deterministic models or probabilistic models)

Each of our bootstrap models have their unique dataset $\mathbb{D}_b$, generated by sampling (with replacement) $N$ times the dynamics dataset recored so far $\mathbb{D}$, where $N$ is the size of $\mathbb{D}$. (We found $B = 5$ sufficient for all our experiments)


### 5. Planning and control with learned dynamics

 MBRL planning in discrete time over long time horizons is generally performed by using the dynamics model.

When planning, we might consider each action $a_t$ to be a function of state, forming a policy $\pi: s_t \rightarrow a_t$, a  function to optimize.
Alternatively, we can plan and optimize for a sequence of actions, a process called model predictive control (MPC).
At each time step $t$, the MPC controller applies the firzst action $a_t$ of the sequence of optimized actions  $\arg\max_{a_{t:t + T}} \sum^{t + T}_{\tau = t} \mathbb{E}_{\tilde{f}}[r(s_\tau, a_\tau)]$.

A common technique to compute the optimal action sequence is a random sampling shooting method.
[[🇺🇸 Neural network dynamics for model-based deep reinforcement learning with model-free ﬁne-tuning]] use deterministic NN models and MPC with random shooting to achieve data efficient control in higher dimensional tasks than what is feasible for GPs to model.

Our work improves upon [[🇺🇸 Neural network dynamics for model-based deep reinforcement learning with model-free ﬁne-tuning]]'s data efficiency in two ways:
1. We capture uncertainty in modeling and planning, to prevent overfitting in the low-data regime.
2. We use CEM instead of random-shooting, which samples actions from a distribution closer to previous action samples that yield high reward.

Computing the expected reward using recursive state prediction in closed-form is generally intractable.
Multiple approaches to approximate uncertainty propagation can be found in the literature. (deterministic, particle, parametric methods)
*Parametric methods have been successfully used in MBRL, experimental results suggest that particle approaches can be competitive both computationally and in terms of accuracy, without making strong assumptions about the distribution used.*
We use particle-based propagation, specifically suited to our PE(Probabilistic Ensembles) dynamics models which distinguishes two types of uncertainty.

#### 5.1 Our state propagation method: trajectory sampling (TS)

Our method to predict plausible state trajectories begins by creating $P$ particles from the current state, $s^p_{t = 0} = s_0 \forall p$.

Each particle is then propagated by: $s^p_{t + 1} \sim \tilde{f}_{\theta_{b(p, t)}}(s^p_t, a_t)$, according to a particular bootstrap $b(p, t)$ in $\{1, \ldots, B\}$, where $B$ is the number of bootstrap models in the ensemble.

- TS1 refers to particles uniformly re-sampling a bootstrap per time step.
- TS$\infty$ refers to particle bootstraps never changing during a trial.
	- An advantage of using TS$\infty$ is that aleatoric and epistemic uncertainties are separable.

We found $P = 20$(number of particles) and $B = 5$(number of bootstrap models) sufficient in all our experiments.

### 6. Algorithm summary

![image](Paper-Reviews/RL/Model-Based/Deep%20Reinforcement%20Learning%20in%20a%20Handful%20of%20Trials%20using%20Probabilistic%20Dynamics%20Models/imgs/algorithm.png)


### Appendix

#### A.1 Well behaved probabilistic networks

Whilst(While) this variance is well behaved at points within the training distribution, its value is undefined outside the trained distribution.
In fact, during the training there is no explicit loss term that regulate the behavior of the variance outside of the training points. (Thus, when this model is then evaluated at previously unseen state, the outputted variance collapse to zero, or explode toward infinity.)

As a remedy, we found that in our model lower bounding and upper bounding the output variance such that they could not be lower or higher than the lowest and highest values in the training data significantly helped.

We used the following code with automatic differentiation:
```
logvar = max_logvar - tf.nn.softplust(max_logvar - logvar)
logvar = min_log_var + tf.nn.softplus(logvar - min_logvar)
var = tf.exp(logvar)
```

#### MPC action selection

We compare to the cross-entropy method (CEM), which iteratively samples solutions from a candidate distribution that is adjusted based on the best sampled solution.
## Results & Discussion

1. Our results show that model-based reinforcement learning with neural network dynamics models can achieve results that are competitive not only with Bayesian nonparametric models such as GPs, but also on par with model-free algorithms such as PPO and SAC in terms of asymptotic performance, while attaining substantially more efficient convergence.
2. The individual components of our model-based reinforcement learning algorithms are not individually new - prior works have suggested both ensembling and outputting Gaussian distribution parameters, as well as the use of MPC for model-based RL.
3. Our results indicate that the gap in asymptotic performance between model-based and model-free reinforcement learning can, at least in part, be bridged by incorporating uncertainty estimation into the model learning process.
4. A compelling alternative class of methods uses the model to train a parameterized policy.
5. A promising direction for future work is investigate how policy learning can be incorporated into our framework to amortize the cost of planning at test-time. (Our initial experiments with policy learning did not yield an effective algorithm by directly propagating gradients through our uncertainty-aware models. We believe this may be due to chaotic policy gradients.)