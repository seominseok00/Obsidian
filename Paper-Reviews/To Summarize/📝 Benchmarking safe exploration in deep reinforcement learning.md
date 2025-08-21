---
title: Benchmarking safe exploration in deep reinforcement learning
author:
published:
created: 2025-08-21 10:48
status:
category:
tags:
---

## The Purpose of This Study


## Lit. Review


## Methods


## Results & Discussion


## Critique


## Cited By

[[🇺🇸 Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm]]

The algorithm in these papers(Lagrangian-based methods) are based on multi-timescale stochastic approximation with updates of the Lagrange parameter performed on the slow timescale, the policy updates on the medium timescale and the updates of the value function (for a given policy) performed on the fast timescale.

Lagrangian relaxation of the Constrained RL problem is used and combined with PPO to give a PPO-Lagrangian algorithm and with TRPO to give a TRPO-Lagrangian algorithm.

These algorithms were seen to outperform CPO in terms of constraint satisfaction on several environments in Safety Gym.

Also, these algorithms are simpler to implement and tune.