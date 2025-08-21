---
title: Improving safety in deep reinforcement learning using unsupervised action planning
author: Hao-Lun Hsu, Qiuhua Huang, Sehoon Ha
published: ICRA 2022
created: 2025-08-21 10:45
status: Queued
category:
tags:
  - ICRA
  - Sehoon-Ha
---
## The Purpose of This Study


## Lit. Review


## Methods


## Results & Discussion


## Critique


## Cited By

[[🇺🇸 Guided Safe Shooting (model based reinforcement learning with safety constraints)]]

A different strategy involves storing all the "recovery" actions that the agent took to leave unsafe regions in a separate replay buffer.

This buffer is then used whenever the agent enters an unsafe state by selecting the most similar transition in the safe replay buffer and performing the same action to escape the unsafe state.