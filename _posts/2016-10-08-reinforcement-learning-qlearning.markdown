---
layout: post
title:  "Q-learning"
date:   2016-10-08 12:00:00 +0100
categories: reinforcement-learning
---

$
\def\RR{\mathbb{R}}
\def\tr{\mathrm{tr}}
\def\th{\mathrm{th}}
\def\EE{\mathbb{E}}
\def\coloneqq{\colon=}
\def\grad{\nabla}
\def\Qhat{\hat{Q}}
$

# Q-learning

Define the state-action value function to be the value of being in state $s$ and
choosing action $a$; denote this by $Q \colon \Sigma \times A \to \RR$.

Recall that the value function, $V \colon \Sigma \to \RR$, is defined as the
expected future reward of following a certain policy given that you start in
some state. That is,

$$
V(s) = \EE_\tau[\sum_{t=0}^T \gamma^t r_t].
$$

Then the $Q$-function is defined as

$$
Q(s,a) = \EE_\tau[r(s,a,s_1) + \sum_{t=1}^T \gamma^t r_t].
$$

There exists an optimal $Q$-function, which we denote by $Q^*$. This satisfies
the following Bellman equation

$$
Q^*(s, a) = \EE_s'[R(s,a,s') + \gamma \max_{a'} Q^*(s, a')].
$$

In words, the Bellman equation says that the $Q$-value of being in state $s$ and
choosing action $a$ is the expected reward of being in state $s$ and choosing
action $a$ plus the expected maximum $Q$-value over all actions in the next
state.

For a finite state space, and if it is small enough, it is possible to converge
on the optimal $Q$-function by a process called $Q$-value iteration. However, we
are not interested in finite state spaces in this post. Instead, we consider
$Q$-function approximation: we try and learn a function that approximates the
$Q$-function.

There are several reasons for this. Firstly, in a lot of interesting problems,
the state space is either infinite or insurmountably large. Secondly, if we try
to learn a value $Q(s,a)$ for each state-action pair, then we do not generalise
the $Q$-value of one state to that of a very similar state. Both of these
problems are demonstrated by the Cart and Pole problem: we have to represent the
position and velocity of the cart, both of which take values in an interval.
Moreover, if the only difference between two states is the position of the cart,
then we should expect to take a similar action in both states.

# Approximate $Q$-functions

We are led to consider approximate $Q$-functions, where we try to approximate
the optimal $Q$-function by some parametrised function $\Qhat(s, a; \theta)$.
Then our task is to learn the best parameter vector $\theta$.

In this post we use neural networks to approximate the $Q$-function. Start with
a randomly initialised neural network, denoted $\Qhat(s,a; \theta)$, and then we
try to find $\theta$ such that $\Qhat$ satisfies the Bellman equation. Sample
the environment and collect lots of transitions $(s, a, r, s')$, where $s$ is
some state, $a$ is the action we choose in that state, $r$ is the reward we get
from the environment, and $s'$ is the state the environment transitions us to.

We use the Bellman equation to get the $Q$-function to converge. Explicitly, we
minimise the loss

$$
L = [\Qhat(s, a) - \max_{a'} \Qhat(s', a') - r]^2,
$$

averaged over many transitions.

# Check that a linear $Q$-function works for our problem
It's usually a good idea to try out the simplest to implement, and quickest to
run, methods first, and only move on to more complicated methods when these have
been shown insufficient. It turns out that computing a good $Q$-function in the
Cart and Pole problem can be solved using the cross-entropy method, rather than
gradient descent.

Since we are interested in using deep q-learning, we do want to use the more
complicated method as well, and check that it works in the Cart and Pole
problem; however, we can exploit the cross-entropy method to check that our
proposed form of the $Q$-function actually works.

We have to choose the architecture of the neural network. The simplest idea is
simply to use a linear network:

$$
\Qhat(s; \theta) = s \theta,
$$

where $s$ is a $1 \times 4$ vector and $\theta$ is a $4 \times 2$ matrix. You'll
notice that $\Qhat$ does not mention the action, which is because the output of
$\Qhat(s; \theta)$ is a $1 \times 2$-vector; the $\Qhat$ value of moving left is
the first value and the $\Qhat$ value of moving right is the second value. This
means we can use just one network, rather than a separate one for each action.

You can see the code for the cross entropy method applied to $Q$-learning
[here](https://github.com/cgnicholls/reinforcement-learning/blob/master/cartpole/cartpole-qlearning-crossentropy.py).
