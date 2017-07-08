---
layout: post
title:  "Reinforcement learning II"
date:   2016-08-21 12:00:00 +0100
categories: reinforcement-learning
---

$
\def\RR{\mathbb{R}}
\def\tr{\mathrm{tr}}
\def\th{\mathrm{th}}
\def\EE{\mathbb{E}}
\def\coloneqq{\colon=}
\def\grad{\nabla}
$

# The policy gradient method on the cart and pole problem #

We now implement the ideas from the previous post. Recall that we are trying to
compute the gradient of the expected reward of our policy with respect to the
parameters in the policy. Recall that last time we defined $\hat{g}(\tau)$ as an
estimator of the expected gradient. It is given by

$$
\hat{g}(\tau) \coloneqq R_\tau \grad_\theta \log P(\tau | \pi; \theta) =
(\sum_{t'=1}^T r_{t'}) \grad_\theta \sum_{t=0}^{T-1} \log \pi(a_t | x_t;
\theta),
$$

and satisfies $\EE_\tau[\hat{g}(\tau) | \pi; \theta] = \grad_\theta
\EE_\tau[R_\tau | \pi; \theta]$. Thus it is an unbiased estimator: if we compute
many samples of $\hat{g}(\tau)$ and take the average of the samples, we will get
close to the gradient we want.

Given the formula above, it suffices to compute $\grad_\theta \log \pi(a_t | x_t;
\theta)$ for each state $x_t$ and action $a_t$ that we see.

## Computing the gradient of our policy ##

We now fix the form of our policy. In general, we could choose any function that
depends on some parameter $\theta$, but we choose the following here.

$$
\begin{align*}
\pi(a_R | x; \theta) &= \sigma(x \theta^\tr) \\
\pi(a_L | x; \theta) &= 1 - \sigma(x \theta^\tr).
\end{align*}
$$

Thus, when we see state $x$ we compute $\sigma(x \theta^\tr)$ as the probability
that we will move right. The remaining probability, i.e. $1 - \pi(a_R | x;
\theta)$, is the probability that we move left.

We want to compute $\grad_\theta \log \pi(a_R | x; \theta)$ and $\grad_\theta
\log \pi(a_L | x; \theta)$. Since $\grad_u \log f(u) = \frac{1}{f(u)} \grad_u
f(u)$, it suffices to compute $\grad_\theta \pi(a_R | x; \theta)$ and
$\grad_\theta \pi(a_L | x; \theta)$. We start by computing the derivative of the
sigmoid function:

$$
\sigma'(u) = \frac{d}{du} \frac{1}{1 + e^{-u}} = \frac{-1}{(1+e^{-u})^2}
(-e^{-u}).
$$

Rearranging this, we can write $\sigma'(u)$ in terms of $\sigma(u)$:

$$
\sigma'(u) = \frac{1}{1+e^{-u}} \frac{e^{-u}}{1+e^{-u}} = \sigma(u)(1 -
\sigma(u)).
$$

Now it is easy to see that

$$
\grad_\theta \sigma(x \theta^\tr) = \sigma(x \theta^\tr) (1 - \sigma(x
\theta^\tr)) x.
$$

Finally, we see that

$$
\begin{align*}
\grad_\theta \log \pi(a_R | x; \theta) &= \pi(a_L | x; \theta) x \\
\grad_\theta \log \pi(a_L | x; \theta) &= - \pi(a_R | x; \theta) x.
\end{align*}
$$

Then we just substitute these expressions into

$$
\hat{g}(\tau) = (\sum_{t'=1}^T r_{t'}) \sum_{t=0}^{T-1} \grad_\theta \log
\pi(a_t | x_t; \theta),
$$

and update by:

$$
\theta \leftarrow \theta + \alpha \hat{g}(\tau).
$$

Code to implement this can be found
[here](https://github.com/cgnicholls/reinforcement-learning/blob/master/cartpole/vanillapolicygradient.py).

## Improvements to make this work better ##

The vanilla version of this actually works pretty well. One improvement to make
is to replace $R_\tau$ with $r_{t+1} + \cdots + r_T$.

Improvement to make: replace $R_\tau$ with the advantage:

$$
A_\tau \coloneqq R_\tau - b,
$$

where $$b$$ is some baseline estimate of the value. This has the same
expectation as $$R_\tau$$ but a lower variance, which makes it easier for the
model to converge.
