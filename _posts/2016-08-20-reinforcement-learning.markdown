---
layout: post
title:  "Reinforcement learning"
date:   2016-08-20 11:00:00 +0100
categories: reinforcement-learning
---

\\(
\def\RR{\mathbb{R}}
\def\tr{\mathrm{tr}}
\def\th{\mathrm{th}}
\def\EE{\mathbb{E}}
\def\coloneqq{\colon=}
\def\grad{\nabla}
\\)

# Introduction to reinforcement learning #

I've been looking into reinforcement learning recently, and discovered the
[OpenAI gym](https://gym.openai.com/). This has many reinforcement learning
problems implemented, and with a nice API. I really enjoyed reading their
[Getting Started guide](https://gym.openai.com/docs/rl), and thought I would
give my own account of it. Here we describe how to solve a classic control
problem: the cart and pole.

## The cart and pole problem ##

In this problem a cart attempts to vertically balance a pole by moving left and
right along a one-dimensional space. At each time step, the cart receives its
position and velocity, and the angle and angular velocity of the pole. We let
$x$ denote the observation, and we use this as the state. Let $a_L$ denote the
action 'accelerate left' and $a_R$ denote the action 'accelerate right'.

We model this problem as a Markov decision process, since the behaviour of the
system only depends on the current state, i.e. our state satisfies the Markov
property: $P(x_{t+1} | x_t) = P(x_{t+1} | x_1, x_2, \ldots, x_t)$. In words, the
probability of transitioning to state $x_{t+1}$ given that you were in state
$x_t$ is independent of all previous states. This really means that your state
captures all you need to know about the system. In contrast, if our state did
not include the cart's velocity, then it wouldn't be Markov, because $x_t$
doesn't give the velocity, while $x_t$ and $x_{t-1}$ together give you a crude
estimate of the velocity.

At time step $t$ we receive an observation $x_t$ from the environment. We then
choose an action $a_t$, and the environment transitions stochastically into a
new state. We receive a reward $r_{t+1}$, and we see the next state $x_{t+1}$.
This process repeats until we see the terminal state: the pole falls off the
cart. In the cart and pole problem, the reward is $r_t = +1$ for every time step
$t$; thus the incentive is to keep the pole balanced for as long as possible. In
general, the reward may depend on $x_t$ and $a_t$.

### Playing randomly ###

Let's first see how well a random agent does against the problem. This is a good
idea, as it gives a benchmark to compare our algorithms against. Sometimes it
looks like your algorithm is working well, when in fact it's not doing any
better than taking decisions uniformly at random.

The following code implements a random agent in OpenAI Gym:

~~~~python
import numpy as np
import random
import gym

# Make the CartPole environment
env = gym.make('CartPole-v0')

# Random agent
def random_agent(num_episodes):
    # Play num_episodes episodes
    for i_episode in range(num_episodes):
        # Initialise episode reward
        episode_reward = 0
        # Get the initial observation
        observation = env.reset()
        # Run for at most 1000 time steps
        for t in range(1000):
            # Render the environment
            env.render()
            # Randomly choose an action from the action space
            action = env.action_space.sample()
            # Take this action
            observation, reward, done, info = env.step(action)
            # Update episode reward
            episode_reward += reward
            # If the episode is over, print the episode reward
            if done:
                print("Total reward for episode: {}".format(episode_reward))
                break

# Play the random agent for 10 episodes
random_agent(10)
~~~~

### Playing with a policy

Instead of playing randomly, we want to choose our actions based on our current
state. We achieve this using a probability distribution $P(a | x)$ over the
possible actions, given the state, from which we sample our action at every
step. If the state space were discrete, then the description $P(a | x)$ would
suffice, but to work with a continuous state space, it is convenient to use a
parametrised distribution $P(a | x; \theta)$, where $\theta \in \RR^d$ for some
$d$. Following conventional notation, we now write $\pi(a | x; \theta)$ instead
of $P(a | x; \theta)$, and we henceforth call this the policy.

Thus $\pi(a | x; \theta)$ is just a probability distribution that depends on $x$
and $\theta$ that we will use to choose our actions: at time step $t$, if we are
in state $x_t$, then we sample $a_t$ from the distribution $\pi(a_t | x_t;
\theta)$.

In the case of the cart and pole problem, the state already consists of
excellent features for predicting the action, and so we just use linear
combinations of the features, and then normalise to get a probability
distribution. More formally, we let $\theta \in \RR^4$, and use the policy

$$
\pi(a | x; \theta) =
\begin{cases}
\sigma(x \theta^\tr), & a = a_R \\
1 - \sigma(x \theta^\tr), & a = a_L,
\end{cases}
$$

where $\sigma$ is the sigmoid function; $\sigma(u) = 1 / (1 + e^{-u})$. The
sigmoid function has range $(0, 1)$, and so this policy is a well-defined
probability distribution over actions given states.

Essentially all we are doing here is using the dot product $x \theta^\tr$ as a
predictor, and then normalising everything to get a probability distribution.

At time step $t$, we receive $x_t$, and compute $\pi(a_t | x_t; \theta)$. Then
we sample $a_t \sim \pi(a_t | x_t; \theta)$, and the environment sends us back
$r_{t+1}$ and $x_{t+1}$. Think of $r_{t+1}$ as the reward for having made action
$a_t$ in state $x_t$.

In more complicated situations, we would want a policy that does more than just
take a linear combination of the features the environment gives us.

# Learning the policy
Our aim now is to learn a good parameter vector $\theta$; that is, find $\theta$
that maximises our expected total reward. For any fixed policy $\pi$, we can
sample our Markov decision process, to get a trajectory $\tau = (x_0, a_0, r_1,
x_1, a_1, r_2, \ldots, x_{T-1}, a_{T-1}, r_T, x_T)$. We let $R_\tau \coloneqq
r_1 + r_2 + \cdots + r_T$ denote the total reward of the trajectory.

Then the expected total reward for a given policy $\pi$ is $\EE_\tau[R | \pi]$,
and our aim is to find $\pi$ that maximises this.

We describe two approaches here: cross-entropy, and policy gradient.

## The cross-entropy method
This is a Bayesian approach to finding $\theta$. We think of each entry
$\theta_i$ as a parameter to learn separately, and we fit a Gaussian
distribution over each one, with mean $\mu_i$ and variance $\sigma_i^2$, for $i
= 1,2,3,4$. Let $\mu = (\mu_1, \ldots, \mu_4)$ and $\sigma^2 = (\sigma_1^2,
\ldots, \sigma_4^2)$, and denote by $N(\mu, \sigma^2)$ the distribution over
$\RR^4$ such that $x \in \RR^4$ has $x_i \sim N(\mu_i, \sigma_i^2)$ for $i =
1,2,3,4$.

Let $N$ be a positive integer, which we call the sample size, and let $p \in (0,
1)$, which we call the elite fraction. Start with some initial values for $\mu$
and $\sigma^2$, say $\mu = (0,0,0,0), \sigma^2 = (1,1,1,1)$, and then repeat the
following process:

~~~~
for j from 1 to N
    theta(j) = sample_from_gaussian(mu, sigma2)
    reward(j) = estimate_reward_with_theta(theta(j))
elite_theta = best p x N theta
mu, sigma2 = maximum_likelihood_gaussian(elite_theta)
~~~~

Note that in the above 'theta(j)' denotes the $j^\th$ parameter vector $\theta$,
as opposed to an entry of a vector; and 'sigma2' denotes $\sigma^2$.

### Implementation details

The function sample_from_gaussian takes in $\mu \in \RR^4$ and $\sigma^2 \in
\RR^4$ and samples $x_i \sim N(\mu_i, \sigma_i^2)$ for $i = 1,2,3,4$. In python,
this can be achieved with:

~~~~python
import numpy as np
import random
mu = [0,0,0,0]
sigma2 = [1,1,1,1]
def sample_from_gaussian(mu, sigma2):
    return np.random.randn(1,4) * np.sqrt(sigma2) + mu
~~~~

By estimate_reward_with_theta, it is meant to estimate the expected total reward
if we follow policy $\pi(a | x; \theta)$. One simple way to estimate this is to
run the agent many times following the policy and use the mean total reward as
the estimate.

Then elite_theta is simply a list of the best $\theta$, ordered by their
estimated reward, where we only keep the top $100p\%$.

The final step in the loop is to fit the new Gaussian distributions. Here we
treat elite_theta as a list of samples, and compute the maximum likelihood
Gaussian distributions, which simply means the most likely Gaussian
distributions, given the sample data. It turns out that the new $\mu$ is just
the sample mean of the elite set, and the new $\sigma^2$ is just the sample
variance of the elite set.

Thus we can implement it as follows. Let elite_theta be an np array of shape $m
\times 4$, where $m$ is the number of samples. Then our new $\mu$ and $\sigma^2$
are given by

~~~~python
mu = np.mean(elite_theta,0)
sigma2 = np.var(elite_theta,0)
~~~~

Now we just repeat the process, sampling more $\theta$ from our distribution,
keeping the ones that score best, and fitting a new distribution to those.

This turns out to work remarkably well on the cart and pole problem, and doesn't
require much insight to get it to work. Formally, we are using a diagonal
multivariate Gaussian distribution, which is really assuming that the entries
$\theta_i$ of our parameter vector are pairwise independent. If there is
dependence in the parameter entries, we could use the more general multivariate
Gaussian distribution. This has the same mean, but the variance $\sigma^2$ is
replaced by a covariance matrix.

### A full implementation

That's the basic idea, and you can see the full implementation using OpenAI Gym
[here](https://github.com/cgnicholls/reinforcement-learning/blob/master/cartpole/crossentropy.py).

It turns out that the cross entropy method as I described it doesn't work very
well with the stochastic policy I described. This can be fixed by making the
policy deterministic: in the code, we actually use the policy 'move right' if $x
\theta^\tr > 0$, and otherwise 'move left'. Equivalently, 'move right' if
$\sigma(x \theta^\tr) > 0.5$, and otherwise 'move left'. Thus, instead of moving
right with probability $\sigma(x \theta^\tr)$, we just move right if the
probability is at least $0.5$.

## The policy gradient method ##
The cross-entropy method worked well for the cart and pole problem, but it
wouldn't generalise well to problems with more parameters, and a nonlinear
policy. Indeed, it didn't even work that well with a stochastic policy, and that
can be a crucial aspect of letting the agent explore the full space of policies.
In contrast, policy gradients are a class of methods that do generalise in this
way.

The basic idea is to use gradient ascent to optimise our policy to maximise the
expected total reward. At each step of the optimisation, we compute
$\grad_\theta \EE_\tau[R_\tau]$, and then update our parameter vector $\theta$
by a small amount, called the learning rate, $\alpha > 0$:

$$
\theta \leftarrow \theta + \alpha \grad_\theta \EE_\tau[R_\tau].
$$

If we gradually decrease the learning rate over time, then this will converge to
a local optimum. In practice, even though we may not reach the global optimum,
this general idea is effective at solving these kinds of optimisation problems.

### Computing the gradient ###
The question now is how to compute $\grad_\theta \EE_\tau[R_\tau]$.

First note that $\EE_\tau[R_\tau]$ is really an integral, and under nice
conditions, the gradient and integral are interchangeable. As before, if we fix
the policy $\pi$, then we can sample the Markov decision process following
policy $\pi$ to get trajectories $\tau = (x_0, a_0, r_1, x_1, \ldots, x_{T-1},
a_{T-1}, r_T, x_T)$.

Thus $\pi$ determines a probability distribution over trajectories $\tau$. Each
such trajectory has a total reward, $R_\tau$, which is also a random variable.
It thus makes sense to consider $\EE_\tau[R_\tau]$, which is formally given by
the following integral over all possible trajectories $\tau$:

$$
\EE_\tau[R_\tau | \pi; \theta] = \int_\tau P(\tau | \pi; \theta) R_\tau d\tau,
$$

We can now compute the derivative with respect to $\theta$:

$$
\grad_\theta \int_\tau P(\tau | \pi; \theta) R_\tau d\tau = \int_\tau
\grad_\theta P(\tau | \pi; \theta) R_\tau d \tau.
$$

We use a standard trick to write the integral in the form of an expectation. In
particular, we just need a factor of $P(\tau | \pi; \theta)$ in the integrand,
so we multiply and divide by this. Then the integrand is

$$
R_\tau  P(\tau | \pi; \theta) \frac{\grad_\theta P(\tau | \pi; \theta)}{P(\tau | \pi;
\theta)},
$$

which can be rewritten as

$$
R_\tau P(\tau | \pi; \theta) \grad_\theta \log P(\tau | \pi; \theta).
$$

Finally, we have
$$
\grad_\theta \EE_\tau[R_\tau | \pi; \theta] = \EE_\tau[R_\tau \grad_\theta \log
P(\tau | \pi; \theta)].
$$

This is crucial, because now we can use $R_\tau \grad_\theta \log P(\tau | \pi;
\theta)$ as an unbiased estimator of $\grad_\theta \EE_\tau[R_\tau | \pi;
\theta]$.

We now have to work out a way to write this just in terms of $\grad_\theta \pi(a
| x; \theta)$, which is something that we can compute.

So let $\tau = (x_0, a_0, r_1, x_1, \ldots, x_{T-1}, a_{T-1}, r_T, x_T)$ be a
trajectory for which we want to compute $P(\tau | \pi; \theta)$. We can rewrite
this as a product

$$
P(\tau | \pi; \theta) = P(x_0 | \pi; \theta) P(a_0 | x_0; \theta) P(r_1 | a_0,
x_0) P(x_1 | a_0, x_0) \cdots \\
P(a_{T-1} | x_{T-1}; \theta) P(r_T | a_{T-1}, x_{T-1}) P(x_T | a_{T-1}, x_{T-1}).
$$

where we have used the Markov property, so that we only have to condition on
the previous state.

The only terms that depend on $\theta$ are $P(a_t | x_t; \theta) = \pi(a_t |
x_t; \theta)$, and so on taking log and differentiating with respect to
$\theta$, we obtain the unbiased estimator:

$$
R_\tau \grad_\theta \log P(\tau | \pi; \theta) = (\sum_{t=1}^T r_t) \grad_\theta
\sum_{t=0}^{T-1} \log \pi(a_t | x_t; \theta).
$$

### Making this practical ###
From the previous discussion, given any trajectory $\tau$, sampled according to
policy $\pi(a | x; \theta)$, we can now compute $\hat{g}(\tau) \coloneqq R_\tau
\grad_\theta \log P(\tau | \pi; \theta)$. If we sample enough trajectories
$\tau$ and compute the mean of $\hat{g}(\tau)$, we will get a good estimate of
$\grad_\theta \EE_\tau[R_\tau | \pi; \theta]$. 

Let's try and implement this idea, and see if this vanilla version works on its
own (spoiler: unfortunately, it doesn't...)


