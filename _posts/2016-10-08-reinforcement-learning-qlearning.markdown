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
\def\PP{\mathbb{P}}
\def\coloneqq{\colon=}
\def\grad{\nabla}
\def\Qhat{\hat{Q}}
\def\argmax{\mathrm{argmax}}
$

## Q-learning ##

In this post we implement the Q-learning algorithm on the Cart and Pole problem. I discussed the Cart and Pole problem and gave a brief introduction to reinforcement learning in [this post](https://cgnicholls.github.io/reinforcement-learning/2016/08/20/reinforcement-learning.html), so it might be worthwhile checking out at least part of that first.

In reinforcement learning, we think of an agent interacting with an environment by observing the environment and choosing actions. At time $t$ the agent receives a state $x_t$ and chooses an action $a_t$. The environment then sends the agent a reward $r_{t+1}$ for having taken that action in that state. In the Cart and Pole problem, the agent has to try and balance a pole on top of a cart. The state at time $t$ consists of the position and velocity of the cart, together with the angle and angular velocity of the pole.

Visually, the Cart and Pole problem looks like this on the OpenAI Gym:

![Cart Pole]({{ site.url }}/assets/CartPole.jpg)

In fact the agent doesn't see this picture and instead just receives the positions and velocities described above. It would actually be a much harder problem to solve if the agent only received a picture of the cart and pole, rather than the positions and velocities. Thinking just in terms of dimensions, in the current formulation there are 4, while in the picture formulation there would be $w \times h \times 3$ dimensions just to represent one picture, where the image has width $w$ and height $h$. That's about 4 orders of magnitude larger! However, in my post on the [A3C algorithm](https://cgnicholls.github.io/reinforcement-learning/2017/03/27/a3c.html), we will apply a state-of-the-art algorithm called A3C to Space Invaders, learning just from the pixels.

Bear in mind though that even in the Cart and Pole problem, where there are just four dimensions, the state space is still infinite, since each of these dimensions is continuous.

## Markov Decision Processes ##
We will model this whole situation (environment and agent) as a *Markov Decision Process*. Formally, a Markov Decision Process is a system with a set of states, $\Sigma$, a set of possible actions, $A$, a transition function $\mathcal{P} \colon \Sigma \times A \to \Sigma$ and a reward function $R \colon \Sigma \times A \times \Sigma \to \RR$. In a given state $x \in \Sigma$ the agent can choose an available action $a \in A$. The system then transitions into a new state $x'$ according to the probability distribution $\mathcal{P}(x, a, x')$ and provides the agent with a reward $R(x, a, x')$.

This can go on indefinitely, giving rise to a sequence $x_0, a_0, r_1, x_1, a_1, r_2, \ldots$, where $x_0$ is the initial state, $a_0$ is the action the agent chooses, and $r_1$ is the reward $R(x_0, a_0, x_1)$ given by the MDP when it transitions into state $x_1$, and so on.

The system is called *Markov* because it satisfies the *Markov property*: the probability of the next state given just the current state and action equal the probability of the next state given the *entire history* of states and actions; formally,

$$
\PP(x_{t+1} \mid x_0, a_0, \ldots, x_t, a_t) = \PP(x_{t+1} \mid x_t, a_t).
$$

We use this as our mathematical model throughout this post.

## Policies ##
A *policy* is a specification of what action to take for a given state. One can either think of deterministic policies, which specify for each state exactly which action to take, or, more generally, stochastic policies. In a stochastic policy, for a given state $s$, we define a probability distribution over the possible actions to take. If we use the notation $\mathcal{D}(A)$ to denote the possible probability distributions over a set $A$, then a policy is a map

$$
\pi \colon \Sigma \to \mathcal{D}(A)
$$

where $\Sigma$ denotes the set of states and $A$ denotes the set of actions. It can be that the possible actions depend on what state you are in (for example, sometimes an action isn't possible), but we ignore this technicality here.

# Example #
Consider the game of Blackjack (in fact a simplified version of Blackjack, because the full rules would take a while to write down properly). The object is to get closer to 21 than the dealer (without going over) by adding the values of the cards you are dealt. The cards 2-10 are worth their face value, while Jacks, Queens, Kings are worth 10 and Aces are worth either 1 or 11 (you choose). Each player is initially dealt two cards. The states in this game are the cards that everyone has been dealt so far this round (this is assuming we don't bother to try and keep track of the deck -- so no counting cards!). On your turn you have two possible actions: hit and stand. Hit draws you another card from the deck and stand ends your turn. A policy specifies, for each combination of cards, the probability that you will hit and the probability you will stand. If you hit, then you are in a new state (including the card you were just dealt) and you again act by your policy, and so on.

A very simple policy might only depend on your cards (i.e. ignoring everyone else's cards). You might have something like: always hit if your current total is at most 10, always stand if your current total is greater than 16. In the total is somewhere in the middle you might hit with higher probability closer to 10, but still sometimes stand. The important thing is that the policy specifies exactly with what probabilities to take the given actions in a given state.

The reward in proper Blackjack depends on the amount you bet. If you beat the dealer's hand then you win the amount you bet (i.e. you get your bet back and also win that amount), and otherwise you lose it. However, in a reinforcement learning setting, one might first want to master the situation where you get 1 point for winning the hand and lose 1 point if you lose the hand.

## Trajectories ##
A *trajectory* is a sequence of states, actions and rewards as sampled from the given MDP: $x_0, a_0, r_1, \ldots, x_T, a_T, r_{T+1}, \ldots$ (it can be finite or infinite). A fixed policy defines a random process for generating trajectories. That is, if we commit to playing with a certain policy, then at each time step $t$, if we see the state $x_t$, we will choose the action $a_t$ according to the distribution $\pi(x_t)$. The system will then receive this action and return us a reward, $r_{t+1}$, and new state, $x_{t+1}$. Continuing this gives us a trajectory $\tau = (x_0, a_0, r_1, x_1, a_1, r_2, \ldots)$.

Note that we write $a \sim \pi(x)$ to mean that $a$ is sampled according to the distribution $\pi(x)$.

## Value functions ##

We want to be able to compare trajectories so that we can try and change the policy to make more favourable trajectories come up more often. What should more favourable mean in this context? The first idea might be to just add up the rewards in a trajectory and call that the value of the trajectory. This works well in episodic settings (where there are terminal states, which end the episode), since this sum will always be finite. However, assuming that trajectories can be arbitrarily long, it is common to *discount* the reward for a trajectory; that is, use $\sum_{i=0}^\infty \gamma^t r_{t+1}$ for some fixed $0 < \gamma < 1$, called the *discount factor*. Assuming that rewards $r_t$ are bounded, this will then always be finite, which means that trajectories can be compared easily.

Starting from a given state and following a given policy, we have determined a probability distribution over entire trajectories. It thus makes sense to speak of the expected value of a trajectory when following policy $\pi$ and starting from state $x_0$. We call this the *value function*

$$V^{\pi} \colon \Sigma \to \RR \\
V^{\pi}(x) = \EE_\tau[\sum_{t=0}^\infty \gamma^t r_{t+1} \mid x_0 = x, a_t \sim \pi(x_t)].
$$

Formally, what is meant here is the expectation of $\sum_{t=0}^T \gamma^t r_{t+1}$ over entire trajectories. Thus it's a sum $\sum_{\tau=x_0, a_0, r_1, \ldots} P(\tau \mid \pi) \sum_{t=0}^\infty \gamma^t r_{t+1}$ over all possible trajectories $\tau$. Here $P(\tau \mid \pi)$ is the probability of trajectory $\tau$ given that we follow policy $\pi$. Note that this is only really helpful as a mathematical definition as for most of the problems we consider it would be intractable to actually compute quantities like $P(\tau \mid \pi)$ explicitly.

# State-action value function (or the Q-function)
Another important value function is called the *state-action value function* or the *Q-function*. This is defined as the expected value for an agent following a policy $\pi$ given that they start in state $x$ and take action $a$ in that state, and then follow policy $\pi$ from then on. Formally,

$$
Q^{\pi} \colon \Sigma \times A \to \RR \\
Q^{\pi}(x, a) = \EE_\tau[\sum_{t=0}^\infty \gamma^t r_{t+1} \mid x_0 = x, a_0 = a, a_t \sim \pi(x_t) \textrm{ if } t \ge 1]
$$

One can define the optimal $Q$-function, denoted $Q^*$, as the maximum expected value over all policies $\pi$ when starting in state $x$ and first taking action $a$. That is,

$$
Q^* \colon \Sigma \times A \to \RR \\
Q^*(x, a) = \max_\pi Q^\pi(x, a).
$$

Similarly, we define the optimal value function $V^*$ as the maximum expected reward over all policies $\pi$ when starting in state $x$

$$
V^* \colon \Sigma \to \RR \\
V^* (x) = \max_\pi V^\pi(x).
$$

Intuitively, the optimal value function tells us the highest value we could possibly achieve by using any policy when starting in state $x$.

Note that actually knowing $Q^* $ is more immediately helpful than knowing $V^* $, since if we know $Q^* $ then we can recover an optimal policy as follows:

$$
\pi^* \colon \Sigma \to A \\
\pi^*(x) = \argmax_a Q^*(x, a).
$$

That is, in state $x$ we choose the action with highest $Q^* $-value. This makes sense because if we choose that action, our expected value from the next state onwards is maximised. This is encapsulated in the Bellman-equation

$$
Q^*(x, a) = \EE_{x' \sim \mathcal{P}}[r + \gamma \max_{a'} Q^*(x', a') | x, a].
$$

In words, the Bellman equation says that the $Q$-value of being in state $s$ and
choosing action $a$ is the expected reward of being in state $s$ and choosing
action $a$ plus the expected maximum $Q$-value over all actions in the next
state.

Note that the policy derived from $Q^* $ is actually deterministic (i.e. the distribution over actions in a given state is just to choose one action with probability 1). This makes sense because in any Markov Decision Process there is actually a deterministic optimal policy.

# How to use $Q$-learning to improve a policy #
OK, so if we know the optimal $Q$-function then we're done! But how do we get this?

<!--
There is a simple relationship between $Q$ and $V$ given by

$$
Q^\pi(x, a) = \EE[r_1 \mid x_0 = x, a_0 = a] + \gamma V^\pi(x_1).
$$
-->

Most approaches for learning $Q^* $ in reinforcement learning involve some kind of optimisation process where you start with an initial estimate of $Q^* $ and gradually improve it.

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

## Approximate $Q$-functions

We are led to consider approximate $Q$-functions, where we try to approximate
the optimal $Q$-function by some parametrised function $\Qhat(s, a; \theta)$.
Then our task is to learn the best parameter vector $\theta$.

<!--
# Example #
In the Cart and Pole problem the state is a 4-dimensional vector $s = (s_1, s_2, s_3, s_4)$ (for the position and velocity of the cart, and angle and angular velocity of the pole). The possible actions are $a = L, R$ (accelerate left and right). A very simple parametrised function we could use is a linear approximation:

$$
Q(s, L) = \theta_{11} s_1 + \theta_{12} s_2 + \theta_{13} s_3 + \theta_{14} s_4 \\
Q(s, R) = \theta_{21} s_1 + \theta_{22} s_2 + \theta_{23} s_3 + \theta_{24} s_4.
$$

We can represent this more succinctly as a matrix multiplication:

$$
Q(s, a; \theta) = \theta s,
$$

where $\theta$ is a $2 \times 4$ matrix and $s$ is a 4-dimensional column vector. The output is a 2-dimensional column vector where the first entry is the $Q$-value for $a = L$ and the second entry is that for $a = R$. This is a function parametrised by the matrix $\theta$, and our task is just to find good values of $\theta$. Note that if the form of the parametrised function isn't general enough then we won't be able to represent our function well enough. If the function is too general then it might be prone to overfitting (finding spurious relationships in the data).
-->

In this post we use neural networks to approximate the $Q$-function. Start with a randomly initialised neural network, denoted $\Qhat(s, a; \theta)$, and then try to find $\theta$ such that $\Qhat$ satisfies the Bellman equation. Sample the environment and collect lots of transitions $(s, a, r, s')$, where $s$ is some state, $a$ is the action we choose in that state, $r$ is the reward we get from the environment, and $s'$ is the state the environment transitions us to.

We use the Bellman equation to get the $Q$-function to converge. Explicitly, we minimise the loss

$$
L = [\Qhat(s, a) - (r + \max_{a'} \Qhat(s', a'))]^2,
$$

averaged over many transitions. Intuitively, the loss wants to minimise the difference between the $Q$-value we predicted, $\Qhat(s, a)$, and the reward that we saw after one step plus the $Q$-value we predicted in the next state. Minimising this corresponds to solving the Bellman equation, and should give us the optimal $Q$-function.

# Check that a linear $Q$-function works for our problem
It's usually a good idea to try out the simplest to implement, and quickest to
run, methods first, and only move on to more complicated methods when these have
been shown insufficient. It turns out that computing a $Q$-function for the Cart
and Pole problem whose associated policy performs well can be solved using the
cross-entropy method, rather than gradient descent.

Note that this doesn't actually show that the Bellman equation is satisfied for this $\Qhat$; rather, the policy $\pi(s) = \mathrm{argmax}_a \Qhat(s, a; \theta)$ performs very well on the Cart and Pole problem.

We have to choose the architecture of the neural network. The simplest idea is
to use a linear network:

$$
\Qhat(s; \theta) = s \theta,
$$

where $s$ is a $1 \times 4$ vector and $\theta$ is a $4 \times 2$ matrix. You'll
notice that $\Qhat$ does not mention the action, which is because the output of
$\Qhat(s; \theta)$ is a $1 \times 2$-vector; the $\Qhat$ value of moving left is
the first value and the $\Qhat$ value of moving right is the second value. This
means we can use just one network, rather than a separate one for each action.

Using the cross entropy method, we can find $\theta$ such that the associated
policy performs very well. You can see the code for the cross entropy method
applied to $Q$-learning
[here](https://github.com/cgnicholls/reinforcement-learning/blob/master/cartpole/cartpole-qlearning-crossentropy.py).

Since we are interested in using deep $Q$-learning, and generalising to more complicated problems, we do actually still implement $Q$-learning, and check that it works in the Cart and Pole problem.

If you include a bias term in $\Qhat$ then you get much better convergence. In
tensorflow, with a linear network (plus bias), it's possible to get quite close
to a solution to the Bellman equation. However, the average reward doesn't seem
to be so stable.

# The exploration-exploitation tradeoff
A fundamental problem in reinforcement learning is the *exploration-exploitation tradeoff*. That is, how should you balance exploiting a strategy that you already know to work versus exploring new strategies that might work better. There are various methods of encouraging exploration in reinforcement learning, but the one that is commonly used with $Q$-learning is called $\epsilon$-greedy.

In an $\epsilon$-greedy strategy, one plays a mixture of two strategies. One of the strategies is to play uniformly at random over all possible actions. The other strategy is some greedy strategy, for example, playing according to the current approximation of the $Q$-function. The greedy strategy is the exploitation part, since we hope that it will work reasonably. At the start, however, we want to explore many possible strategies, which is what playing randomly gives us.

One usually chooses a probability $0 \le \epsilon \le 1$ with which to play the random strategy and with probability $1-\epsilon$ one plays the greedy strategy. It is common to shrink $\epsilon$ to zero (or very close to zero) over a large number of iterations. Though exactly how one does this is a hyperparameter to tune!

# $Q$-learning on the cart and pole problem

The following code implements $Q$-learning on the cart and pole problem.

First we import all the required libraries. We use numpy for general mathematical operations, gym for the game environment and tensorflow for the neural network.

~~~~python
import numpy as np
import gym
import tensorflow as tf
from random import sample, random
from collections import deque
~~~~

The following code creates the network that we use to approximate the $Q$-function.

We use xavier initialisation for the weights and initialise the biases to be zero. Xavier initialisation is explained in my post on the A3C algorithm, but basically ensures that if we send in uniformly distributed input to a layer then it will still be uniformly distributed after it goes through the layer. We also regularise the weights with L2 regularisation, which is one method of preventing overfitting. I experimented with dropout as well, but I don't think it's actually that effective here since the network is very small.

~~~~python
# Create a fully connected layer.
def fully_connected(inputs, num_outputs, activation_fn, l2_reg=0.1):
    return tf.contrib.layers.fully_connected(inputs=inputs,
    num_outputs=num_outputs, activation_fn=activation_fn,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    biases_initializer=tf.zeros_initializer(),
    weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

# Create a network with input_dim neurons in the input and output_dim
# neurons in the output. The intermediate layers all have relu activations.
def create_network(input_dim, hidden_dims, output_dim, scope):
    with tf.variable_scope(scope):
        input_layer = tf.placeholder('float32', [None, input_dim])

        hidden_layer = input_layer
        for dim in hidden_dims:
            hidden_layer = fully_connected(inputs=hidden_layer,
            num_outputs=dim, activation_fn=tf.nn.relu)
            # Use dropout if you want.
            #hidden_layer = tf.nn.dropout(hidden_layer, 0.5)

        output_layer = fully_connected(inputs=hidden_layer,
        num_outputs=output_dim, activation_fn=None)
        return input_layer, output_layer
~~~~

In the standard $Q$-learning algorithm one just uses a single function approximator for the $Q$-function, but it is now standard to use two functions: one that we update as usual with the $Q$-loss and one that we hold fixed for some number of steps at a time. This second network is called the target network, and means that we are actually trying to minimise

$$
L = [\Qhat(s, a) - \max_{a'} \Qhat^{\textrm{target}}(s', a') - r]^2,
$$

where $\Qhat$ is the usual network and $\Qhat^{\textrm{target}}$ is the target network. At regular intervals we update the target network to equal the $Q$-network. The intuition is that by fixing the target $Q$-values we make it easier for the training to converge.

The following code updates the weights in one network to be the weights of the other. This is achieved using scopes in tensorflow. So when we create the `current` and `target` networks we just give them each a scope and then it's easy to update the variables.

~~~~python
# Update the weights in to_scope to the ones from from_scope.
def update_ops_from_to(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=to_scope)

    ops = []
    for from_var, to_var in zip(from_vars, to_vars):
        ops.append(tf.assign(to_var, from_var))
    return ops
~~~~

Next we define the main $Q$-learning function. We create two networks: the `current` network and the `target` network, as well as creating the operations in tensorflow to update the `target` network to equal the `current` network.

We then define the loss function that we want to minimise. Suppose we are given a transition $(s, a, r, s', b)$, where $s$ is the current state, $a$ is the action we took, $r$ is the reward we received, $s'$ is the state we transitioned to, and $b$ is a boolean denoting whether or not the state was terminal. Then the loss for this transition is defined as

$$
L(s, a, r, s', b) =
\begin{cases}
(Q(s, a) - r - \gamma \max_{a'} Q^\textrm{target}(s', a'))^2, & \textrm{ if } b = 0 \\
(Q(s, a) - r)^2, & \textrm{ if } b = 1.
\end{cases}
$$

In the following, we pass in the state, action and reward for a given transition. But instead of also passing in the next state and computing the final term in the loss function, we compute this separately and pass that in as `tf_next_q`. This ensures that we don't train the target network also when minimising the loss (we want to keep that fixed until we update it). Also it is slightly easier to program this way.

To compute the $Q$-value for $s, a$ we actually compute the $Q$-values for $s$ (i.e. for all $a$) and then just take the dot product with the one-hot vector that has a one in the index corresponding to $a$. This ensures the computation is vectorised. Finally, we just compute the loss and tell tensorflow to minimise it using the Adam optimiser.

~~~~python
def qlearning(env, input_dim, num_actions, max_episodes=100000,
    update_target_every=2000, min_transitions=10000, batch_size=128,
    discount=0.9):
    transitions = deque()

    # Create the current network as well as the target network. The dimensions
    # given in hidden_dims define the sizes of the hidden layers (if any).
    hidden_dims = []
    input_layer, output_layer = create_network(input_dim, hidden_dims,
        num_actions, 'current')
    target_input_layer, target_output_layer = create_network(input_dim,
        hidden_dims, num_actions, 'target')
    update_ops = update_ops_from_to('current', 'target')

    tf_q_values = output_layer
    tf_action = tf.placeholder('int32', [None])
    tf_one_hot_action = tf.one_hot(tf_action, num_actions)
    tf_q_for_action = tf.reduce_sum(tf.multiply(tf_one_hot_action,
        tf_q_values), reduction_indices=1)
    tf_next_q = tf.placeholder('float32', [None])
    tf_reward = tf.placeholder('float32', [None])
    tf_loss = tf.reduce_mean(tf.square(tf_q_for_action - tf_reward - tf_next_q))

    tf_train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(tf_loss)
~~~~

Next we set up the session and do some book-keeping for tracking the loss and average reward. Then we have the main for-loop over all episodes. For each episode, we reset the environment and store the first state, using `state = env.reset()`, and start the loop for this episode.

We then follow the $\epsilon$-greedy policy with the current $Q$-function. For each step, with probability $\epsilon$ we choose an action randomly. With probability $1-\epsilon$ we actually compute the $Q$-value for the current state and choose the action with highest $Q$-value. We store all the transitions we see in the `transitions` deque (double ended queue) and also keep track of rewards.

~~~~python
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sess.run(update_ops)

    all_rewards = []
    epsilon = 1.0

    for ep in xrange(max_episodes):
        state = env.reset()
        terminal = False

        # Gradually anneal epsilon to 0.01.
        epsilon -= 0.0001
        if epsilon <= 0.01:
            epsilon = 0.01

        t = 0
        while not terminal:
            # Given the state, we predict the action to be the one with largest
            # q-value
            q_values = sess.run(output_layer, feed_dict={
                input_layer: [state]
            })
            action = np.argmax(q_values)
            if random() < epsilon:
                action = np.random.choice(num_actions)

            next_state, reward, terminal, _ = env.step(action)
            transition = {'state': state,
                        'action': action,
                        'next_state': next_state,
                        'reward': reward,
                        'terminal': terminal}

            transitions.append(transition)
            state = next_state
            t += 1
            if t > 1000:
                print "Maximum episode length"
                break
        all_rewards.append(t)
~~~~

Still inside the main (over episodes) for-loop, we now get to the training part. We don't train until we've collected a good few transitions though (another hyperparameter to tune!). Once we have, we sample `batch_size` of the transitions and compute the `next_qs` for each one. For this, we just run the target network with the `next_states`, take the maximum over the actions (for each one), and then scale with the discount factor and whether or not the frame was terminal.

~~~~python
        # Only train if we have enough transitions
        if len(transitions) >= min_transitions:
            samples = sample(transitions, batch_size)
            states = [d['state'] for d in samples]
            next_states = [d['next_state'] for d in samples]
            rewards = [d['reward'] for d in samples]
            actions = [d['action'] for d in samples]
            not_terminals = np.array([not d['terminal'] for d in samples],
                'float32')

            next_qs = sess.run(target_output_layer, feed_dict={
                target_input_layer: next_states
            })
            max_next_qs = np.amax(next_qs, axis=1)

            target_qs = discount * max_next_qs * not_terminals

            _, loss = sess.run([tf_train_op, tf_loss], feed_dict={
                input_layer: states,
                tf_reward: rewards,
                tf_next_q: target_qs,
                tf_action: actions
            })
~~~~

Print out the loss after every 100 episodes and update the target network every `update_target_every` episodes.

~~~~python
            if ep % 100 == 0:
                print "Loss:", loss
                print "Average ep length", np.mean(all_rewards)
                print "Epsilon", epsilon
                all_rewards = []

            if ep % update_target_every == 0:
                print "Updating target"
                sess.run(update_ops)
~~~~

Finally, we $Q$-learn!

~~~~python
qlearning(gym.make('CartPole-v0'), 4, 2)
~~~~
