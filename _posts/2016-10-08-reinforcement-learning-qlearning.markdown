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
Q^*(s, a) = \EE_{s'}[R(s,a,s') + \gamma \max_{a'} Q^*(s, a')].
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
been shown insufficient. It turns out that computing a $Q$-function for the Cart
and Pole problem whose associated policy performs well can be solved using the
cross-entropy method, rather than gradient descent.

Note that this doesn't actually show that the Bellman equation is satisfied for
this $\Qhat$; rather, the policy $\pi(s) = \mathrm{argmax}_a \Qhat(s, a; \theta)$
performs very well on the Cart and Pole problem.

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

Since we are interested in using deep q-learning, we do want to use the more
complicated method as well, and check that it works in the Cart and Pole
problem.

If you include a bias term in $\Qhat$ then you get much better convergence. In
tensorflow, with a linear network (plus bias), it's possible to get quite close
to a solution to the Bellman equation. However, the average reward doesn't seem
to be so stable.

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
    return tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=num_outputs, activation_fn=activation_fn, weights_initializer=tf.contrib.layers.xavier_initializer(),
    biases_initializer=tf.zeros_initializer(),
    weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

# Create a network with input_dim neurons in the input and output_dim neurons in the output. The intermediate layers all have relu activations.
def create_network(input_dim, hidden_dims, output_dim, scope):
    with tf.variable_scope(scope):
        input_layer = tf.placeholder('float32', [None, input_dim])

        hidden_layer = input_layer
        for dim in hidden_dims:
            hidden_layer = fully_connected(inputs=hidden_layer, num_outputs=dim, activation_fn=tf.nn.relu)
            # Use dropout if you want.
            #hidden_layer = tf.nn.dropout(hidden_layer, 0.5)

        output_layer = fully_connected(inputs=hidden_layer, num_outputs=output_dim, activation_fn=None)
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
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)

    ops = []
    for from_var, to_var in zip(from_vars, to_vars):
        ops.append(tf.assign(to_var, from_var))
    return ops
~~~~

Next we define the main $Q$-learning function. We create two networks: the `current` network and the `target` network, as well as creating the operations in tensorflow to update the `target` network to equal the `current` network.

We then define the loss function that we want to minimise. Suppose we are given a transition $(s, a, r, s', \tau)$, where $s$ is the current state, $a$ is the action we took, $r$ is the reward we received, $s'$ is the state we transitioned to, and $\tau$ is a boolean denoting whether or not the state was terminal. Then the loss for this transition is defined as

$$
L(s, a, r, s', \tau) = (Q(s, a) - r - \gamma \mathbb{1}_{\tau=0} \max_{a'} Q^\textrm{target}(s', a'))^2.
$$

Here I'm using the notation $\mathbb{1}_{\tau=0}$ to mean the constant 1 if $\tau$ is false and otherwise 0. In the following, we pass in the state, action and reward for a given transition. But instead of also passing in the next state and computing the final term in the loss function, we compute this separately and pass that in as `tf_next_q`. This ensures that we don't train the target network also when minimising the loss (we want to keep that fixed until we update it). Also it is slightly easier to program this way.

To compute the $Q$-value for $s, a$ we actually compute the $Q$-values for $s$ (i.e. for all $a$) and then just take the dot product with the one-hot vector that has a one in the index corresponding to $a$. This ensures the computation is vectorised. Finally, we just compute the loss and tell tensorflow to minimise it using the Adam optimiser.

~~~~python
def qlearning(env, input_dim, num_actions, max_episodes=100000, update_target_every=2000, min_transitions=10000, batch_size=128, discount=0.9):
    transitions = deque()

    # Create the current network as well as the target network. The dimensions given in hidden_dims define the sizes of the hidden layers (if any).
    hidden_dims = []
    input_layer, output_layer = create_network(input_dim, hidden_dims, num_actions, 'current')
    target_input_layer, target_output_layer = create_network(input_dim, hidden_dims, num_actions, 'target')
    update_ops = update_ops_from_to('current', 'target')

    tf_q_values = output_layer
    tf_action = tf.placeholder('int32', [None])
    tf_one_hot_action = tf.one_hot(tf_action, num_actions)
    tf_q_for_action = tf.reduce_sum(tf.multiply(tf_one_hot_action, tf_q_values), reduction_indices=1)
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
            # Given the state, we predict the action to be the one with largest q-value
            q_values = sess.run(output_layer, feed_dict={
                input_layer: [state]
            })
            action = np.argmax(q_values)
            if random() < epsilon:
                action = np.random.choice(num_actions)

            next_state, reward, terminal, _ = env.step(action)
            transition = {'state': state, 'action': action, 'next_state': next_state, 'reward': reward, 'terminal': terminal}

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
            not_terminals = np.array([not d['terminal'] for d in samples], 'float32')

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
