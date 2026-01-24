# general_FEP_RL
This package provides a configurable, customizable Reinforcement Learning (RL) agent implementing the Free Energy Principle (FEP).

This is based on the papers 

Intrinsic Rewards for Exploration without Harm from Observational Noise: A Simulation Study Based on the Free Energy Principle
https://arxiv.org/abs/2405.07473

Curiosity-Driven Development of Action and Language in Robots Through Self-Exploration
https://arxiv.org/abs/2510.05013

In summary, the agent uses a World Model (also known as a Forward Model) is a Recurrent Neural Network (RNN) which uses probabilistic prior and posterior inner states to predict future observations. This provides accuracy and complexity values, and trains to minimize Free Energy.

\begin{align} 
F_t = \underbrace{D_{KL}[q(z_t)||p(z_t)]}_{\text{Complexity}} - \underbrace{\mathbb{E}_{q(z_t)}[\log p(o_{t+1}|z_t)]}_{\text{Accuracy}}.
\end{align}

The hidden states are used as inputs for a Soft Actor Critic (SAC). The critic predicts a Q-value consisting of extrinsic rewards, entropy, curiosity, and imitation. The actor samples an action which maximizes predicted Q-value and entropy. This minimizes Expected Free Energy.

\begin{align}
Q_t &= r_t + \eta D_{KL}[q(z_{t}|o_{t},h_{t-1})||p(z_{t}|h_{t-1})] + \alpha \mathcal{H}(\pi_{\phi} (a_{t+1} | h_{t}))\nonumber \\ 
+ &\gamma (1 - done_t) \mathbb{E}_{o_{t+1} \sim D, a_{t+1} \sim \pi_\phi} [Q_{\bar{\theta}}(o_{t+1}, a_{t+1})].
\end{align}

Combined, the World Model and SAC are somewhat adversarial. The World Model tries to avoid surprise by understanding the relationship between observations, actions, and the environment. The SAC tseeks surprises to which the World Model must adapt.

To use this package for any RL setup, provide an encoder and decoder for each module of observation and action. These models require these variables:

Encoder:
    example_input
    example_output
    arg_dict:
        encode_size
        zp_zq_sizes

    This returns one value: the encoding of the input.

Decoder:
    example_input
    example_output
    loss_func

    This returns two values: the generated output and log-probabilities of those actions.