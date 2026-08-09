#%%
#------------------
# actor_critic.py provides a model for an actor (policy) and critic (Q-network).
#
# Both read the lowest world model layer's hidden state. That layer is the fastest,
# which suits reactive control, and it already carries the hierarchy's context: its
# hidden state is computed with the layer above it as an input, so the slow layers
# reach the actor through it rather than around it.
#------------------

from math import log

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from shape_to_shape_models import Shape_to_Shape_Model, Combinor, Divider



#------------------
# Tanh-squashed Gaussian, as SAC needs: bounded actions, and a log-probability that
# accounts for the squashing so motor entropy is measured on the action actually taken.
#------------------

class Action_Decoder(Shape_to_Shape_Model):

    def __init__(
            self,
            name,
            input_size,
            output_size,
            hidden_size = 32,
            min_std = 1e-2,
            verbose = False):

        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            arg_dict = {'hidden_size' : hidden_size, 'min_std' : min_std},
            verbose = verbose)

    def build_model(self, arg_dict):
        hidden_size = arg_dict.get('hidden_size', 32)
        self.min_std = arg_dict.get('min_std', 1e-2)

        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_shape[0], hidden_size),
            nn.LeakyReLU())
        self.mu = nn.Linear(hidden_size, self.output_shape[0])
        self.std = nn.Linear(hidden_size, self.output_shape[0])

    def forward(self, value):
        shared = self.shared_layers(value)
        mu = self.mu(shared)
        # min_std + softplus rather than a clamp, so there is no boundary where the
        # gradient dies and a unit can get stuck.
        std = self.min_std + F.softplus(self.std(shared))

        unsquashed = mu + std * torch.randn_like(std)
        action = torch.tanh(unsquashed)

        log_prob = torch.distributions.Normal(mu, std).log_prob(unsquashed)
        # The change-of-variables correction for tanh, written as
        # log(1 - tanh(x)^2) = 2*(log 2 - x - softplus(-2x)) to stay stable at large |x|.
        log_prob = log_prob - 2 * (log(2) - unsquashed - F.softplus(-2 * unsquashed))
        log_prob = log_prob.sum(dim = -1, keepdim = True)

        return {'action' : action, 'log_prob' : log_prob, 'mu' : mu, 'std' : std}

    @staticmethod
    def loss_func(predicted_values, target_values):
        return F.mse_loss(predicted_values, target_values, reduction = 'none')



#------------------
# Actor generates actions from World Model hidden states.
# a_t = \pi_\phi(h^q_t)
#------------------

class Actor(nn.Module):

    def __init__(
            self,
            hidden_state_size,
            dict_of_action_decoder_class_dicts,      # name -> {'class' : partial(...)}, one per part
                                                     # of the action ('move', 'make_voice', ...).
            verbose = False):

        super().__init__()

        self.hidden_state_size = hidden_state_size
        self.action_decoder = Divider(
            'action_decoder',
            [class_dict['class'](input_size = hidden_state_size)
             for class_dict in dict_of_action_decoder_class_dicts.values()],
            verbose = verbose)

    def forward(self, hidden_state, best_action_dict = None):
        outputs = self.action_decoder(hidden_state)

        action_dict = {name : output['action'] for name, output in outputs.items()}
        log_prob_dict = {name : output['log_prob'] for name, output in outputs.items()}

        if best_action_dict is None:
            return action_dict, log_prob_dict

        imitation_loss_dict = {}
        for name, model in self.action_decoder.models_dict.items():
            # (predicted, target), matching every other loss_func in this codebase.
            imitation_loss_dict[name] = model.loss_func(action_dict[name], best_action_dict[name])
        return action_dict, log_prob_dict, imitation_loss_dict

    def total_log_prob(self, log_prob_dict):
        # The parts of an action are independent given the hidden state, so their
        # log-probabilities add. This is what the motor entropy term needs.
        return sum(log_prob_dict.values())



#------------------
# Critic predicts Q-values.
# \widehat{Q}_t = Q_\theta(h^q_t, a_t)
#------------------

class Critic(nn.Module):

    def __init__(
            self,
            hidden_state_size,
            dict_of_action_encoder_class_dicts,      # name -> {'class' : partial(...)}, matching
                                                     # the actor's action parts.
            value_decoder = None,
            verbose = False):

        super().__init__()

        self.hidden_state_size = hidden_state_size
        self.action_encoder = Combinor(
            'action_encoder',
            [class_dict['class']() for class_dict in dict_of_action_encoder_class_dicts.values()],
            verbose = verbose)

        full_encoding_size = hidden_state_size + self.action_encoder.total_output_shape[-1]

        if value_decoder is not None:
            self.value_decoder = value_decoder
        else:
            self.value_decoder = nn.Sequential(
                nn.Linear(full_encoding_size, hidden_state_size),
                nn.PReLU(),
                nn.Linear(hidden_state_size, 1))

    def forward(self, hidden_state, action_dict):
        # The Combinor concatenates in sorted-name order and checks the dictionaries
        # match, so a missing or misnamed action part fails here rather than silently.
        encoded_action = self.action_encoder(action_dict)
        hidden_state_and_action = torch.cat([hidden_state, encoded_action], dim = -1)
        return self.value_decoder(hidden_state_and_action)



######################



if __name__ == '__main__':

    from functools import partial

    print("\n\n\n\n\n\n\n\n\n\n")

    class Vector_Encoder(Shape_to_Shape_Model):
        def __init__(self, name, input_size, output_size, verbose = False):
            super().__init__(name = name, input_shape = (input_size,),
                             output_shape = (output_size,), verbose = verbose)
        def build_model(self, arg_dict):
            self.model = nn.Sequential(
                nn.Linear(self.input_shape[0], 32), nn.LeakyReLU(),
                nn.Linear(32, self.output_shape[0]), nn.LeakyReLU())
        def forward(self, value):
            return self.model(value)

    class Vector_Decoder(Shape_to_Shape_Model):
        def __init__(self, name, input_size, output_size, verbose = False):
            super().__init__(name = name, input_shape = (input_size,),
                             output_shape = (output_size,), verbose = verbose)
        def build_model(self, arg_dict):
            self.model = nn.Sequential(
                nn.Linear(self.input_shape[0], 32), nn.LeakyReLU(),
                nn.Linear(32, self.output_shape[0]))
        def forward(self, value):
            return self.model(value)
        @staticmethod
        def loss_func(predicted_values, target_values):
            return F.mse_loss(predicted_values, target_values, reduction = 'none')

    ######################
    # An action with two parts, as intended.
    ######################

    move_size, voice_size = 2, 5
    hidden_state_size = 24

    dict_of_action_decoder_class_dicts = {
        'move' : {'class' : partial(Action_Decoder, name = 'move', output_size = move_size)},
        'make_voice' : {'class' : partial(Action_Decoder, name = 'make_voice', output_size = voice_size)}}

    dict_of_action_encoder_class_dicts = {
        'move' : {'class' : partial(Vector_Encoder, name = 'move',
                                    input_size = move_size, output_size = 16)},
        'make_voice' : {'class' : partial(Vector_Encoder, name = 'make_voice',
                                          input_size = voice_size, output_size = 16)}}

    actor = Actor(hidden_state_size, dict_of_action_decoder_class_dicts)
    critic_1 = Critic(hidden_state_size, dict_of_action_encoder_class_dicts)
    critic_2 = Critic(hidden_state_size, dict_of_action_encoder_class_dicts)

    print(f"actor parameters:  {sum(p.numel() for p in actor.parameters()):,}")
    print(f"critic parameters: {sum(p.numel() for p in critic_1.parameters()):,}\n")

    ######################
    # One step.
    ######################

    batch_size, episode_length = 4, 1
    hidden_state = torch.randn(batch_size, episode_length, hidden_state_size)

    action_dict, log_prob_dict = actor(hidden_state)
    for name in action_dict:
        print(f"action '{name}': \t{list(action_dict[name].shape)}\t"
              f"log_prob {list(log_prob_dict[name].shape)}")

    print(f"\nactions are bounded to (-1, 1): "
          f"{all(v.abs().max().item() < 1 for v in action_dict.values())}")

    total_log_prob = actor.total_log_prob(log_prob_dict)
    print(f"total log_prob: \t{list(total_log_prob.shape)}  "
          f"(motor entropy is its negative mean: {-total_log_prob.mean().item():.4f})")

    value_1 = critic_1(hidden_state, action_dict)
    value_2 = critic_2(hidden_state, action_dict)
    print(f"\nQ from each critic: \t{list(value_1.shape)}")
    print(f"SAC takes the smaller: \t{torch.min(value_1, value_2).mean().item():.4f}")

    ######################
    # Imitation, when the buffer has a best action to copy.
    ######################

    best_action_dict = {name : torch.tanh(torch.randn_like(value))
                        for name, value in action_dict.items()}
    action_dict, log_prob_dict, imitation_loss_dict = actor(hidden_state, best_action_dict)
    print("\nimitation loss per action part:")
    for name, loss in imitation_loss_dict.items():
        print(f"\t{name}: \t{list(loss.shape)}\tmean {loss.mean().item():.4f}")

    ######################
    # The Combinor catches a mismatched action.
    ######################

    print("\nmismatched action parts:")
    try:
        critic_1(hidden_state, {'move' : action_dict['move']})
    except ValueError as e:
        print(f"\traised ValueError: only in models_dict "
              f"{ {'make_voice'} }")