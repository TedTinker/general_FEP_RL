#%%
#------------------
# actor_critic.py provides a model for an soft-actor (policy) and critic (Q-network).
#
# Both read the lowest world model layer's hidden state. 
# The critic also reads the action.
# The actor may be given a "best action" for imitation.
#------------------

from math import log
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from general_FEP_RL.shape_to_shape_models import Shape_to_Shape_Model, Combiner, Divider



# Part of a soft actor, making an action out of a probability distribution. 
# Returns action, entropy, mean, and standard deviation.
class Action_Decoder(Shape_to_Shape_Model):

    def __init__(
            self,
            name,               # String. Should be unique.
            input_size,         # Size of hidden_state.
            output_size,        # Size of action.
            hidden_size = 32,   # One linear layer shared by mu and std.
            verbose = False):

        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            arg_dict = {'hidden_size' : hidden_size},
            verbose = verbose)

    def build_model(self, arg_dict):
        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features = self.input_shape[0], 
                out_features = arg_dict['hidden_size']),
            nn.LeakyReLU())
        
        self.mu = nn.Linear(
            in_features = arg_dict['hidden_size'], 
            out_features =self.output_shape[0])
        
        self.std = nn.Linear(
            in_features = arg_dict['hidden_size'], 
            out_features = self.output_shape[0])

    def forward(self, value):
        shared = self.shared_layers(value)
        mu = self.mu(shared)
        std = 1e-2 + F.softplus(self.std(shared))

        unsquashed = mu + std * torch.randn_like(std)
        action = torch.tanh(unsquashed)

        log_prob = torch.distributions.Normal(mu, std).log_prob(unsquashed)
        log_prob = log_prob - 2 * (log(2) - unsquashed - F.softplus(-2 * unsquashed))
        log_prob = log_prob.sum(dim = -1, keepdim = True)

        return {'action' : action, 'log_prob' : log_prob, 'mu' : mu, 'std' : std}

    @staticmethod
    def loss_func(predicted_values, target_values):
        return F.mse_loss(predicted_values, target_values, reduction = 'none')



# Soft actor makes all actions. 
class Actor(nn.Module):

    def __init__(
            self,
            hidden_state_size,                      # Size of the world_model's lowest layer's hidden state.
            dict_of_action_decoder_class_dicts,     # name -> {'class' : partial(...)}, one per part of the action.
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
            imitation_loss_dict[name] = model.loss_func(action_dict[name], best_action_dict[name])
        return action_dict, log_prob_dict, imitation_loss_dict

    # Find complete entropy values.
    def total_log_prob(self, log_prob_dict):
        return sum(log_prob_dict.values())



# Critic predicting Q-value.
class Critic(nn.Module):

    def __init__(
            self,
            hidden_state_size,                          # Size of the world_model's lowest layer's hidden state.
            dict_of_action_encoder_class_dicts,         # name -> {'class' : partial(...)}, matching the actor's action parts.
            value_decoder = None,                       # If you have another model you want to use, put it here.
            verbose = False):

        super().__init__()

        self.hidden_state_size = hidden_state_size
        self.action_encoder = Combiner(
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
        encoded_action = self.action_encoder(action_dict)
        hidden_state_and_action = torch.cat([hidden_state, encoded_action], dim = -1)
        return self.value_decoder(hidden_state_and_action)



# Examples.
######################
if __name__ == '__main__':

    

    print("\n\n\n\n\n\n\n\n\n\n")
    
    

    # Two models just to make example actions.
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



    # An actor and critic, with actions having two parts. 
    move_size, voice_size = 2, 5
    hidden_state_size = 24

    dict_of_action_decoder_class_dicts = {
        'move' : {'class' : partial(Action_Decoder, name = 'move', output_size = move_size)},
        'make_voice' : {'class' : partial(Action_Decoder, name = 'make_voice', output_size = voice_size)}}

    dict_of_action_encoder_class_dicts = {
        'move' : {'class' : partial(Vector_Encoder, name = 'move', input_size = move_size, output_size = 16)},
        'make_voice' : {'class' : partial(Vector_Encoder, name = 'make_voice', input_size = voice_size, output_size = 16)}}

    actor = Actor(hidden_state_size, dict_of_action_decoder_class_dicts)
    critic_1 = Critic(hidden_state_size, dict_of_action_encoder_class_dicts)
    critic_2 = Critic(hidden_state_size, dict_of_action_encoder_class_dicts)

    print(f"actor parameters:  {sum(p.numel() for p in actor.parameters()):,}")
    print(f"critic parameters: {sum(p.numel() for p in critic_1.parameters()):,}\n")

    

    # One step.
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



    # Imitation, when the buffer has a best action to copy.
    best_action_dict = {name : torch.tanh(torch.randn_like(value))
                        for name, value in action_dict.items()}
    action_dict, log_prob_dict, imitation_loss_dict = actor(hidden_state, best_action_dict)
    print("\nimitation loss per action part:")
    for name, loss in imitation_loss_dict.items():
        print(f"\t{name}: \t{list(loss.shape)}\tmean {loss.mean().item():.4f}")
