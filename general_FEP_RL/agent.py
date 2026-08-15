#%%
#------------------
# agent.py provides a class combining the world model, actor, and critics.
#------------------

from copy import deepcopy

import torch
from torch import nn
import torch.optim as optim

from general_FEP_RL.buffer import Recurrent_Replay_Buffer, shapes_from_world_model
from general_FEP_RL.world_model import World_Model, make_world_model
from general_FEP_RL.actor_critic import Actor, Critic
from general_FEP_RL.agent_methods import Agent_Methods, GENERATED_INPUT_NAMES



#------------------
# Training scalars for inner states:
#   upsilon_prior       accuracy scalar for prediction
#   upsilon_posterior   accuracy scalar for reconstruction
#   beta                complexity scalar
#   eta_before_clamp    curiosity scalar, applied BEFORE clamping to [0, 1]
#   eta                 curiosity scalar, applied after
#------------------

DEFAULT_INNER_STATE_SCALARS = {
    'upsilon_prior' : 1.0,
    'upsilon_posterior' : 1.0,
    'beta' : 0.03,
    'eta_before_clamp' : 1.0,
    'eta' : 1.0}

#------------------
# Training scalars for actions:
#   target_entropy      entropy desired for alpha's loss
#   alpha_normal        keep actions near baseline
#   initial_alpha       initial value of alpha
#   lr_alpha            learning rate of alpha
#   delta               imitation scalar
#------------------

DEFAULT_ACTION_SCALARS = {
    'target_entropy' : -1.0,
    'alpha_normal' : 1.0,
    'initial_alpha' : 1.0,
    'lr_alpha' : None,              # None falls back to the shared lr.
    'delta' : 0.0}                  # Imitation scalar.



# Function to check scalars. 
def fill_scalar_dicts(provided, required_names, defaults, description):

    provided = {} if provided is None else provided
    required_names = list(required_names)

    unknown_names = set(provided) - set(required_names)
    missing_is_fine = set(required_names) - set(provided)
    if unknown_names:
        raise ValueError(
            f"{description} has scalars for names which do not exist: {sorted(unknown_names)}"
            f"Names which do exist: {sorted(required_names)}")

    filled = {}
    for name in required_names:
        given = provided.get(name, {})
        unknown_keys = set(given) - set(defaults)
        if unknown_keys:
            raise ValueError(
                f"{description}, '{name}' has unknown scalars: {sorted(unknown_keys)}"
                f"Scalars which exist: {sorted(defaults)}")
        filled[name] = {**defaults, **given}

    return filled, sorted(missing_is_fine)



#------------------
# An agent acts based on an understanding of the relationship
# between its observations, its actions, and its environment.
#------------------

class Agent(Agent_Methods, nn.Module):

    def __init__(
            self,

            # Structure of the world model for make_world_model.
            hidden_state_sizes,
            list_of_dict_of_prior_input_encoder_class_dicts,
            list_of_dict_of_posterior_input_encoder_class_dicts,
            list_of_dict_of_prediction_decoder_class_dicts,
            lower_layer_posterior_sample_decoding_output_sizes,
            time_constants,

            # Structure of actor and critics.
            dict_of_action_decoder_class_dicts,         # For the actor.
            dict_of_action_encoder_class_dicts,         # For the critics.

            # Training scalars. Both are optional and fall back to the defaults above.
            list_of_dict_of_inner_state_scalar_dicts = None,   # For each layer, name : scalars.
            dict_of_action_scalar_dicts = None,                # name : scalars.

            # For reinforcement learning.
            number_of_critics = 2,                      # How many critics? The actor is judged by the lowest opinion of the critics. 
            tau = 0.01,                                 # How much should target critics meet critics?
            gamma = 0.99,                               # How much should critics think of the future?
            d = 1,                                      # Train the actor every d epochs.
            make_value_decoder = None,                  # A CALLABLE returning a fresh module.
                                                        # Sharing one instance would make every critic and target the same network.

            # Optimisation.
            lr = 0.0001,                    # General learning rate, replacing many not provided.
            lr_world_model = None,
            lr_critic = None,
            lr_actor = None,
            weight_decay = 0.00001,

            # Buffer and logging.
            capacity = 128,
            max_steps = 32,
            max_epochs_in_log = 64,

            verbose = False):

        super().__init__()

        self.tau = tau
        self.gamma = gamma
        self.d = d
        self.max_steps = max_steps
        self.max_epochs_in_log = max_epochs_in_log
        self.verbose = verbose

        lr_world_model = lr if lr_world_model is None else lr_world_model
        lr_critic = lr if lr_critic is None else lr_critic
        lr_actor = lr if lr_actor is None else lr_actor

        # World model.
        self.world_model = make_world_model(
            hidden_state_sizes,
            list_of_dict_of_prior_input_encoder_class_dicts,
            list_of_dict_of_posterior_input_encoder_class_dicts,
            list_of_dict_of_prediction_decoder_class_dicts,
            lower_layer_posterior_sample_decoding_output_sizes,
            time_constants,
            verbose = verbose)

        self.hidden_state_sizes = hidden_state_sizes
        self.time_constants = time_constants

        self.world_model_opt = optim.Adam(
            self.world_model.parameters(), lr = lr_world_model, weight_decay = weight_decay)

        # Training scalars, checked against what the world model actually built.
        if list_of_dict_of_inner_state_scalar_dicts is None:
            list_of_dict_of_inner_state_scalar_dicts = [None] * len(hidden_state_sizes)
        if len(list_of_dict_of_inner_state_scalar_dicts) != len(hidden_state_sizes):
            raise ValueError(
                "list_of_dict_of_inner_state_scalar_dicts needs one entry per layer.")

        self.list_of_dict_of_inner_state_scalar_dicts = []
        self.list_of_defaulted_inner_state_names = []
        for i, world_model_layer in enumerate(self.world_model.list_of_world_model_layers):
            inner_state_names = world_model_layer.posterior_inner_state_decoder.models_dict.keys()
            prediction_names = world_model_layer.prediction_decoder.models_dict.keys()
            if set(inner_state_names) != set(prediction_names):
                raise ValueError(
                    f"Layer {i} decodes {sorted(prediction_names)} but has inner states "
                    f"{sorted(inner_state_names)}. These must match.")
            filled, defaulted = fill_scalar_dicts(
                list_of_dict_of_inner_state_scalar_dicts[i],
                inner_state_names,
                DEFAULT_INNER_STATE_SCALARS,
                f"Layer {i}")
            self.list_of_dict_of_inner_state_scalar_dicts.append(filled)
            self.list_of_defaulted_inner_state_names.append(defaulted)

        self.dict_of_action_scalar_dicts, self.defaulted_action_names = fill_scalar_dicts(
            dict_of_action_scalar_dicts,
            dict_of_action_decoder_class_dicts.keys(),
            DEFAULT_ACTION_SCALARS,
            "The actor")

        # Actor, reading the lowest layer's hidden state.
        self.actor = Actor(
            hidden_state_sizes[0], dict_of_action_decoder_class_dicts, verbose = verbose)
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr = lr_actor, weight_decay = weight_decay)

        # Alpha, the entropy weight, learned in log space.
        self.log_alphas = nn.ParameterDict({
            name : nn.Parameter(torch.log(torch.tensor(
                float(scalars['initial_alpha']))))
            for name, scalars in self.dict_of_action_scalar_dicts.items()})

        self.alpha_opts = {
            name : optim.Adam(
                params = [self.log_alphas[name]],
                lr = lr if scalars['lr_alpha'] is None else scalars['lr_alpha'],
                weight_decay = 0)
            for name, scalars in self.dict_of_action_scalar_dicts.items()}

        # Critics and target critics.
        self.critics = nn.ModuleList()
        self.critic_targets = nn.ModuleList()
        self.critic_opts = []
        for _ in range(number_of_critics):
            critic = Critic(
                hidden_state_sizes[0], dict_of_action_encoder_class_dicts,
                value_decoder = None if make_value_decoder is None else make_value_decoder(),
                verbose = verbose)
            critic_target = Critic(
                hidden_state_sizes[0], dict_of_action_encoder_class_dicts,
                value_decoder = None if make_value_decoder is None else make_value_decoder(),
                verbose = False)
            critic_target.load_state_dict(critic.state_dict())
            for parameter in critic_target.parameters():
                parameter.requires_grad = False      # Target critics move by polyak averaging with tau.

            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.critic_opts.append(optim.Adam(
                critic.parameters(), lr = lr_critic, weight_decay = weight_decay))

        # Recurrent replay buffer, with shapes read off the world model.
        dict_of_observation_shapes, dict_of_action_shapes = shapes_from_world_model(self.world_model)
        self.buffer = Recurrent_Replay_Buffer(
            dict_of_observation_shapes, dict_of_action_shapes, capacity, max_steps)

        self.training_log = {'max_epochs_in_log' : self.max_epochs_in_log}
        self.training_log_actor = {'max_epochs_in_log' : self.max_epochs_in_log}
        self.epoch_num = 0

        self.build_episode_routing()

        if verbose:
            self.print_scalars()