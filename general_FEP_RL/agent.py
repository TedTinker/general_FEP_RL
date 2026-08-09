#%%
#------------------
# agent.py provides a class combining the world model, actor, and critics.
#------------------

from copy import deepcopy

import torch
from torch import nn
import torch.optim as optim

from general_FEP_RL.buffer import RecurrentReplayBuffer, shapes_from_world_model
from general_FEP_RL.world_model import World_Model, make_world_model
from general_FEP_RL.actor_critic import Actor, Critic



#------------------
# Training scalars.
#
# Every inner state gets all four. An inner state is named for the thing it is decoded
# into, so 'vision' and 'lower_layer_posterior_sample' are treated the same way: the
# old beta_hidden list, indexed by layer, is now just the beta of that layer's
# lower_layer_posterior_sample.
#
#   upsilon             accuracy scalar
#   beta                complexity scalar
#   eta_before_clamp    curiosity scalar, applied BEFORE clamping to [0, 1]
#   eta                 curiosity scalar, applied after
#------------------

DEFAULT_INNER_STATE_SCALARS = {
    'upsilon' : 1.0,
    'beta' : 0.03,                  # The value used throughout chapter 2.
    'eta_before_clamp' : 1.0,
    'eta' : 1.0}

DEFAULT_ACTION_SCALARS = {
    'target_entropy' : -1.0,
    'alpha_normal' : 1.0,
    'initial_alpha' : 1.0,
    'lr_alpha' : None,              # None falls back to the shared lr.
    'delta' : 0.0}                  # Imitation scalar.



def fill_scalar_dicts(provided, required_names, defaults, description):

    # Fills in defaults, and refuses names or keys that do not exist. A mistyped
    # modality would otherwise silently keep its default beta forever.

    provided = {} if provided is None else provided
    required_names = list(required_names)

    unknown_names = set(provided) - set(required_names)
    missing_is_fine = set(required_names) - set(provided)
    if unknown_names:
        raise ValueError(
            f"""
{description} has scalars for names which do not exist: {sorted(unknown_names)}
Names which do exist: {sorted(required_names)}
            """)

    filled = {}
    for name in required_names:
        given = provided.get(name, {})
        unknown_keys = set(given) - set(defaults)
        if unknown_keys:
            raise ValueError(
                f"""
{description}, '{name}' has unknown scalars: {sorted(unknown_keys)}
Scalars which exist: {sorted(defaults)}
                """)
        filled[name] = {**defaults, **given}

    return filled, sorted(missing_is_fine)



#------------------
# An agent acts based on an understanding of the relationship
# between its observations, its actions, and its environment.
#------------------

class Agent(nn.Module):

    def __init__(
            self,

            # Structure of the world model, passed straight through to make_world_model.
            hidden_state_sizes,
            list_of_dict_of_prior_input_encoder_class_dicts,
            list_of_dict_of_posterior_input_encoder_class_dicts,
            list_of_dict_of_prediction_decoder_class_dicts,
            lower_layer_posterior_sample_decoding_output_sizes,
            time_constants,

            # Structure of the actor and critics.
            dict_of_action_decoder_class_dicts,      # For the actor.
            dict_of_action_encoder_class_dicts,      # For the critics.

            # Training scalars. Both are optional and fall back to the defaults above.
            list_of_dict_of_inner_state_scalar_dicts = None,   # Per layer: name -> scalars.
            dict_of_action_scalar_dicts = None,                # name -> scalars.

            # Reinforcement learning.
            number_of_critics = 2,
            tau = 0.01,
            gamma = 0.99,
            d = 1,                                   # Train the actor every d epochs.
            make_value_decoder = None,               # A CALLABLE returning a fresh module.
                                                     # Sharing one instance would make every
                                                     # critic and target the same network.

            # Optimisation.
            lr = 0.0001,
            lr_world_model = None,
            lr_critic = None,
            lr_actor = None,
            weight_decay = 0.00001,

            # Buffer and logging.
            capacity = 128,
            max_steps = 32,
            max_epochs_in_log = 64,

            isolate_modality_posteriors = True,
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

        #------------------
        # World model.
        #------------------

        self.world_model = make_world_model(
            hidden_state_sizes,
            list_of_dict_of_prior_input_encoder_class_dicts,
            list_of_dict_of_posterior_input_encoder_class_dicts,
            list_of_dict_of_prediction_decoder_class_dicts,
            lower_layer_posterior_sample_decoding_output_sizes,
            time_constants,
            isolate_modality_posteriors = isolate_modality_posteriors,
            verbose = verbose)

        self.hidden_state_sizes = hidden_state_sizes
        self.time_constants = time_constants

        self.world_model_opt = optim.Adam(
            self.world_model.parameters(), lr = lr_world_model, weight_decay = weight_decay)

        #------------------
        # Training scalars, checked against what the world model actually built.
        #------------------

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
            # One scalar dict covers accuracy and complexity because these are the same
            # set: every inner state is decoded into exactly one prediction.
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

        #------------------
        # Actor, reading the lowest layer's hidden state.
        #------------------

        self.actor = Actor(
            hidden_state_sizes[0], dict_of_action_decoder_class_dicts, verbose = verbose)
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr = lr_actor, weight_decay = weight_decay)

        #------------------
        # Alpha, the entropy weight, learned in log space.
        #
        # There is no separate self.alphas holding stale copies: alphas is derived from
        # log_alphas on demand, so the two cannot drift apart when d > 1 and the actor
        # does not train every epoch.
        #------------------

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

        #------------------
        # Critics and target critics.
        #------------------

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
                parameter.requires_grad = False      # Targets move only by polyak averaging.

            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.critic_opts.append(optim.Adam(
                critic.parameters(), lr = lr_critic, weight_decay = weight_decay))

        #------------------
        # Recurrent replay buffer, with shapes read off the world model.
        #------------------

        dict_of_observation_shapes, dict_of_action_shapes = shapes_from_world_model(self.world_model)
        self.buffer = RecurrentReplayBuffer(
            dict_of_observation_shapes, dict_of_action_shapes, capacity, max_steps)

        self.training_log = {'max_epochs_in_log' : self.max_epochs_in_log}
        self.training_log_actor = {'max_epochs_in_log' : self.max_epochs_in_log}
        self.epoch_num = 0

        if verbose:
            self.print_scalars()




    
            
            
    def forward_one_step(
            self,
            list_of_previous_hidden_states,
            list_of_prior_values_dicts,
            list_of_posterior_values_dicts,
            use_posterior = True):      # False dreams: the hierarchy runs on its own priors.
     
        layers = self.list_of_world_model_layers
        num_layers = len(layers)
     
        list_of_inner_states = []
        list_of_prior_samples = []
        list_of_posterior_samples = []
        list_of_prior_prediction_dicts = []
        list_of_posterior_prediction_dicts = []
     
        # Whichever branch is driving is what feeds the layer above AND what advances the
        # hidden state. Aliasing one list here means those two can never disagree.
        list_of_driving_samples = (
            list_of_posterior_samples if use_posterior else list_of_prior_samples)
     
        # From bottom to top.
        for i, world_model_layer in enumerate(layers):
            prior_values = {
                **list_of_prior_values_dicts[i],
                'previous_hidden_state' : list_of_previous_hidden_states[i]}
            posterior_values = {
                **list_of_posterior_values_dicts[i],
                'previous_hidden_state' : list_of_previous_hidden_states[i]}
            if i > 0:
                posterior_values['lower_layer_posterior_sample'] = list_of_driving_samples[i - 1]
     
            inner_states = world_model_layer.make_inner_states(prior_values, posterior_values)
     
            list_of_inner_states.append(inner_states)
            list_of_prior_samples.append(
                world_model_layer.combine_inner_state_samples(inner_states, 'prior'))
            list_of_posterior_samples.append(
                world_model_layer.combine_inner_state_samples(inner_states, 'posterior'))
            list_of_prior_prediction_dicts.append(
                world_model_layer.make_predictions(list_of_prior_samples[i]))
            list_of_posterior_prediction_dicts.append(
                world_model_layer.make_predictions(list_of_posterior_samples[i]))
     
        # From top to bottom.
        list_of_new_hidden_states = [None] * num_layers
        for i in range(num_layers - 1, -1, -1):
            list_of_new_hidden_states[i] = layers[i].make_hidden_state(
                previous_hidden_state = list_of_previous_hidden_states[i],
                inner_state_sample = list_of_driving_samples[i],
                higher_layer_hidden_state = (
                    None if i == num_layers - 1 else list_of_new_hidden_states[i + 1]))
     
        return {
            'list_of_hidden_states' : list_of_new_hidden_states,
            'list_of_inner_states' : list_of_inner_states,
            'list_of_prior_samples' : list_of_prior_samples,          # New.
            'list_of_posterior_samples' : list_of_posterior_samples,
            'list_of_prior_predictions' : list_of_prior_prediction_dicts,
            'list_of_posterior_predictions' : list_of_posterior_prediction_dicts}
            
    

class EpochMixin:

    #------------------
    # A buffer batch, laid out the way the world model wants it.
    #
    # obs holds one more step than action: o_0 ... o_T against a_0 ... a_{T-1}.
    # Model step k pairs o_k with a_{k-1}, the action that CAUSED it, so the prior sees
    # the action whose consequences it is predicting. Step 0 gets a zero action, which
    # is why nothing at step 0 is ever scored -- the mask starts at step 1.
    #
    # This is the same prepend-a-zero-action trick as the old epoch, and the reason the
    # buffer keeps that extra observation slot.
    #------------------

    def episode_values_from_batch(self, batch):
        example = next(iter(batch['action'].values()))
        episode_length = example.shape[1]

        complete_action = {
            name : torch.cat([torch.zeros_like(value[:, :1]), value], dim = 1)
            for name, value in batch['action'].items()}

        list_of_lists_of_prior_values_dicts = []
        list_of_lists_of_posterior_values_dicts = []
        for k in range(episode_length + 1):
            value_dict = {
                **{name : value[:, k : k + 1] for name, value in batch['obs'].items()},
                **{name : value[:, k : k + 1] for name, value in complete_action.items()}}
            list_of_lists_of_prior_values_dicts.append(
                self.route(value_dict, self.list_of_prior_input_names, 'prior'))
            list_of_lists_of_posterior_values_dicts.append(
                self.route(value_dict, self.list_of_posterior_input_names, 'posterior'))

        return list_of_lists_of_prior_values_dicts, list_of_lists_of_posterior_values_dicts


    @staticmethod
    def per_step(value):
        # (batch, steps, ...) -> (batch, steps, 1), averaging over everything a step
        # contains. Doing this unconditionally matters: a (batch, steps, 5) tensor has
        # the same ndim as the mask, so a shape test would leave it alone and the
        # denominator below would then be five times too small.
        return value.reshape(*value.shape[:2], -1).mean(dim = -1, keepdim = True)

    def masked_mean(self, value, mask):
        # Everything past the end of an episode contributes to neither numerator nor
        # denominator.
        return (self.per_step(value) * mask).sum() / mask.sum().clamp(min = 1.0)



    #------------------
    # Train the world model, actor, critics, and alpha parameters.
    #------------------

    def epoch(self, batch_size):
        self.train()

        batch = self.buffer.sample(batch_size)
        if batch is None:
            return None, None

        reward = batch['reward']
        done = batch['done']
        mask = batch['mask']                            # (batch, T, 1)
        best_action = batch['best_action']
        best_action_mask = batch['best_action_mask']
        episode_length = mask.shape[1]

        accuracy_losses, complexity_losses = {}, {}
        prior_stds, posterior_stds = {}, {}
        dkls, curiosities, curiosity_saturations = {}, {}, {}
        critic_losses, critic_predictions = [], []
        entropies_target_critic = {}
        sac_entropies_target_critic = {}
        normal_entropies_target_critic = {}
        alpha_entropies, alpha_normal_entropies = {}, {}
        entropies, target_entropies = {}, {}
        total_entropies, imitation_losses, alpha_losses = {}, {}, {}



        #------------------
        # World model: minimise free energy.
        #
        # Accuracy is scored on the PRIOR predictions, which is what the diagram draws
        # and what makes the prior a real one-step-ahead predictor. The posterior
        # predictions come back too and cost nothing extra, so adding a reconstruction
        # term later is a one-line change here.
        #------------------

        prior_values, posterior_values = self.episode_values_from_batch(batch)
        list_of_step_dicts = self.world_model(prior_values, posterior_values)

        accuracy_loss = 0.
        complexity_loss = 0.

        for i, world_model_layer in enumerate(self.world_model.list_of_world_model_layers):
            layer_key = f'layer_{i}'
            accuracy_losses[layer_key] = {}
            complexity_losses[layer_key] = {}
            prior_stds[layer_key] = {}
            posterior_stds[layer_key] = {}
            dkls[layer_key] = {}
            scalars_for_layer = self.list_of_dict_of_inner_state_scalar_dicts[i]

            for name, scalars in scalars_for_layer.items():

                # Steps 1..T. Step 0 saw only a zero action and is never scored.
                predicted = torch.cat(
                    [step_dict['list_of_prior_predictions'][i][name]
                     for step_dict in list_of_step_dicts[1:]], dim = 1)

                if name == 'lower_layer_posterior_sample':
                    # A network output, so detached: without this the layer below learns
                    # to be predictable rather than informative.
                    target = torch.cat(
                        [step_dict['list_of_posterior_samples'][i - 1]
                         for step_dict in list_of_step_dicts[1:]], dim = 1).detach()
                else:
                    target = batch['obs'][name][:, 1:]

                loss_func = world_model_layer.prediction_decoder.models_dict[name].loss_func
                layer_accuracy = self.masked_mean(loss_func(predicted, target), mask)
                accuracy_loss = accuracy_loss + scalars['upsilon'] * layer_accuracy
                accuracy_losses[layer_key][name] = layer_accuracy.item()

                dkl = torch.cat(
                    [step_dict['list_of_inner_states'][i][name]['dkl']
                     for step_dict in list_of_step_dicts[1:]], dim = 1)
                dkl = dkl.mean(dim = -1, keepdim = True)
                dkls[layer_key][name] = dkl.detach()

                layer_complexity = self.masked_mean(dkl, mask)
                complexity_loss = complexity_loss + scalars['beta'] * layer_complexity
                complexity_losses[layer_key][name] = layer_complexity.item()

                # sigma sits in the denominator of the complexity term, so a collapsing
                # prior_std inflates every curiosity value. Worth watching directly.
                for which, store in [('prior_std', prior_stds), ('posterior_std', posterior_stds)]:
                    std = torch.cat(
                        [step_dict['list_of_inner_states'][i][name][which]
                         for step_dict in list_of_step_dicts[1:]], dim = 1).detach()
                    store[layer_key][name] = self.masked_mean(std, mask).item()

        # Hidden states are taken before the graph is freed. h_all[k] is the hidden
        # state after step k, so h_t and h_{t+1} are two slices of one tensor rather
        # than two things that could fall out of step.
        h_all = torch.cat(
            [step_dict['list_of_hidden_states'][0] for step_dict in list_of_step_dicts],
            dim = 1).detach()                           # (batch, T+1, hidden_state_size)
        h_t = h_all[:, :-1]                             # chose a_t after seeing o_t
        h_tp1 = h_all[:, 1:]

        self.world_model_opt.zero_grad()
        (accuracy_loss + complexity_loss).backward()
        self.world_model_opt.step()



        #------------------
        # Curiosity: the complexity of the step the action led to.
        #------------------

        curiosity = torch.zeros_like(reward)

        for i in range(len(self.world_model.list_of_world_model_layers)):
            layer_key = f'layer_{i}'
            curiosities[layer_key] = {}
            curiosity_saturations[layer_key] = {}
            for name, scalars in self.list_of_dict_of_inner_state_scalar_dicts[i].items():
                before_clamp = dkls[layer_key][name] * scalars['eta_before_clamp']
                this_curiosity = scalars['eta'] * torch.clamp(before_clamp, min = 0, max = 1)
                this_curiosity = this_curiosity * mask
                curiosity = curiosity + this_curiosity
                curiosities[layer_key][name] = self.masked_mean(this_curiosity, mask).item()
                # A source pinned at the ceiling carries no information: it reports the
                # same number for a mild surprise and a total one.
                curiosity_saturations[layer_key][name] = self.masked_mean(
                    (before_clamp >= 1.0).float(), mask).item()

        total_reward = (reward + curiosity).detach() * mask



        #------------------
        # Target critics make target Q-values.
        #
        # G(o_t, a_t) =
        #   -DKL[q(z_t | o_t, h_{t-1}) || p(z_t | h_{t-1})]     (Curiosity)
        #   -r(s_t, a_t)                                        (Extrinsic Reward)
        #   -H(pi(a_t | o_t))                                   (Entropy)
        #   -E_{pi(a_t | o_t)}[log p(a_t* | o_t)]               (Imitation)
        #------------------

        with torch.no_grad():

            action_tp1, log_prob_tp1 = self.actor(h_tp1)

            Q_tp1 = torch.min(torch.stack(
                [critic_target(h_tp1, action_tp1) for critic_target in self.critic_targets],
                dim = 0), dim = 0)[0]

            entropy_bonus_tp1 = torch.zeros_like(Q_tp1)
            sac_entropy_tp1 = torch.zeros_like(Q_tp1)
            normal_prior_tp1 = torch.zeros_like(Q_tp1)

            for name, log_prob in log_prob_tp1.items():
                scalars = self.dict_of_action_scalar_dicts[name]
                sac_entropy = self.alpha(name) * (-log_prob)
                flat_action = action_tp1[name].flatten(start_dim = 2)
                normal_prior = (0.5 * scalars['alpha_normal']
                                * (flat_action ** 2).sum(-1, keepdim = True))
                key_bonus = sac_entropy - normal_prior

                entropy_bonus_tp1 = entropy_bonus_tp1 + key_bonus
                sac_entropy_tp1 = sac_entropy_tp1 + sac_entropy
                normal_prior_tp1 = normal_prior_tp1 + normal_prior

                sac_entropies_target_critic[name] = self.masked_mean(sac_entropy, mask).item()
                normal_entropies_target_critic[name] = self.masked_mean(normal_prior, mask).item()
                entropies_target_critic[name] = self.masked_mean(key_bonus, mask).item()

            not_done = (1.0 - done) * mask
            future_Q_value = self.gamma * not_done * (Q_tp1 + entropy_bonus_tp1)
            Q_target = (total_reward + future_Q_value) * mask



        #------------------
        # Train critics to match Q_target.
        #------------------

        for i, critic in enumerate(self.critics):
            Q_pred = critic(h_t, batch['action'])
            critic_loss = 0.5 * self.masked_mean((Q_pred - Q_target) ** 2, mask)
            critic_losses.append(critic_loss.item())

            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()

            with torch.no_grad():
                for target_parameter, parameter in zip(
                        self.critic_targets[i].parameters(), critic.parameters()):
                    target_parameter.copy_(
                        self.tau * parameter + (1.0 - self.tau) * target_parameter)

            critic_predictions.append(self.masked_mean(Q_pred, mask).item())



        #------------------
        # Train the actor to minimise expected free energy.
        #------------------

        if self.epoch_num % self.d != 0:
            trained_actor = False
            epoch_dict_actor = None
        else:
            trained_actor = True

            new_action, new_log_prob, imitation_loss = self.actor(h_t, best_action)

            Q = torch.min(torch.stack(
                [critic(h_t, new_action) for critic in self.critics], dim = 0), dim = 0)[0]

            entropy = torch.zeros_like(Q)
            total_imitation_loss = torch.zeros_like(Q)

            for name in new_action.keys():
                scalars = self.dict_of_action_scalar_dicts[name]

                # Detached: alpha is trained by its own loss below, not by the actor's.
                alpha_entropy = self.alpha(name).detach() * (-new_log_prob[name])
                flat_action = new_action[name].flatten(start_dim = 2)
                alpha_normal_entropy = (0.5 * scalars['alpha_normal']
                                        * (flat_action ** 2).sum(-1, keepdim = True))
                total_entropy = alpha_entropy - alpha_normal_entropy
                entropy = entropy + total_entropy

                this_imitation = (self.per_step(imitation_loss[name])
                                  * scalars['delta'] * best_action_mask)
                total_imitation_loss = total_imitation_loss + this_imitation

                entropies[name] = self.masked_mean(-new_log_prob[name], mask).item()
                target_entropies[name] = float(scalars['target_entropy'])
                alpha_entropies[name] = self.masked_mean(alpha_entropy, mask).item()
                alpha_normal_entropies[name] = self.masked_mean(alpha_normal_entropy, mask).item()
                total_entropies[name] = self.masked_mean(total_entropy, mask).item()
                imitation_losses[name] = self.masked_mean(
                    this_imitation, best_action_mask * mask).item()

            Q = self.masked_mean(Q, mask)
            entropy = self.masked_mean(entropy, mask)
            total_imitation_loss = self.masked_mean(total_imitation_loss, mask)

            actor_loss = -Q - entropy + total_imitation_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Train alpha values.
            for name, log_prob in new_log_prob.items():
                target_entropy = self.dict_of_action_scalar_dicts[name]['target_entropy']
                alpha_loss = self.log_alphas[name] * (-log_prob - target_entropy).detach()
                alpha_loss = self.masked_mean(alpha_loss, mask)

                self.alpha_opts[name].zero_grad()
                alpha_loss.backward()
                self.alpha_opts[name].step()

                alpha_losses[name] = alpha_loss.item()

            epoch_dict_actor = {
                'epoch_num' : self.epoch_num,
                'actor_loss' : actor_loss.item(),
                'Q_for_actor' : -Q.item(),
                'entropy_for_actor' : -entropy.item(),
                'total_imitation_loss' : total_imitation_loss.item(),
                'entropies' : entropies,
                'target_entropies' : target_entropies,
                'alpha_entropies' : alpha_entropies,
                'alpha_normal_entropies' : alpha_normal_entropies,
                'total_entropies' : total_entropies,
                'imitation_losses' : imitation_losses,
                'alpha_losses' : alpha_losses,
                'alphas' : {name : self.alpha(name).item() for name in self.log_alphas},
                'log_alphas' : {name : p.item() for name, p in self.log_alphas.items()}}

            self.add_to_training_log(epoch_dict_actor, actor = True)



        epoch_dict = {
            'epoch_num' : self.epoch_num,

            'obs' : {name : v.detach().cpu() for name, v in batch['obs'].items()},
            'action' : {name : v.detach().cpu() for name, v in batch['action'].items()},
            'best_action' : {name : v.detach().cpu() for name, v in best_action.items()},
            'reward' : reward.detach().cpu(),
            'done' : done.detach().cpu(),
            'mask' : mask.detach().cpu(),
            'best_action_mask' : best_action_mask.detach().cpu(),

            'accuracy_losses' : accuracy_losses,
            'complexity_losses' : complexity_losses,
            'prior_stds' : prior_stds,
            'posterior_stds' : posterior_stds,
            'curiosity_saturations' : curiosity_saturations,
            'average_reward' : self.masked_mean(reward, mask).item(),
            'curiosity' : self.masked_mean(curiosity, mask).item(),
            'curiosities' : curiosities,
            'total_reward' : self.masked_mean(total_reward, mask).item(),

            'critic_losses' : critic_losses,
            'target_critic_output' : self.masked_mean(Q_tp1, mask).item(),
            'entropy_target_critic' : self.masked_mean(entropy_bonus_tp1, mask).item(),
            'entropies_target_critic' : entropies_target_critic,
            'sac_entropies_target_critic' : sac_entropies_target_critic,
            'normal_entropies_target_critic' : normal_entropies_target_critic,
            'sac_entropy_target_critic' : self.masked_mean(sac_entropy_tp1, mask).item(),
            'normal_entropy_target_critic' : self.masked_mean(normal_prior_tp1, mask).item(),
            'future_Q_value' : self.masked_mean(future_Q_value, mask).item(),
            'Q_target' : self.masked_mean(Q_target, mask).item(),
            'critic_predictions' : critic_predictions,
            'trained_actor' : trained_actor}

        self.epoch_num += 1
        self.add_to_training_log(epoch_dict)
        return epoch_dict, epoch_dict_actor



    #------------------
    # Derived on demand, never cached, so it cannot drift from log_alphas when d > 1.
    #------------------

    def alpha(self, name):
        return torch.exp(self.log_alphas[name])



    def apply_mask(self, tensor, mask):
        ndims_to_add = tensor.ndim - mask.ndim
        expanded_mask = mask.view(*mask.shape, *(1,) * ndims_to_add)
        return tensor * expanded_mask



    #------------------
    # Logging. Unchanged from the old version.
    #------------------

    def _index_to_drop(self, epochs):
        best_i, best_merged = 1, float('inf')
        for i in range(1, len(epochs) - 1):
            merged = epochs[i + 1] - epochs[i - 1]
            if merged < best_merged:
                best_merged, best_i = merged, i
        return best_i

    def _drop_index(self, log, k):
        for value in log.values():
            if isinstance(value, dict):
                self._drop_index(value, k)
            elif isinstance(value, list):
                if value and isinstance(value[0], list):
                    for series in value:
                        if k < len(series):
                            del series[k]
                elif k < len(value):
                    del value[k]

    def _prune_log(self, log):
        epochs = log.get('epoch_num', [])
        while len(epochs) > self.max_epochs_in_log:
            self._drop_index(log, self._index_to_drop(epochs))

    def recursive_log_append(self, log, new_data):
        for key, value in new_data.items():
            if isinstance(value, dict):
                if key not in log:
                    log[key] = {}
                self.recursive_log_append(log[key], value)
            elif isinstance(value, (list, tuple)):
                if key not in log:
                    log[key] = [[] for _ in range(len(value))]
                for i, item in enumerate(value):
                    log[key][i].append(deepcopy(item))
            else:
                if key not in log:
                    log[key] = []
                log[key].append(deepcopy(value))

    def add_to_training_log(self, epoch_dict, actor = False):
        log = self.training_log_actor if actor else self.training_log
        self.recursive_log_append(log, epoch_dict)
        self._prune_log(log)

        