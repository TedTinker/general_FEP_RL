#%%
#------------------
# agent_methods.py provides methods for the actor, including epoch.
#------------------

import torch
from copy import deepcopy



# These input names are generated inside the world model, so they should not be used elsewhere.
GENERATED_INPUT_NAMES = {'previous_hidden_state', 'lower_layer_posterior_sample'}



class Agent_Methods:



    # Make lists of names of prior inputs, posterior inputs, and predictions.
    # Call at the end of __init__.
    def build_episode_routing(self):
        self.list_of_prior_input_names = []
        self.list_of_posterior_input_names = []
        self.list_of_observation_prediction_names = []
        for world_model_layer in self.world_model.list_of_world_model_layers:
            self.list_of_prior_input_names.append([
                name for name in world_model_layer.prior_input_encoder.models_dict
                if name not in GENERATED_INPUT_NAMES])
            self.list_of_posterior_input_names.append([
                name for name in world_model_layer.posterior_input_encoder.models_dict
                if name not in GENERATED_INPUT_NAMES])
            # In a "dream," the agent feeds itself its own predictions.
            self.list_of_observation_prediction_names.append([
                name for name in world_model_layer.prediction_decoder.models_dict
                if name not in GENERATED_INPUT_NAMES])

    def route(self, value_dict, list_of_names, description):
        list_of_value_dicts = []
        for i, names in enumerate(list_of_names):
            missing = sorted(name for name in names if name not in value_dict)
            if missing:
                raise ValueError(
                    f"Layer {i}'s {description} encoder asks for {missing}, which this step was not given."
                    f"Given this step: {sorted(value_dict)}")
            list_of_value_dicts.append({name : value_dict[name] for name in names})
        return list_of_value_dicts



    # To begin an episode, initiate hidden states and actions as dummies (just zeros). 
    def begin(self, batch_size = 1):
        self.step_num = 0
        example_parameter = next(self.parameters())
        self.hallucinated_observation = None    # What a dream feeds itself next step.

        self.action = {
            name : torch.zeros(
                batch_size, 1, *model.output_shape,
                device = example_parameter.device, dtype = example_parameter.dtype)
            for name, model in self.actor.action_decoder.models_dict.items()}

        self.hidden_states = self.world_model.start_hidden_states(batch_size)



    #------------------
    # In each step, the agent encodes the current observation and its previous action to
    # update hidden states, then decodes a new action and predicts observations and Q.
    #
    # use_posterior = True  is a "real" step: observations arrive and the posterior sample advances the hierarchy.
    # use_posterior = False is a "dream" step: the prior sample advances it instead, and observations are not observed.
    #------------------

    def step_in_episode(
            self,
            observation = None,         # name : (batch, 1, ...). Required unless dreaming.
            use_posterior = True,       # True if a "real" step, False if a "dream" step.
            deterministic = False):     # tanh(mu) rather than a sample, for evaluation.

        self.step_num += 1

        with torch.no_grad():

            # Check if real step/dream step is handled correctly.
            if observation is None:
                if use_posterior:
                    raise ValueError(
                        "A posterior step is a real episode and needs an observation. "
                        "Pass use_posterior = False to dream instead.")
                if self.hallucinated_observation is None:
                    raise ValueError(
                        "The first step after begin has nothing to hallucinate from. "
                        "Take one step with an observation before dreaming on.")
                observation = self.hallucinated_observation

            # Use the world model.
            value_dict = {**observation, **self.action}
            step_dict = self.world_model.forward_one_step(
                self.hidden_states,
                self.route(value_dict, self.list_of_prior_input_names, 'prior'),
                self.route(value_dict, self.list_of_posterior_input_names, 'posterior'),
                use_posterior = use_posterior)

            # Get hidden states, actions, and Q-value predictions. 
            self.hidden_states = step_dict['list_of_hidden_states']
            self.action, log_prob = self.actor(self.hidden_states[0], deterministic = deterministic) 
            values = [critic(self.hidden_states[0], self.action) for critic in self.critics]

            # Sort out predictions.
            predictions = step_dict[
                'list_of_posterior_predictions' if use_posterior else 'list_of_prior_predictions']
            self.hallucinated_observation = {
                name : predictions[i][name]
                for i, names in enumerate(self.list_of_observation_prediction_names)
                for name in names}

        return {
            'observation' : observation,
            'action' : self.action,
            'log_prob' : log_prob,
            'values' : values,
            'hidden_states' : self.hidden_states,
            'list_of_inner_states' : step_dict['list_of_inner_states'],
            'prior_predictions' : step_dict['list_of_prior_predictions'],
            'posterior_predictions' : step_dict['list_of_posterior_predictions'],
            'dreamed' : not use_posterior}



    #------------------
    # Organize a batch of episodes the way the world model needs.
    #
    # Observations have one more step than actions: o_0 ... o_T against a_0 ... a_{T-1}.
    # Model step k pairs o_k with a_{k-1}, the action that CAUSED it, so the prior sees
    # the action whose consequences it is predicting. Step 0 gets a dummy action.
    # The mask starts at step 1.
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



    # Averaging (batch, steps, ...) to make (batch, steps, 1).
    # This is to match the shape of the mask.
    @staticmethod
    def per_step(value):
        return value.reshape(*value.shape[:2], -1).mean(dim = -1, keepdim = True)

    # Steps after the end of an episode contributes nothing.
    def masked_mean(self, value, mask):
        return (self.per_step(value) * mask).sum() / mask.sum().clamp(min = 1.0)
    
    # Applying masks.
    def apply_mask(self, tensor, mask):
        ndims_to_add = tensor.ndim - mask.ndim
        expanded_mask = mask.view(*mask.shape, *(1,) * ndims_to_add)
        return tensor * expanded_mask
    
    # Making alpha values.
    def alpha(self, name):
        return torch.exp(self.log_alphas[name])




    # Train the world model, actor, critics, and alpha parameters.
    def epoch(self, batch_size):
        self.train()

        # Sample a batch. If unable, return nothing.
        batch = self.buffer.sample(batch_size)
        if batch is None:
            return None, None

        reward = batch['reward']
        done = batch['done']
        mask = batch['mask']                        
        best_action = batch['best_action']
        best_action_mask = batch['best_action_mask']
        episode_length = mask.shape[1]

        # Dictionaries and lists for logging training data.
        accuracy_losses_prior, accuracy_losses_posterior, complexity_losses = {}, {}, {}
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
        # The world model trains to minimise free energy, by maximizing accuracy and minimizing complexity.
        #
        # Accuracy: The prior predicts the observations of the current state
        # (which is effectively the same as the posterior of the previous step predicting it).
        # We also find the posterior's "predictions" (reconstructions), which can also be employed.
        
        # Complexity: Kullback-Leibler Divergence (DKL) comparing posterior and prior inner states.
        #------------------

        prior_values, posterior_values = self.episode_values_from_batch(batch)
        list_of_step_dicts = self.world_model(prior_values, posterior_values)

        accuracy_loss = 0.
        complexity_loss = 0.

        for i, world_model_layer in enumerate(self.world_model.list_of_world_model_layers):
            layer_key = f'layer_{i}'
            accuracy_losses_prior[layer_key] = {}
            accuracy_losses_posterior[layer_key] = {}
            complexity_losses[layer_key] = {}
            prior_stds[layer_key] = {}
            posterior_stds[layer_key] = {}
            dkls[layer_key] = {}
            scalars_for_layer = self.list_of_dict_of_inner_state_scalar_dicts[i]

            for name, scalars in scalars_for_layer.items():

                # Ignore predictions of step 0.
                predicted = torch.cat(
                    [step_dict['list_of_prior_predictions'][i][name]
                     for step_dict in list_of_step_dicts[1:]], dim = 1)
                reconstructed = torch.cat(
                    [step_dict['list_of_posterior_predictions'][i][name]
                     for step_dict in list_of_step_dicts[1:]], dim = 1)

                # Detach targets if they are generated by the model itself.
                # Otherwise, the model learns to be predictable instead of storing information.
                if name == 'lower_layer_posterior_sample':
                    target = torch.cat(
                        [step_dict['list_of_posterior_samples'][i - 1]
                         for step_dict in list_of_step_dicts[1:]], dim = 1).detach()
                else:
                    target = batch['obs'][name][:, 1:]

                loss_func = world_model_layer.prediction_decoder.models_dict[name].loss_func
                layer_accuracy_prior = self.masked_mean(loss_func(predicted, target), mask)
                layer_accuracy_posterior = self.masked_mean(loss_func(reconstructed, target), mask)
                accuracy_losses_prior[layer_key][name] = layer_accuracy_prior.item()
                accuracy_losses_posterior[layer_key][name] = layer_accuracy_posterior.item()
                accuracy_loss = accuracy_loss + scalars['upsilon_prior'] * layer_accuracy_prior + scalars['upsilon_posterior'] * layer_accuracy_posterior

                dkl = torch.cat(
                    [step_dict['list_of_inner_states'][i][name]['dkl']
                     for step_dict in list_of_step_dicts[1:]], dim = 1)
                dkl = dkl.mean(dim = -1, keepdim = True)
                dkls[layer_key][name] = dkl.detach()

                layer_complexity = self.masked_mean(dkl, mask)
                complexity_loss = complexity_loss + scalars['beta'] * layer_complexity
                complexity_losses[layer_key][name] = layer_complexity.item()

                # Prepare std for calculating curiosity. If std collapses, curiosity inflates. 
                for which, store in [('prior_std', prior_stds), ('posterior_std', posterior_stds)]:
                    std = torch.cat(
                        [step_dict['list_of_inner_states'][i][name][which]
                         for step_dict in list_of_step_dicts[1:]], dim = 1).detach()
                    store[layer_key][name] = self.masked_mean(std, mask).item()

        # Prepare hidden states for use by other models. 
        h_all = torch.cat(
            [step_dict['list_of_hidden_states'][0] for step_dict in list_of_step_dicts],
            dim = 1).detach()                           # (batch, T+1, hidden_state_size)
        h_t = h_all[:, :-1]                             # chose a_t after seeing o_t
        h_tp1 = h_all[:, 1:]

        self.world_model_opt.zero_grad()
        (accuracy_loss + complexity_loss).backward()
        self.world_model_opt.step()



        # Calculating curiosity, which is the complexity value of the next step.
        # This is clamped to avoid massive DKLs.
        curiosity = torch.zeros_like(reward)
        for i in range(len(self.world_model.list_of_world_model_layers)):
            layer_key = f'layer_{i}'
            curiosities[layer_key] = {}
            curiosity_saturations[layer_key] = {}
            for name, scalars in self.list_of_dict_of_inner_state_scalar_dicts[i].items():
                before_clamp = dkls[layer_key][name] * scalars['eta_before_clamp']
                this_curiosity = scalars['eta'] * torch.clamp(before_clamp, min = 0, max = 1)   # Clamping at one stops exploding DKL.
                this_curiosity = this_curiosity * mask
                curiosity = curiosity + this_curiosity
                curiosities[layer_key][name] = self.masked_mean(this_curiosity, mask).item()
                curiosity_saturations[layer_key][name] = self.masked_mean(
                    (before_clamp >= 1.0).float(), mask).item()

        total_reward = (reward + curiosity).detach() * mask



        #------------------
        # Target critics make target Q-values. 
        # This applies a Bellman equation to expected free energy:
        #
        # G(o_t, a_t) =
        #   -r(s_t, a_t)                                        (Extrinsic Reward)
        #   -H(pi(a_t | o_t))                                   (Entropy)
        #   -DKL[q(z_t | o_t, h_{t-1}) || p(z_t | h_{t-1})]     (Curiosity)
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



        # The critics train to predict Q_target.
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



        # The actor trains to minimise expected free energy.
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
            
            

            # Alpha values train to match target entropy. Train alpha values
            for name, log_prob in new_log_prob.items():
                target_entropy = self.dict_of_action_scalar_dicts[name]['target_entropy']
                alpha_loss = self.log_alphas[name] * (-log_prob - target_entropy).detach()
                alpha_loss = self.masked_mean(alpha_loss, mask)

                self.alpha_opts[name].zero_grad()
                alpha_loss.backward()
                self.alpha_opts[name].step()

                alpha_losses[name] = alpha_loss.item()


            # Record training data.
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

            'accuracy_losses_prior' : accuracy_losses_prior,
            'accuracy_losses_posterior' : accuracy_losses_posterior,
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



    # Logging training data.
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
        
        

    # Called from __init__ when verbose. 
    def print_scalars(self):
        print("\nInner state scalars, per layer:")
        for i, scalars_for_layer in enumerate(self.list_of_dict_of_inner_state_scalar_dicts):
            defaulted = set(self.list_of_defaulted_inner_state_names[i])
            print(f"  layer {i}  (hidden_state_size {self.hidden_state_sizes[i]}, "
                  f"time_constant {self.time_constants[i]}):")
            for name, scalars in sorted(scalars_for_layer.items()):
                note = "   [all defaults]" if name in defaulted else ""
                print(f"    {name:34s} " + "  ".join(
                    f"{key} {value}" for key, value in sorted(scalars.items())) + note)

        print("\nAction scalars:")
        for name, scalars in sorted(self.dict_of_action_scalar_dicts.items()):
            note = "   [all defaults]" if name in self.defaulted_action_names else ""
            print(f"  {name:36s} " + "  ".join(
                f"{key} {value}" for key, value in sorted(scalars.items())) + note)

        print("\nBuffer:")
        for name, buffer in sorted(self.buffer.observation_buffers.items()):
            print(f"  observation {name:26s} {tuple(buffer.shape)}")
        for name, buffer in sorted(self.buffer.action_buffers.items()):
            print(f"  action      {name:26s} {tuple(buffer.shape)}")
        print()