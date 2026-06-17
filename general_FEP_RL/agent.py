#%% 



#------------------
# agent.py provides a class combining the world model, actor, and critics.
#------------------

from copy import deepcopy 

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from general_FEP_RL.utils_torch import tile_batch_dim
from general_FEP_RL.buffer import RecurrentReplayBuffer
from general_FEP_RL.world_model import World_Model
from general_FEP_RL.actor_critic import Actor, Critic
    
    

#------------------
# An agent acts based on an understanding of the relationship 
# between its observations, its actions, and its environment.
#------------------

class Agent:
    
    def __init__(
            self,             
            observation_dict,       # Keys: observation_names
                                    # Values: 
                                        # encoder
                                            # example_input
                                            # example_output
                                        # encoder_arg_dict
                                            # encode_size
                                            # zp_zq_sizes
                                        # decoder
                                            # example_input
                                            # example_output
                                            # loss_func
                                        # decoder_arg_dict
                                        # upsilon_obs (accuracy scalar)
                                        # beta_obs (complexity scalar)
                                        # eta_before_clamp
                                        # eta (curiosity scalar)
            
            action_dict,            # Keys: action_names
                                    # Values: 
                                        # encoder
                                            # example_input
                                            # example_output
                                        # encoder_arg_dict
                                            # encode_size
                                        # decoder
                                            # example_input
                                            # example_output
                                            # loss_func
                                        # decoder_arg_dict
                                        # target_entropy
                                        # alpha_normal
                                        # lr_alpha
                                        # initial_alpha
                                        # delta (imitation scalar)
            
            hidden_state_sizes,
            time_scales = [1],
            beta_hidden = [],
            eta_before_clamp = [],
            eta = [],
            upsilon_reward = 1,
            
            number_of_critics = 2, 
            tau = .1,
            gamma = .99,
            value_decoder = None,
            d = 1,
            
            lr = .0001,
            lr_world_model = None,
            lr_critic = None,
            lr_actor = None,
            lr_alpha = None,          # This should be separate for each action.
            weight_decay = .00001,
            
            capacity = 128, 
            max_steps = 32,
            max_epochs_in_log = 64,
            
            verbose = False):

        # Miscellaneous. 
        self.observation_dict = observation_dict
        self.action_dict = action_dict
        self.hidden_state_sizes = hidden_state_sizes
        self.time_scales = time_scales
        self.beta_hidden = beta_hidden
        self.eta_before_clamp = eta_before_clamp
        self.eta = eta
        self.upsilon_reward = upsilon_reward
        self.tau = tau
        if lr_world_model is None:
            lr_world_model = lr 
        if lr_critic is None:
            lr_critic = lr 
        if lr_actor is None:
            lr_actor = lr 
        lr_alpha = {key : lr if action_dict[key]['lr_alpha'] == None else action_dict[key]['lr_alpha'] for key in action_dict.keys()}
        self.gamma = gamma
        self.d = d
        self.max_epochs_in_log = max_epochs_in_log
        self.verbose = verbose

        # World model.
        self.world_model = World_Model(hidden_state_sizes, observation_dict, action_dict, time_scales, verbose = self.verbose)
        self.world_model_opt = optim.Adam(
            self.world_model.parameters(), 
            lr = lr_world_model, 
            weight_decay = weight_decay)
        
        # Actor.
        self.actor = Actor(hidden_state_sizes[0], action_dict, verbose = self.verbose)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr = lr_actor, weight_decay = weight_decay) 
        
        # Alpha values (entropy hyperparameter).
        self.alphas = {key : 1.0 if action_dict[key]['initial_alpha'] == None else action_dict[key]['initial_alpha'] for key in action_dict.keys()} 
        self.log_alphas = nn.ParameterDict({key: nn.Parameter(torch.log(torch.tensor(value))) for key, value in self.alphas.items()})        
        self.alpha_opt = {key : optim.Adam(
            params=[self.log_alphas[key]], 
            lr = lr_alpha[key], 
            weight_decay = 0) for key in action_dict.keys()} 
        
        # Critics and target critics.
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []
        for _ in range(number_of_critics):
            self.critics.append(Critic(hidden_state_sizes[0], action_dict, value_decoder, verbose = self.verbose))
            self.critic_targets.append(Critic(hidden_state_sizes[0], action_dict, value_decoder, verbose = False))
            self.critic_targets[-1].load_state_dict(self.critics[-1].state_dict())
            self.critic_opts.append(optim.Adam(
                self.critics[-1].parameters(), 
                lr = lr_critic, 
                weight_decay = weight_decay))
        
        # Recurrent replay buffer.
        self.buffer = RecurrentReplayBuffer(
            self.world_model.observation_model_dict, 
            self.world_model.action_model_dict, 
            capacity, 
            max_steps)
        
        self.training_log = {"max_epochs_in_log" : self.max_epochs_in_log}
        self.training_log_actor = {"max_epochs_in_log" : self.max_epochs_in_log}
        self.epoch_num = 0
        
        self.begin()
        
        
        
    # To begin an episode, initiate prior hidden state and action.
    def begin(self, batch_size = 1):
        self.action = {} 
        for key, model in self.actor.action_model_dict.items(): 
            action = torch.zeros_like(model['decoder'].example_output[0, 0].unsqueeze(0).unsqueeze(0))
            self.action[key] = tile_batch_dim(action, batch_size)
        self.hp = [torch.zeros((batch_size, 1, hidden_state_size)) for hidden_state_size in self.hidden_state_sizes] 
        self.hq = [torch.zeros((batch_size, 1, hidden_state_size)) for hidden_state_size in self.hidden_state_sizes] 
        
        
    
    # In each step, an agent encodes current observation and previous action to update hidden states.
    # Then, decode a new action and predict future observations and Q-values.
    def step_in_episode(self, obs, posterior = True, best_action = None, eval = False):
        with torch.no_grad():
            if best_action is not None:
                self.action = best_action
            if eval:
                self.set_eval()
            else:
                self.set_train()
            self.hp, self.hq, inner_state_dict = self.world_model(
                prev_hidden_states = self.hq if posterior else self.hp, 
                obs = obs, prev_action = self.action, one_step = True)
            self.action, log_prob = self.actor(self.hq[0] if posterior else self.hp[0]) 
            encoded_action = self.world_model.action_in(self.action)
            pred_obs_p = self.world_model.predict(self.hp[0], encoded_action)
            pred_obs_q = self.world_model.predict(self.hq[0], encoded_action)
            values = []
            for i in range(len(self.critics)):
                value = self.critics[i](self.hq[0], self.action) 
                values.append(value)
                
        return {
            'obs' : obs,
            'action' : self.action,
            'log_prob' : log_prob,
            'values' : values,
            'inner_state_dict' : inner_state_dict,
            'pred_obs_p' : pred_obs_p,
            'pred_obs_q' : pred_obs_q}
        
        

    # Train the world model, actor, critics, and alpha parameters.
    def epoch(self, batch_size):
        self.set_train()
                                
        # Gather data. 
        batch = self.buffer.sample(batch_size)
        obs = batch['obs']
        action = batch['action'] 
        best_action = batch['best_action'] 
        reward = batch['reward']
        done = batch['done']
        mask = batch['mask']
        best_action_mask = batch['best_action_mask']
        
        batch_size = reward.shape[0]
        
        # Shapes: (batch, steps, ...)
        # Steps: 
        #   t = 0, 1, 2, 3, ..., n.
        # obs includes final observation, t = n+1.
        
        # Add initial action t = -1.
        complete_action = {}
        for key, value in action.items(): 
            empty_action = torch.zeros_like(
                self.actor.action_model_dict[key]['decoder'].example_output[0, 0].unsqueeze(0).unsqueeze(0))
            empty_action = tile_batch_dim(empty_action, batch_size)
            complete_action[key] = torch.cat([empty_action, value], dim = 1)
            
        # This mask also masks t = -1 
        complete_mask = torch.cat([torch.zeros_like(mask[:, :1]), mask], dim=1)



        # Some dictionaries for logging process.
        accuracy_losses = {}
        complexity_losses = {}
        dkls = {}
        
        curiosities = {}
        
        critic_losses = []
        entropies_target_critic = {}
        sac_entropies_target_critic = {}
        normal_entropies_target_critic = {}
        critic_predictions = []
        
        alpha_entropies = {}
        alpha_normal_entropies = {}
        total_entropies = {}
        imitation_losses = {}
        
        alpha_losses = {}
        
        
        
        # Train world model to minimize Free Energy.
        hp, hq, inner_state_dict, pred_obs_p, pred_obs_q = self.world_model(
            prev_hidden_states = None, obs = obs, prev_action = complete_action)
        
        # hp and hq steps: 
        #   t = -1, 0, 1, ..., n+1
        
        
        
        # Accuracy of observation prediction.
        # Given T steps and i = 0, ..., n parts of observations,
        # Accuracy = E_{q(z_{0:T,i})}[log p(o_{1:T+1,i}|z_{0:T,i})]) * mask_{0:T}
        accuracy_loss = 0
        
        for key, value in self.observation_dict.items():
            true_obs = obs[key][:, 1:]
            predicted_obs = pred_obs_q[key]
            loss_func = self.observation_dict[key]['decoder'].loss_func
            scalar = self.observation_dict[key]['upsilon_obs']
            obs_accuracy_loss = loss_func(predicted_obs, true_obs)
            obs_accuracy_loss = obs_accuracy_loss.mean(dim=tuple(range(2, obs_accuracy_loss.ndim))).unsqueeze(-1)
            obs_accuracy_loss = (obs_accuracy_loss * scalar * mask).sum() / mask.sum()
            accuracy_loss = accuracy_loss + obs_accuracy_loss
            accuracy_losses[key] = obs_accuracy_loss.item()
        reward_accuracy_loss = F.mse_loss(pred_obs_q['extrinsic_reward'], reward, reduction = 'mean')
        reward_accuracy_loss = (reward_accuracy_loss * self.upsilon_reward * mask).sum() / mask.sum()
        accuracy_loss = accuracy_loss + reward_accuracy_loss
        accuracy_losses[key] = reward_accuracy_loss.item()
            
        # Complexity of predictions.
        # Given T steps and i = 0, ..., n parts of observations,
        # Complexity = beta_i DKL[q(z_{0:T,i}) || p(z_{0:T,i})] * mask_{0:T}
        complexity_loss = 0
        
        for key, value in self.observation_dict.items():
            dkl = inner_state_dict[key]['dkl'].mean(-1).unsqueeze(-1) * complete_mask
            dkls[key] = dkl.detach()
            complexity = dkl * self.observation_dict[key]['beta_obs']
            complexity_for_key = (complexity[:, 1:] * mask).sum() / mask.sum()
            complexity_loss = complexity_loss + complexity_for_key
            complexity_losses[key] = complexity_for_key.detach()

        for i, beta in enumerate(self.beta_hidden):
            dkl = inner_state_dict[i+1]['dkl'].mean(-1).unsqueeze(-1) * complete_mask 
            dkls[f'hidden_layer_{i+2}'] = dkl.detach()
            complexity = dkl * beta
            complexity_for_key = (complexity[:, 1:] * mask).sum() / mask.sum()
            complexity_loss = complexity_loss + complexity_for_key
            complexity_losses[f'hidden_layer_{i+2}'] = complexity_for_key.detach()
                        
        
                                
        # Minimize Free Energy = Complexity - Accuracy
        self.world_model_opt.zero_grad()
        (accuracy_loss + complexity_loss).backward()
        self.world_model_opt.step()
                

        
        # Get curiosity values based on complexity of next step.
        curiosity = torch.zeros_like(reward)
                
        for key, value in self.observation_dict.items():
            obs_curiosity = self.observation_dict[key]['eta'] * \
                torch.clamp(dkls[key][:, 1:] * self.observation_dict[key]['eta_before_clamp'], min = 0, max = 1)
            obs_curiosity = obs_curiosity * mask
            curiosity = curiosity + obs_curiosity
            curiosities[key] = obs_curiosity.sum().item() / mask.sum().item()
            
        for i in range(len(self.hidden_state_sizes) - 1):
            obs_curiosity = self.eta[i] * \
                torch.clamp(dkls[f'hidden_layer_{i+2}'][:, 1:] * self.eta_before_clamp[i], min = 0, max = 1)            
            obs_curiosity = obs_curiosity * mask
            curiosity = curiosity + obs_curiosity
            curiosities[f'hidden_layer_{i+2}'] = obs_curiosity.sum().item() / mask.sum().item()
            
            
            
        # The actor and critics are concerned with both extrinsic rewards and intrinsic rewards in Expected Free Energy.
        # G(o_t, a_t) = 
        #   -DKL[q(z_t | o_t, h_{t-1}) || p(z_t | h_{t-1})]     (Curiosity)
        #   -r(s_t, a_t)}                                       (Extrinsic Reward)
        #   -H(\pi_\phi(a_t | o_t))                             (Entropy)
        #   -E_{\pi_\phi(a_t | o_t)} [\log p(a_t^\ast | o_t)]   (Imitation)
        total_reward = (reward + curiosity).detach() * mask
        
        hq_all = hq[0].detach()                 # (B, T+2, H), including initializing hq and final, unused hq.
        h_t    = hq_all[:, 1:-1]                # (B, T,   H)  -> h_t
        h_tp1  = hq_all[:, 2:]                  # (B, T,   H)  -> h_{t+1}
        
                
        
        # Target critics make target Q-values.
        with torch.no_grad():

            # Next-step action and log-prob.
            a_tp1, logp_tp1 = self.actor(h_tp1.detach())
        
            # Target critics evaluate next-step value.
            Q_tp1_list = [critic_tgt(h_tp1.detach(), a_tp1) * mask for critic_tgt in self.critic_targets]
            Q_tp1 = torch.min(torch.stack(Q_tp1_list, dim=0), dim=0)[0]
            # Q_tp1 shape: (B, T, 1)
        
            # Entropy bonus (rewarded entropy).
            entropy_bonus_tp1 = torch.zeros_like(Q_tp1)  
            sac_entropy_tp1   = torch.zeros_like(Q_tp1)  
            normal_prior_tp1  = torch.zeros_like(Q_tp1)  
            for key, lp in logp_tp1.items():
                sac_entropy  = self.alphas[key] * (-lp)
                flat_a       = a_tp1[key].flatten(start_dim=2)
                normal_prior = (0.5 * self.action_dict[key]['alpha_normal']
                                * (flat_a ** 2).sum(-1, keepdim=True))
                key_bonus = sac_entropy - normal_prior
            
                entropy_bonus_tp1 += key_bonus            # <-- only this feeds future_Q_value
                sac_entropy_tp1   += sac_entropy
                normal_prior_tp1  += normal_prior
            
                sac_entropies_target_critic[key]    = (sac_entropy  * mask).sum().item() / mask.sum().item()
                normal_entropies_target_critic[key] = (normal_prior * mask).sum().item() / mask.sum().item()
                entropies_target_critic[key]        = (key_bonus    * mask).sum().item() / mask.sum().item()
            
            not_done = (1.0 - done) * mask
            future_Q_value = self.gamma * not_done * (Q_tp1 + entropy_bonus_tp1)
            Q_target = total_reward + future_Q_value
        
            # Mask invalid timesteps.
            Q_target = Q_target * mask
                        
        
        
        # Train critics to match Q_target        
        for i, critic in enumerate(self.critics):
            Q_pred = critic(h_t.detach(), action)
            td_error = Q_pred - Q_target
            critic_loss = 0.5 * (td_error**2 * mask).sum() / mask.sum()
            critic_losses.append(critic_loss.item())
        
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
        
            # Soft-update target critic with polyak averaging.
            for tgt_p, p in zip(self.critic_targets[i].parameters(), critic.parameters()):
                tgt_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * tgt_p.data)
                
            critic_predictions.append((Q_pred * mask).sum().item() / mask.sum().item())
        
        
        
        # Train actor to minimize expected free energy.
        
        if self.epoch_num % self.d != 0:
            trained_actor = False
            epoch_dict_actor = None
        else:
            trained_actor = True 
                
            new_action_dict, new_log_pis_dict, imitation_loss = self.actor(h_t.detach(), best_action)
            
            Q_list = [critic(h_t.detach(), new_action_dict) for critic in self.critics]
            Q = torch.min(torch.stack(Q_list, dim=0), dim=0)[0]
            
            entropy = torch.zeros_like(Q)
            
            for key in new_action_dict.keys():
                lp = new_log_pis_dict[key]
                alpha_entropy = self.alphas[key] * (-lp)
            
                flat_a = new_action_dict[key].flatten(start_dim=2)
                alpha_normal_entropy = (
                    0.5 * self.action_dict[key]['alpha_normal']
                    * (flat_a ** 2).sum(-1, keepdim=True))
            
                total_entropy = alpha_entropy - alpha_normal_entropy
                entropy += total_entropy
                
                alpha_entropies[key] = (alpha_entropy * mask).sum().item() / mask.sum().item()
                alpha_normal_entropies[key] = (alpha_normal_entropy * mask).sum().item() / mask.sum().item()
                total_entropies[key] = (total_entropy * mask).sum().item() / mask.sum().item()
            
            total_imitation_loss = torch.zeros_like(Q)
            
            for key in new_action_dict.keys():
                scalar = self.action_dict[key]['delta']
                il = imitation_loss[key] * scalar * best_action_mask * mask
                total_imitation_loss = total_imitation_loss + il
                imitation_losses[key] = il.sum().item() / (best_action_mask * mask).sum()
            
            Q = (Q * mask).sum() / mask.sum()
            entropy = (entropy * mask).sum() / mask.sum()
            total_imitation_loss = (total_imitation_loss * mask).sum() / mask.sum()
            
            actor_loss = - Q - entropy - total_imitation_loss
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            
            
            # Train alpha values.
            logp_t = new_log_pis_dict
            
            for key, lp in logp_t.items():
                alpha_loss = self.log_alphas[key] * (-lp - self.action_dict[key]['target_entropy']).detach()
                alpha_loss = (alpha_loss * mask).sum() / mask.sum()
            
                self.alpha_opt[key].zero_grad()
                alpha_loss.backward()
                self.alpha_opt[key].step()
            
                self.alphas[key] = torch.exp(self.log_alphas[key])
                alpha_losses[key] = alpha_loss.item()
                
            
                
            epoch_dict_actor = {
                'epoch_num' : self.epoch_num,
                
                # Save values related to actor.
                'actor_loss' : actor_loss.item(),
                'Q_for_actor' : -Q.item(),
                'entropy_for_actor' : -entropy.item(),
                'total_imitation_loss' : -total_imitation_loss.item(),
                'alpha_entropies' : alpha_entropies,
                'alpha_normal_entropies' : alpha_normal_entropies,
                'total_entropies' : total_entropies,
                'imitation_losses' : imitation_losses,
                
                # Save values related to alpha-values.
                'alpha_losses' : alpha_losses,
                'alphas' : {key : a.item() for key, a in self.alphas.items()},
                'log_alphas' : {key : a.item() for key, a in self.log_alphas.items()},
                }
            
            self.add_to_training_log(epoch_dict_actor, actor = True) 
            


        epoch_dict = {
            # Save batch.
            'epoch_num' : self.epoch_num,
            'obs' : {key: v.detach().cpu() for key, v in obs.items()},
            'action' : {key: v.detach().cpu() for key, v in action.items()},
            'best_action' : {key: v.detach().cpu() for key, v in best_action.items()},
            'reward' : reward.detach().cpu(),
            'done' : done.detach().cpu(),
            'mask' : mask.detach().cpu(),
            'best_action_mask' : best_action_mask.detach().cpu(),
            
            # Save predictions and inner states.
            'pred_obs_p': {key: self.apply_mask(v, mask).detach().cpu() for key, v in pred_obs_p.items()},
            'pred_obs_q': {key: self.apply_mask(v, mask).detach().cpu() for key, v in pred_obs_q.items()}, 
            # inner_state_dict
            
            'accuracy_losses' : accuracy_losses,
            'complexity_losses' : complexity_losses,
            'average_reward' : reward.mean().item(),
            'curiosity' : curiosity.mean().item(),
            'curiosities' : curiosities,
            'total_reward' : total_reward.mean().item(),
            
            # Save values related to critics.
            'critic_losses' : critic_losses,
            'target_critic_output' : Q_tp1.mean().item(),
            'entropy_target_critic' : entropy_bonus_tp1.mean().item(),
            'entropies_target_critic'        : entropies_target_critic,          # net (per key)
            'sac_entropies_target_critic'    : sac_entropies_target_critic,      # new
            'normal_entropies_target_critic' : normal_entropies_target_critic,   # new
            'sac_entropy_target_critic'      : sac_entropy_tp1.mean().item(),    # new
            'normal_entropy_target_critic'   : normal_prior_tp1.mean().item(),   # new
            'future_Q_value' : future_Q_value.mean().item(),
            'Q_target' : Q_target.mean().item(),
            'critic_predictions' : critic_predictions,
            }
        
        self.epoch_num += 1
        self.add_to_training_log(epoch_dict) 
        return(epoch_dict, epoch_dict_actor)
        
        
        
    def apply_mask(self, tensor, mask):
        ndims_to_add = tensor.ndim - mask.ndim
        expanded_mask = mask.view(*mask.shape, *(1,) * ndims_to_add)
        return (tensor * expanded_mask)
    
    
    
    # Pick which logged epoch to evict: never the first or last, and among
    # interior points choose the one in the densest region (smallest gap left
    # behind when removed). This keeps coverage even from start to finish
    # instead of hollowing out the early epochs into one smooth void.
    def _index_to_drop(self, epochs):
        best_i, best_merged = 1, float('inf')
        for i in range(1, len(epochs) - 1):
            merged = epochs[i + 1] - epochs[i - 1]   # gap that remains if i is dropped
            if merged < best_merged:
                best_merged, best_i = merged, i
        return best_i

    # Remove one position from every series in the (possibly nested) log,
    # so all series stay aligned with epoch_num.
    def _drop_index(self, log, k):
        for value in log.values():
            if isinstance(value, dict):
                self._drop_index(value, k)
            elif isinstance(value, list):
                if value and isinstance(value[0], list):     # list-of-series, e.g. per-critic
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
                
                
            
    def add_to_training_log(self, epoch_dict, actor=False):
        log = self.training_log_actor if actor else self.training_log
        self.recursive_log_append(log, epoch_dict)
        self._prune_log(log)
                                    
    

    def set_eval(self):
        self.world_model.eval()
        self.world_model.use_sample = False
        
        self.actor.eval() 
        for key, module in self.actor.action_model_dict.items():            
            module["decoder"].mu_std.deterministic = True
                
        for i in range(len(self.critics)):
            self.critics[i].eval()
            self.critic_targets[i].eval()

    def set_train(self):
        self.world_model.train()
        self.world_model.use_sample = True
        
        self.actor.train()
        for key, module in self.actor.action_model_dict.items():            
            module["decoder"].mu_std.deterministic = False
        
        for i in range(len(self.critics)):
            self.critics[i].train()
            self.critic_targets[i].train()
        
    
    
    def save_state_dict(self, file = None):
        state = {}
    
        # -------- World model --------
        state["world_model"] = self.world_model.state_dict()
    
        # Observation models (explicit, for convenience)
        state["observation_models"] = {}
        for key, model in self.world_model.observation_model_dict.items():
            state["observation_models"][key] = {
                "encoder": model["encoder"].state_dict(),
                "decoder": model["decoder"].state_dict()}
    
        # -------- Actor --------
        state["actor"] = self.actor.state_dict()
    
        # -------- Critics --------
        state["critics"] = [critic.state_dict() for critic in self.critics]
        state["critic_targets"] = [tc.state_dict() for tc in self.critic_targets]
    
        # -------- Entropy / temperature parameters --------
        state["alphas"] = {
            key: v.detach().cpu()
            for key, v in self.alphas.items()}
        state["log_alphas"] = {
            key: v.detach().cpu()
            for key, v in self.log_alphas.items()}
    
        # -------- Metadata --------
        state["meta"] = {
            "hidden_state_sizes": self.hidden_state_sizes,
            "time_scales": self.time_scales,
            "gamma": self.gamma}
        
        if file is not None:
            torch.save(state, f"{file}.pth")
    
        return state
            
        
        
    def load_state_dict(self, file, keys = None):
        if keys is None:
            keys = ["world_model", "observation_models", "actor", "critics", "critic_targets", "alphas", "log_alphas"]
        state = torch.load(f"{file}.pth", map_location="cpu")
    
        # -------- World model --------
        if "world_model" in keys and "world_model" in state:
            self.world_model.load_state_dict(state["world_model"])
    
        # -------- Observation models (encoders / decoders) --------
        if "observation_models" in keys and "observation_models" in state:
            for obs_key, obs_state in state["observation_models"].items():
                if obs_key not in self.world_model.observation_model_dict:
                    continue
    
                model = self.world_model.observation_model_dict[obs_key]
    
                if "encoder" in obs_state:
                    model["encoder"].load_state_dict(obs_state["encoder"])
    
                if "decoder" in obs_state:
                    model["decoder"].load_state_dict(obs_state["decoder"])
    
        # -------- Actor --------
        if "actor" in keys and "actor" in state:
            self.actor.load_state_dict(state["actor"])
    
        # -------- Critics --------
        if "critics" in keys and "critics" in state:
            for critic, critic_state in zip(self.critics, state["critics"]):
                critic.load_state_dict(critic_state)
    
        # -------- Target critics --------
        if "critic_targets" in keys and "critic_targets" in state:
            for tc, tc_state in zip(self.critic_targets, state["critic_targets"]):
                tc.load_state_dict(tc_state)
    
        # -------- Entropy / temperature --------
        if "alphas" in keys and "alphas" in state:
            for key in self.alphas:
                if key in state["alphas"]:
                    self.alphas[key] = state["alphas"][key]
    
        if "log_alphas" in keys and "log_alphas" in state:
            for key in self.log_alphas:
                if key in state["log_alphas"]:
                    self.log_alphas[key].data.copy_(state["log_alphas"][key])
    
        # -------- Metadata (optional sanity check) --------
        if "meta" in state:
            meta = state["meta"]
    
            if hasattr(self, "hidden_state_sizes"):
                if meta.get("hidden_state_sizes") != self.hidden_state_sizes:
                    print("Warning: hidden_state_sizes differ from checkpoint")
    
            if hasattr(self, "time_scales"):
                if meta.get("time_scales") != self.time_scales:
                    print("Warning: time_scales differ from checkpoint")
    
        return state

        
        
        
#------------------
# Example
#------------------

if __name__ == '__main__':
    
    from general_FEP_RL.utils_torch import generate_dummy_inputs
    from general_FEP_RL.encoders.encode_image import Encode_Image
    from general_FEP_RL.decoders.decode_image import Decode_Image
    
    
    
    observation_dict = {
        'see_image' : {
            'encoder' : Encode_Image,
            'encoder_arg_dict' : {                
                'encode_size' : 256,
                'zp_zq_sizes' : [256]},
            'decoder' : Decode_Image,
            'decoder_arg_dict' : {},
            'upsilon_obs' : 1,                               
            'beta_obs' : 1,                             
            'eta_before_clamp' : 1,
            'eta' : 1},
        'see_image_2' : {
            'encoder' : Encode_Image,
            'encoder_arg_dict' : {                
                'encode_size' : 16,
                'zp_zq_sizes' : [16]},
            'decoder' : Decode_Image,
            'decoder_arg_dict' : {},
            'upsilon_obs' : 1,                               
            'beta_obs' : 1,
            'eta_before_clamp' : 1,                               
            'eta' : 1}}
    
    action_dict = {
        'make_image' : {
            'encoder' : Encode_Image,
            'encoder_arg_dict' : {                
                'encode_size' : 128,
                'zp_zq_sizes' : [128]},
            'decoder' : Decode_Image,
            'decoder_arg_dict' : {},
            'target_entropy' : -1,
            'alpha_normal' : 1,
            'delta' : 0}}
    
    
    
    agent = Agent(
        observation_dict = observation_dict,       
        action_dict = action_dict,       
        hidden_state_sizes = [128],
        time_scales = [1],
        eta_before_clamp = [],
        eta = [],
        number_of_critics = 2, 
        tau = .1,
        lr = .0001,
        weight_decay = .00001,
        gamma = .99,
        capacity = 128, 
        max_steps = 32)
    
    dummies = generate_dummy_inputs(
        agent.world_model.observation_model_dict, 
        agent.world_model.action_model_dict, 
        agent.hidden_state_sizes, 
        batch=1, 
        steps=1)
    dummy_inputs = dummies['obs_enc_in']
        
    agent.step_in_episode(dummy_inputs)
        
    agent.world_model.summary()
    
    

    agent = Agent(
        observation_dict = observation_dict,       
        action_dict = action_dict,       
        hidden_state_sizes = [128, 128],
        time_scales = [1, 2],
        eta_before_clamp = [1],
        eta = [1],
        number_of_critics = 2, 
        tau = .99,
        lr = .0001,
        weight_decay = .00001,
        gamma = .99,
        capacity = 128, 
        max_steps = 32)
    
    dummies = generate_dummy_inputs(
        agent.world_model.observation_model_dict, 
        agent.world_model.action_model_dict, 
        agent.hidden_state_sizes, 
        batch=1, 
        steps=1)
    dummy_inputs = dummies['obs_enc_in']
        
    agent.step_in_episode(dummy_inputs)
        
    agent.world_model.summary()
# %%
