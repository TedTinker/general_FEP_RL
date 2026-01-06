#------------------
# agent.py provides a class combining the world model, actor, and critics.
#------------------

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
                                        # accuracy_scalar
                                        # beta_obs (complexity scalar)
                                        # 'eta_before_clamp'
                                        # eta
            
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
                                        # delta
            
            hidden_state_sizes,
            time_scales = [1],
            beta_hidden = [],
            eta_before_clamp = [],
            eta = [],
            
            number_of_critics = 2, 
            tau = .1,
            
            lr = .0001,
            weight_decay = .00001,
            gamma = .99,
            capacity = 128, 
            max_steps = 32):

        # Miscellaneous. 
        self.observation_dict = observation_dict
        self.action_dict = action_dict
        self.hidden_state_sizes = hidden_state_sizes
        self.beta_hidden = beta_hidden
        self.eta_before_clamp = eta_before_clamp
        self.eta = eta
        self.tau = tau
        self.gamma = gamma

        # World model.
        self.world_model = World_Model(hidden_state_sizes, observation_dict, action_dict, time_scales)
        self.world_model_opt = optim.Adam(self.world_model.parameters(), lr = lr, weight_decay = weight_decay)
                           
        # Actor.
        self.actor = Actor(hidden_state_sizes[0], action_dict)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr = lr, weight_decay = weight_decay) 
        
        # Alpha values (entropy hyperparameter).
        self.alphas = {key : 1 for key in action_dict.keys()} 
        self.log_alphas = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(1))
            for key in action_dict})        
        self.alpha_opt = {key : optim.Adam(params=[self.log_alphas[key]], lr = lr, weight_decay = weight_decay) 
                          for key in action_dict.keys()} 
        
        # Critics and target critics.
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []
        for _ in range(number_of_critics):
            self.critics.append(Critic(hidden_state_sizes[0], action_dict))
            self.critic_targets.append(Critic(hidden_state_sizes[0], action_dict))
            self.critic_targets[-1].load_state_dict(self.critics[-1].state_dict())
            self.critic_opts.append(optim.Adam(self.critics[-1].parameters(), lr = lr, weight_decay = weight_decay))
        
        # Recurrent replay buffer.
        self.buffer = RecurrentReplayBuffer(
            self.world_model.observation_model_dict, 
            self.actor.action_model_dict, 
            capacity, 
            max_steps)
        
        self.begin()
        
        
        
    # To begin an episode, initiate prior hidden state and action.
    def begin(self, batch_size = 1):
        self.action = {} 
        for key, model in self.actor.action_model_dict.items(): 
            action = torch.zeros_like(model['decoder'].example_output[0, 0].unsqueeze(0).unsqueeze(0))
            self.action[key] = tile_batch_dim(action, batch_size)
        self.hp = [torch.zeros((batch_size, 1, hidden_state_size)) for hidden_state_size in self.hidden_state_sizes] 
        self.hq = [torch.zeros((batch_size, 1, hidden_state_size)) for hidden_state_size in self.hidden_state_sizes] 
        
        
    
    # In each step, an agent processes an observation and an action to update hidden states.
    # Then, make a new action and predict future observations and Q-values.
    def step_in_episode(self, obs, posterior = True, best_action = None):
        with torch.no_grad():
            if best_action is not None:
                self.action = best_action
            self.set_eval()
            self.hp, self.hq, inner_state_dict = self.world_model(
                self.hq if posterior else self.hp, obs, self.action, one_step = True)
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
        
        

    # Train the world model, actor, and critics.
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
        
        # Shapes: (batch, steps, ...)
        # Steps: 
        #   t = 0, 1, 2, 3, ..., n.
        # obs includes final observation, t = n+1.
        
        # Add initial action t = -1.
        complete_action = {}
        for key, value in action.items(): 
            empty_action = torch.zeros_like(self.actor.action_model_dict[key]['decoder'].example_output[0, 0].unsqueeze(0).unsqueeze(0))
            empty_action = tile_batch_dim(empty_action, batch_size)
            complete_action[key] = torch.cat([empty_action, value], dim = 1)
            
        # This mask also masks t = -1 
        complete_mask = torch.cat([torch.ones(mask.shape[0], 1, 1), mask], dim = 1)


                                    
        # Train world model to minimize Free Energy.
        hp, hq, inner_state_dict, pred_obs_p, pred_obs_q = self.world_model(None, obs, complete_action)
        
        # hp and hq steps: 
        #   t = -1, 0, 1, ..., n+1
        
        
        
        # Accuracy of observation prediction.
        accuracy_losses = {}
        accuracy_loss = 0
        for key, value in self.observation_dict.items():
            true_obs = obs[key][:, 1:]
            predicted_obs = pred_obs_q[key]
            loss_func = self.observation_dict[key]['decoder'].loss_func
            scalar = self.observation_dict[key]['accuracy_scalar']
            obs_accuracy_loss = loss_func(predicted_obs, true_obs)
            obs_accuracy_loss = obs_accuracy_loss.mean(dim=tuple(range(2, obs_accuracy_loss.ndim))).unsqueeze(-1)
            obs_accuracy_loss = obs_accuracy_loss * scalar * mask
            accuracy_losses[key] = obs_accuracy_loss.mean().item()
            accuracy_loss = accuracy_loss + obs_accuracy_loss.mean()
            
        # Complexity of predictions.
        complexity_losses = {}
        complexity_loss = 0
        for key, value in self.observation_dict.items():
            dkl = inner_state_dict[key]['dkl'].mean(-1).unsqueeze(-1) * complete_mask
            complexity = dkl * self.observation_dict[key]['beta_obs']
            complexity_losses[key] = complexity[:,1:]
            complexity_loss = complexity_loss + complexity.mean()
        for i, beta in enumerate(self.beta_hidden):
            dkl = inner_state_dict[i+1]['dkl'].mean(-1).unsqueeze(-1) * complete_mask 
            complexity = dkl * beta
            complexity_losses[f'hidden_layer_{i+2}'] = complexity[:,1:]
            complexity_loss = complexity_loss + complexity.mean()
                        
        
                                
        # Minimize Free Energy.
        self.world_model_opt.zero_grad()
        (accuracy_loss + complexity_loss).backward()
        self.world_model_opt.step()
                

        
        # Get curiosity values based on complexity.
        # Idea: Maybe complexity should equal curiosity?
        curiosities = {}
        curiosity = torch.zeros_like(reward)
                
        for key, value in self.observation_dict.items():
            obs_curiosity = self.observation_dict[key]['eta'] * \
                torch.clamp(complexity_losses[key] * self.observation_dict[key]['eta_before_clamp'], min = 0, max = 1)
            complexity_losses[key] = complexity_losses[key].mean().item() # Replace tensor with scalar for plotting.
            curiosities[key] = obs_curiosity.mean().item()
            curiosity = curiosity + obs_curiosity
            
        for i in range(len(self.hidden_state_sizes) - 1):
            obs_curiosity = self.eta[i] * \
                torch.clamp(complexity_losses[f'hidden_layer_{i+2}'] * self.eta_before_clamp[i], min = 0, max = 1)
            complexity_losses[f'hidden_layer_{i+2}'] = complexity_losses[f'hidden_layer_{i+2}'].mean().item()
            curiosities[f'hidden_layer_{i+2}'] = obs_curiosity.mean().item() # Replace tensor with scalar for plotting.
            curiosity = curiosity + obs_curiosity
            
            
            
        # The actor and critics are concerned with both extrinsic rewards and intrinsic rewards.
        total_reward = (reward + curiosity).detach()
        

                
        # Train critics. First, target critics predict future Q-values.
        with torch.no_grad():         
            new_action_dict, new_log_pis_dict = self.actor(hq[0][:, 1:-1].detach())
            new_action_dict = {k: v[:, :1] for k, v in new_action_dict.items()}
            Q_target_nexts = []
            for i in range(len(self.critics)):
                Q_target_next = self.critic_targets[i](hq[0][:, 1:-1].detach(), new_action_dict)
                Q_target_nexts.append(Q_target_next)                
            Q_target_nexts_stacked = torch.stack(Q_target_nexts, dim=0)
            Q_target_next, _ = torch.min(Q_target_nexts_stacked, dim=0)

            new_entropy = torch.zeros_like(list(new_log_pis_dict.values())[0])
            for key, new_log_pis in new_log_pis_dict.items():
                new_entropy += self.alphas[key] * new_log_pis                        
            Q_target = total_reward + self.gamma * (1 - done) * (Q_target_next - new_entropy) 
            Q_target *= mask
        
        # Then, critics predict rewards plus predicted Q-values, with Bellman's Equation.
        critic_losses = []
        for i in range(len(self.critics)):
            action = {k: v[:, :-1] for k, v in action.items()}
            Q = self.critics[i](hq[0][:, :-1].detach(), action) * mask
            critic_loss = 0.5*F.mse_loss(Q, Q_target)
            critic_losses.append(critic_loss.item())
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
            for critic_target, critic in zip(self.critic_targets[i].parameters(), self.critics[i].parameters()):
                critic_target.data.copy_(self.tau*critic.data + (1.0-self.tau)*critic_target.data)
                                            
            
        
        # Train actor. First, actor makes new actions, and the critic grades them.
        new_action_dict, new_log_pis_dict, imitation_loss = self.actor(hq[0][:, 1:-1].detach(), best_action)
        Qs = []
        for i in range(len(self.critics)):
            Q = self.critics[i](hq[0][:, 1:-1].detach(), new_action_dict)
            Qs.append(Q)
        Qs_stacked = torch.stack(Qs, dim=0)
        Q, _ = torch.min(Qs_stacked, dim=0)
        Q = Q.mean(-1).unsqueeze(-1)
        
        # Then, calculate entropy values.
        alpha_entropies = {}
        alpha_normal_entropies = {}
        total_entropies = {}
        complete_entropy = torch.zeros_like(Q)
        for key in new_action_dict.keys():
            flattened_new_action = new_action_dict[key].flatten(start_dim = 2)
            alpha_entropy = self.alphas[key] * new_log_pis_dict[key]
            alpha_normal_entropy = 0.5 * self.action_dict[key]['alpha_normal'] * (flattened_new_action**2).sum(-1, keepdim=True)
            total_entropy = alpha_entropy + alpha_normal_entropy
            
            alpha_entropies[key] = alpha_entropy.mean().item()
            alpha_normal_entropies[key] = alpha_normal_entropy.mean().item()
            total_entropies[key] = total_entropy.mean().item()
            
            complete_entropy += total_entropy 
            
        # Also calculate imatation value. 
        imitations = {}
        total_imitation_loss = torch.zeros_like(Q)
        for key in new_action_dict.keys():
            scalar = self.action_dict[key]['delta']
            action_imitation_loss = imitation_loss[key].mean(-1) * scalar * mask.squeeze(-1) * best_action_mask.squeeze(-1)
            imitations[key] = action_imitation_loss.mean().item()
            total_imitation_loss += action_imitation_loss.mean()
                    
        actor_loss = (complete_entropy - Q - total_imitation_loss) * mask    
        actor_loss = actor_loss.mean() / mask.mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
            
            
        # Train alpha values to satisfy target entropies.
        alpha_losses = {}
        _, new_log_pis_dict = self.actor(hq[0][:,1:-1].detach())
        for key, log_pis in new_log_pis_dict.items():
            alpha_loss = -(self.log_alphas[key] * (log_pis + self.action_dict[key]['target_entropy']))*mask
            alpha_loss = alpha_loss.mean() / mask.mean()
            self.alpha_opt[key].zero_grad()
            alpha_loss.backward()
            self.alpha_opt[key].step()
            self.alphas[key] = torch.exp(self.log_alphas[key])
            alpha_losses[key] = alpha_loss.detach()
            
        
        
        return({
            'total_reward' : total_reward.mean().item(),
            'reward' : reward.mean().item(),
            'critic_losses' : critic_losses,
            'actor_loss' : actor_loss.item(),
            'alpha_losses' : alpha_losses,
            'accuracy_losses' : accuracy_losses,
            'complexity_losses' : complexity_losses,
            'curiosities' : curiosities,
            'imitations' : imitations,
            'alpha_entropies' : alpha_entropies,
            'alpha_normal_entropies' : alpha_normal_entropies,
            'total_entropies' : total_entropies
            })
                                
    

    # These need to be changed!
    def set_eval(self):
        self.world_model.eval()
        self.actor.eval()
        for i in range(len(self.critics)):
            self.critics[i].eval()
            self.critic_targets[i].eval()

    def set_train(self):
        self.world_model.train()
        self.actor.train()
        for i in range(len(self.critics)):
            self.critics[i].train()
            self.critic_targets[i].train()
        
        
        
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
            'accuracy_scalar' : 1,                               
            'complexity_scalar' : 1,                                 
            'eta_before_clamp' : 1,
            'eta' : 1},
        'see_image_2' : {
            'encoder' : Encode_Image,
            'encoder_arg_dict' : {                
                'encode_size' : 16,
                'zp_zq_sizes' : [16]},
            'decoder' : Decode_Image,
            'decoder_arg_dict' : {},
            'accuracy_scalar' : 1,                               
            'complexity_scalar' : 1,  
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
            'target_entropy' : 1,
            'alpha_normal' : 1,
            'delta' : 0}}
    
    
    
    # SOMETHING IS WRONG WITH MULTI-LAYER PVRNN!
    agent = Agent(
        observation_dict = observation_dict,       
        action_dict = action_dict,       
        hidden_state_sizes = [128],
        time_scales = [1],
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
## %%