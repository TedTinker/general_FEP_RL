#%%

### TO DO:
    # Make hidden states a list.
    # Use however many levels there are.

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim

from general_FEP_RL.utils_torch import tile_batch_dim
from general_FEP_RL.buffer import RecurrentReplayBuffer
from general_FEP_RL.world_model import World_Model
from general_FEP_RL.actor_critic import Actor, Critic



class Agent:
    
    def __init__(
            self, 
            hidden_state_sizes,
            
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
                                        # beta (complexity scalar)
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
            
            time_scales = [1],
            
            number_of_critics = 2, 
            tau = .1,
            
            lr = .0001,
            weight_decay = .00001,
            gamma = .99,
            capacity = 128, 
            max_steps = 32):

        self.observation_dict = observation_dict
        self.action_dict = action_dict
        self.hidden_state_sizes = hidden_state_sizes

        self.world_model = World_Model(hidden_state_sizes, observation_dict, action_dict, time_scales)
        self.world_model_opt = optim.Adam(self.world_model.parameters(), lr = lr, weight_decay = weight_decay)
                           
        self.actor = Actor(hidden_state_sizes[0], action_dict)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr = lr, weight_decay = weight_decay) 
        
        self.alphas = {key : 1 for key in action_dict.keys()} 
        self.log_alphas = {key : torch.tensor([0.0]).requires_grad_() for key in action_dict.keys()}
        self.alpha_opt = {key : optim.Adam(params=[self.log_alphas[key]], lr = lr, weight_decay = weight_decay) for key in action_dict.keys()} 
        
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []
        for _ in range(number_of_critics):
            self.critics.append(Critic(hidden_state_sizes[0], action_dict))
            self.critic_targets.append(Critic(hidden_state_sizes[0], action_dict))
            self.critic_targets[-1].load_state_dict(self.critics[-1].state_dict())
            self.critic_opts.append(optim.Adam(self.critics[-1].parameters(), lr = lr, weight_decay = weight_decay))
        
        self.tau = tau
        self.gamma = gamma
        self.buffer = RecurrentReplayBuffer(self.world_model.observation_dict, self.world_model.action_dict, capacity, max_steps)
        
        self.begin()
        
        
        
    def begin(self, batch_size = 1):
        self.action = {} 
        for key, value in self.world_model.action_dict.items(): 
            action = torch.zeros_like(self.world_model.action_dict[key]["decoder"].example_output[0, 0].unsqueeze(0).unsqueeze(0))
            self.action[key] = tile_batch_dim(action, batch_size)
        self.hp = [torch.zeros((batch_size, 1, hidden_state_size)) for hidden_state_size in self.hidden_state_sizes] 
        self.hq = [torch.zeros((batch_size, 1, hidden_state_size)) for hidden_state_size in self.hidden_state_sizes] 
        
        
    
    def step_in_episode(self, obs, posterior = True):
        with torch.no_grad():
            self.eval()
            # THIS SEEMS TO CARE ABOUT THE ORDER OF OBSERVATIONS!
            # I believe this is because of making lists with .keys, .values, or .items
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
            "obs" : obs,
            "action" : self.action,
            "log_prob" : log_prob,
            "values" : values,
            "inner_state_dict" : inner_state_dict,
            "pred_obs_p" : pred_obs_p,
            "pred_obs_q" : pred_obs_q}
        
        

    def epoch(self, batch_size):
        self.train()
                                
        batch = self.buffer.sample(batch_size)
        obs = batch["obs"]
        action = batch["action"] 
        reward = batch["reward"]
        done = batch["done"]
        mask = batch["mask"]
        complete_mask = torch.cat([torch.ones(mask.shape[0], 1, 1), mask], dim = 1)
        
        complete_action = {}
        for key, value in action.items(): 
            empty_action = torch.zeros_like(self.world_model.action_dict[key]["decoder"].example_output[0, 0].unsqueeze(0).unsqueeze(0))
            empty_action = tile_batch_dim(empty_action, batch_size)
            complete_action[key] = torch.cat([empty_action, value], dim = 1)
                                    
        hp, hq, inner_state_dict, pred_obs_p, pred_obs_q = self.world_model(None, obs, complete_action)
        
        accuracy_losses = {}
        accuracy_loss = torch.zeros((1,)).requires_grad_()
        for key, value in self.observation_dict.items():
            true_obs = obs[key][:, 1:]
            predicted_obs = pred_obs_q[key]
            loss_func = self.observation_dict[key]["decoder"].loss_func
            scalar = self.observation_dict[key]["accuracy_scalar"]
            obs_accuracy_loss = loss_func(true_obs, predicted_obs)
            obs_accuracy_loss = obs_accuracy_loss.mean(dim=tuple(range(2, obs_accuracy_loss.ndim)))
            obs_accuracy_loss = obs_accuracy_loss * mask.squeeze(-1) * scalar
            accuracy_losses[key] = obs_accuracy_loss.mean().item()
            accuracy_loss = accuracy_loss + obs_accuracy_loss.mean()
            
        complexity_losses = {}
        complexity_loss = torch.zeros((1,)).requires_grad_()
        for key, value in self.observation_dict.items():
            dkl = inner_state_dict[key]["dkl"].mean(-1).unsqueeze(-1) * complete_mask
            complexity_loss = complexity_loss + dkl.mean() * self.observation_dict[key]["beta"]
            complexity_losses[key] = complexity_loss
            
        
                                
        self.world_model_opt.zero_grad()
        (accuracy_loss + complexity_loss).backward()
        self.world_model_opt.step()
                

        
        # Get curiosity  
        curiosities = {}
        curiosity = torch.zeros((1,)).requires_grad_()
        for key, value in self.observation_dict.items():
            obs_curiosity = self.observation_dict[key]["eta"] * \
                torch.clamp(complexity_losses[key] * self.observation_dict[key]["eta_before_clamp"], min = 0, max = 1) 
            complexity_losses[key] = complexity_losses[key].mean().item()
            curiosities[key] = obs_curiosity.mean().item()
            curiosity = curiosity + obs_curiosity
        total_reward = reward + curiosity


                
        # Train critics
        with torch.no_grad():
            new_action_dict, new_log_pis_dict = self.actor(hq[0].detach())
            for key, new_log_pis in new_log_pis_dict.items():
                new_log_pis_dict[key] = new_log_pis[:,1:]  
                
            Q_target_nexts = []
            for i in range(len(self.critics)):
                Q_target_next = self.critic_targets[i](hq[0].detach(), new_action_dict)
                Q_target_nexts.append(Q_target_next)                
                        
            Q_target_nexts_stacked = torch.stack(Q_target_nexts, dim=0)
            Q_target_next, _ = torch.min(Q_target_nexts_stacked, dim=0)
            Q_target_next = Q_target_next[:,1:]
            new_entropy = torch.zeros_like(list(new_log_pis_dict.values())[0])
            for key, new_log_pis in new_log_pis_dict.items():
                new_entropy += self.alphas[key] * new_log_pis
            Q_targets = total_reward + self.gamma * (1 - done) * (Q_target_next - new_entropy) 
        
        critic_losses = []
        for i in range(len(self.critics)):
            Q = self.critics[i](hq[:,:-1].detach(), action)
            critic_loss = 0.5*F.mse_loss(Q*mask, Q_targets*mask)
            critic_losses.append(critic_loss.item())
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
            for critic_target, critic in zip(self.critic_targets[i].parameters(), self.critics[i].parameters()):
                critic_target.data.copy_(self.tau*critic.data + (1.0-self.tau)*critic_target.data)
                                            
            
        
        # Train actor
        new_action_dict, new_log_pis_dict = self.actor(hq[0][:,:-1].detach())
        
        Qs = []
        for i in range(len(self.critics)):
            Q = self.critics[i](hq[0][:,:-1].detach(), new_action_dict)
            Qs.append(Q)
        Qs_stacked = torch.stack(Qs, dim=0)
        Q, _ = torch.min(Qs_stacked, dim=0)
        Q = Q.mean(-1).unsqueeze(-1)
        
        alpha_entropies = {}
        alpha_normal_entropies = {}
        total_entropies = {}
        complete_entropy = torch.zeros_like(Q)
        for key in new_action_dict.keys():
            flattened_new_action = new_action_dict[key].flatten(start_dim = 2)
            loc = torch.zeros_like(flattened_new_action, device=flattened_new_action.device).float() 
            scale_tril = torch.eye(flattened_new_action.shape[-1], device=flattened_new_action.device) 
            policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
            policy_prior_log_prrgbd = policy_prior.log_prob(flattened_new_action).unsqueeze(-1)
            
            alpha_entropy = self.alphas[key] * new_log_pis_dict[key]
            alpha_normal_entropy = -self.action_dict[key]["alpha_normal"] * policy_prior_log_prrgbd
            total_entropy = alpha_entropy + alpha_normal_entropy
            
            alpha_entropies[key] = alpha_entropy.mean().item()
            alpha_normal_entropies[key] = alpha_normal_entropy.mean().item()
            total_entropies[key] = total_entropy.mean().item()
            
            complete_entropy += total_entropy 
            
        actor_loss = (complete_entropy - Q) * mask    
        actor_loss = actor_loss.mean() / mask.mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
            
            
        # Train alpha
        alpha_losses = {}
        _, new_log_pis_dict = self.actor(hq[:,:-1].detach())
        for key, log_pis in new_log_pis_dict.items():
            alpha_loss = -(self.log_alphas[key] * (log_pis + self.action_dict[key]["target_entropy"]))*mask
            alpha_loss = alpha_loss.mean() / mask.mean()
            self.alpha_opt[key].zero_grad()
            alpha_loss.backward()
            self.alpha_opt[key].step()
            self.alphas[key] = torch.exp(self.log_alphas[key])
            alpha_losses[key] = alpha_loss.detach()
            
        
        
        return({
            "total_reward" : total_reward.mean().item(),
            "reward" : reward.mean().item(),
            "critic_losses" : critic_losses,
            "actor_loss" : actor_loss.item(),
            "alpha_losses" : alpha_losses,
            "accuracy_losses" : accuracy_losses,
            "complexity_losses" : complexity_losses,
            "curiosities" : curiosities,
            "alpha_entropies" : alpha_entropies,
            "alpha_normal_entropies" : alpha_normal_entropies,
            "total_entropies" : total_entropies
            })
                                
    

    # These need to be changed!
    def eval(self):
        self.world_model.eval()
        self.actor.eval()
        for i in range(len(self.critics)):
            self.critics[i].eval()
            self.critic_targets[i].eval()

    def train(self):
        self.world_model.train()
        self.actor.train()
        for i in range(len(self.critics)):
            self.critics[i].train()
            self.critic_targets[i].train()
        
        
        
if __name__ == "__main__":
    
    from general_FEP_RL.utils_torch import generate_dummy_inputs
    from general_FEP_RL.encoders.encode_image import Encode_Image
    from general_FEP_RL.decoders.decode_image import Decode_Image
    
    
    
    observation_dict = {
        "see_image" : {
            "encoder" : Encode_Image,
            "encoder_arg_dict" : {                
                "encode_size" : 256,
                "zp_zq_sizes" : [256]},
            "decoder" : Decode_Image,
            "decoder_arg_dict" : {},
            "accuracy_scalar" : 1,                               
            "complexity_scalar" : 1,                                 
            "eta_before_clamp" : 1,
            "eta" : 1},
        "see_image_2" : {
            "encoder" : Encode_Image,
            "encoder_arg_dict" : {                
                "encode_size" : 16,
                "zp_zq_sizes" : [16]},
            "decoder" : Decode_Image,
            "decoder_arg_dict" : {},
            "accuracy_scalar" : 1,                               
            "complexity_scalar" : 1,  
            "eta_before_clamp" : 1,                               
            "eta" : 1}}
    
    action_dict = {
        "make_image" : {
            "encoder" : Encode_Image,
            "encoder_arg_dict" : {                
                "encode_size" : 128,
                "zp_zq_sizes" : [128]},
            "decoder" : Decode_Image,
            "decoder_arg_dict" : {},
            "target_entropy" : 1,
            "alpha_normal" : 1}}
    
    
    
    agent = Agent(
        hidden_state_sizes = [128, 128],
        observation_dict = observation_dict,       
        action_dict = action_dict,            
        number_of_critics = 2, 
        tau = .99,
        lr = .0001,
        weight_decay = .00001,
        gamma = .99,
        capacity = 128, 
        max_steps = 32)
    
    dummies = generate_dummy_inputs(agent.world_model.observation_dict, agent.world_model.action_dict, agent.hidden_state_sizes, batch=1, steps=1)
    dummy_inputs = dummies["obs_enc_in"]
        
    agent.step_in_episode(dummy_inputs)
        
        
    agent.world_model.summary()
# %%