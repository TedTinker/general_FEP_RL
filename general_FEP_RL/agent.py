#%%

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
            hidden_state_size,
            
            observation_dict,       # Keys: observation_names
                                    # Values: 
                                        # encoder
                                            # example_input
                                            # example_output
                                            # out_features
                                        # decoder
                                            # example_input
                                            # example_output
                                            # loss_func
                                        # accuracy_scalar
                                        # complexity_scalar
                                        # beta
                                        # eta
            
            action_dict,            # Keys: action_names
                                    # Values: 
                                        # encoder
                                            # example_input
                                            # example_output
                                            # out_features
                                        # decoder
                                            # example_input
                                            # example_output
                                            # loss_func
                                        # target_entropy
                                        # alpha_normal
            
            number_of_critics = 2, 
            tau = .1,
            
            lr = .0001,
            weight_decay = .00001,
            gamma = .99,
            capacity = 128, 
            max_steps = 32):

        self.observation_dict = observation_dict
        self.action_dict = action_dict
        self.hidden_state_size = hidden_state_size

        self.world_model = World_Model(hidden_state_size, observation_dict, action_dict)
        self.world_model_opt = optim.Adam(self.world_model.parameters(), lr = lr, weight_decay = weight_decay)
                           
        self.actor = Actor(hidden_state_size, action_dict)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr = lr, weight_decay = weight_decay) 
        
        self.alphas = {key : 1 for key in action_dict.keys()} 
        self.log_alphas = {key : torch.tensor([0.0]).requires_grad_() for key in action_dict.keys()}
        self.alpha_opt = {key : optim.Adam(params=[self.log_alphas[key]], lr = lr, weight_decay = weight_decay) for key in action_dict.keys()} 
        
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []
        for _ in range(number_of_critics):
            self.critics.append(Critic(hidden_state_size, action_dict))
            self.critic_targets.append(Critic(hidden_state_size, action_dict))
            self.critic_targets[-1].load_state_dict(self.critics[-1].state_dict())
            self.critic_opts.append(optim.Adam(self.critics[-1].parameters(), lr = lr, weight_decay = weight_decay))
        
        self.tau = tau
        self.gamma = gamma
        self.buffer = RecurrentReplayBuffer(self.world_model.observation_dict, self.world_model.action_dict, capacity, max_steps)
        
        self.begin()
        
        
        
    # I THINK BEGIN AND STEP_IN_EPISODE SHOULD WORK WITH MULTIPLE STEPS
    def begin(self, batch_size = 1):
        self.prev_action = {} 
        for key, value in self.world_model.action_dict.items(): 
            action = torch.zeros_like(self.world_model.action_dict[key]["decoder"].example_output[0, 0].unsqueeze(0).unsqueeze(0))
            self.prev_action[key] = tile_batch_dim(action, batch_size)
        self.hp = torch.zeros((batch_size, 1, self.hidden_state_size)) 
        self.hq = torch.zeros((batch_size, 1, self.hidden_state_size))
        
        
    
    def step_in_episode(self, obs, posterior = True):
        with torch.no_grad():
            self.eval()
            self.hp, self.hq, inner_state_dict, pred_obs_p, pred_obs_q = self.world_model(self.hq if posterior else self.hp, obs, self.prev_action)
            new_action_dict, new_log_prob_dict = self.actor(self.hq if posterior else self.hp) 
            values = []
            for i in range(len(self.critics)):
                value = self.critics[i](self.hq, new_action_dict) 
                values.append(value)
            self.prev_action = new_action_dict
        return {
            "obs" : obs,
            "action" : new_action_dict,
            "log_prob" : new_log_prob_dict,
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
                        
        # Train world_model
        hp, hq, inner_state_dict, pred_obs_p, pred_obs_q = self.world_model(None, obs, complete_action)

        accuracies = {}
        accuracy = torch.zeros((1,)).requires_grad_()
        for key, value in self.observation_dict.items():
            true_obs = obs[key]
            predicted_obs = pred_obs_q[key]
            loss_func = self.observation_dict[key]["decoder"].loss_func
            scalar = self.observation_dict[key]["accuracy_scalar"]
            obs_accuracy = loss_func(true_obs, predicted_obs)
            obs_accuracy = obs_accuracy.mean(dim=tuple(range(2, obs_accuracy.ndim)))
            obs_accuracy = obs_accuracy * complete_mask.squeeze(-1) * scalar
            accuracies[key] = obs_accuracy.mean()
            accuracy = accuracy + obs_accuracy.mean()
            
        complexities = {}
        complexity = torch.zeros((1,)).requires_grad_()
        for key, value in self.observation_dict.items():
            scalar = self.observation_dict[key]["complexity_scalar"]
            inner_state = inner_state_dict[key]
            dkl = inner_state["dkl"].mean(-1).unsqueeze(-1) * complete_mask * scalar
            complexity = complexity + dkl.mean() * self.observation_dict[key]["beta"]
            complexities[key] = dkl[:,1:]
            
        
                                
        self.world_model_opt.zero_grad()
        (accuracy + complexity).backward()
        self.world_model_opt.step()
                

        
        # Get curiosity  
        curiosities = {}
        curiosity = torch.zeros((1,)).requires_grad_()
        for key, value in self.observation_dict.items():
            eta = self.observation_dict[key]["eta"]
            obs_curiosity = torch.clamp(complexities[key], min = 0, max = 1) * eta
            curiosities[key] = obs_curiosity.mean()
            curiosity = curiosity + obs_curiosity
        total_reward = reward + curiosity


                
        # Train critics
        with torch.no_grad():
            new_action_dict, new_log_pis_dict = self.actor(hq.detach())
            for key, new_log_pis in new_log_pis_dict.items():
                new_log_pis_dict[key] = new_log_pis[:,1:]  
                
            Q_target_nexts = []
            for i in range(len(self.critics)):
                Q_target_next = self.critic_targets[i](hq.detach(), new_action_dict)
                Q_target_nexts.append(Q_target_next)                
                        
            Q_target_nexts_stacked = torch.stack(Q_target_nexts, dim=0)
            Q_target_next, _ = torch.min(Q_target_nexts_stacked, dim=0)
            Q_target_next = Q_target_next[:,1:]
            new_entropy = torch.zeros_like(list(new_log_pis_dict.values())[0])
            for key, new_log_pis in new_log_pis_dict.items():
                new_entropy += self.alphas[key] * new_log_pis
            Q_targets = total_reward + self.gamma * (1 - done) * (Q_target_next - new_entropy)  # This might need lists?
        
        critic_losses = []
        for i in range(len(self.critics)):
            Q = self.critics[i](hq[:,:-1].detach(), action)
            critic_loss = 0.5*F.mse_loss(Q*mask, Q_targets*mask)
            critic_losses.append(critic_loss)
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
            for critic_target, critic in zip(self.critic_targets[i].parameters(), self.critics[i].parameters()):
                critic_target.data.copy_(self.tau*critic.data + (1.0-self.tau)*critic_target.data)
                                            
            
        
        # Train actor
        new_action_dict, new_log_pis_dict = self.actor(hq[:,:-1].detach())
        
        Qs = []
        for i in range(len(self.critics)):
            Q = self.critics[i](hq[:,:-1].detach(), new_action_dict)
            Qs.append(Q)
        Qs_stacked = torch.stack(Qs, dim=0)
        Q, _ = torch.min(Qs_stacked, dim=0)
        Q = Q.mean(-1).unsqueeze(-1)
        
        entropy = torch.zeros_like(Q)
        for key in new_action_dict.keys():
            flattened_new_action = new_action_dict[key].flatten(start_dim = 2)
            loc = torch.zeros_like(flattened_new_action, device=flattened_new_action.device).float() 
            scale_tril = torch.eye(flattened_new_action.shape[-1], device=flattened_new_action.device) 
            policy_prior = MultivariateNormal(loc=loc, scale_tril=scale_tril)
            policy_prior_log_prrgbd = policy_prior.log_prob(flattened_new_action).unsqueeze(-1)
            entropy += self.alphas[key] * new_log_pis_dict[key] - self.action_dict[key]["alpha_normal"] * policy_prior_log_prrgbd
            
        actor_loss = (entropy - Q) * mask    
        actor_loss = actor_loss.mean() / mask.mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
            
            
        # Train alpha
        _, new_log_pis_dict = self.actor(hq[:,:-1].detach())
        for key, log_pis in new_log_pis_dict.items():
            alpha_loss = -(self.log_alphas[key] * (log_pis + self.action_dict[key]["target_entropy"]))*mask
            alpha_loss = alpha_loss.mean() / mask.mean()
            self.alpha_opt[key].zero_grad()
            alpha_loss.backward()
            self.alpha_opt[key].step()
            self.alphas[key] = torch.exp(self.log_alphas[key])
            
        
        
        return({
            "reward" : reward.mean(),
            "critic_losses" : critic_losses,
            "actor_loss" : actor_loss,
            "accuracies" : accuracies,
            "complexities" : complexities,
            "curiosities" : curiosities
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
    from general_FEP_RL.encoders.encode_description import Encode_Description
    from general_FEP_RL.decoders.decode_description import Decode_Description
    
    
    
    # This works!
    """observation_dict = {
        "see_image" : {
            "encoder" : Encode_Image,
            "decoder" : Decode_Image,
            "accuracy_scalar" : 1,                               
            "complexity_scalar" : 1,                                 
            "eta" : 1}}
    
    action_dict = {
        "make_image" : {
            "encoder" : Encode_Image,
            "decoder" : Decode_Image,
            "target_entropy" : 1,
            "alpha_normal" : 1}}"""
    
    
    
    # This doesn't!
    observation_dict = {
        "see_description" : {
            "encoder" : Encode_Description,
            "decoder" : Decode_Description,
            "accuracy_scalar" : 1,                               
            "complexity_scalar" : 1,                                 
            "eta" : 1,
            "beta" : 1},
        "see_image" : {
            "encoder" : Encode_Image,
            "decoder" : Decode_Image,
            "accuracy_scalar" : 1,                               
            "complexity_scalar" : 1,                                 
            "eta" : 1,
            "beta" : 1}}
    
    action_dict = {
        "make_description" : {
            "encoder" : Encode_Description,
            "decoder" : Decode_Description,
            "accuracy_scalar" : 1,                               
            "complexity_scalar" : 1,                                 
            "eta" : 1,
            "beta" : 1},
        "make_image" : {
            "encoder" : Encode_Image,
            "decoder" : Decode_Image,
            "target_entropy" : 1,
            "alpha_normal" : 1}}
    
    
    
    agent = Agent(
        hidden_state_size = 128,
        observation_dict = observation_dict,       
        action_dict = action_dict,            
        number_of_critics = 2, 
        tau = .1,
        lr = .0001,
        weight_decay = .00001,
        gamma = .99,
        capacity = 128, 
        max_steps = 32)
    
    dummies = generate_dummy_inputs(agent.world_model.observation_dict, agent.world_model.action_dict, agent.hidden_state_size, batch=1, steps=1)
    dummy_inputs = dummies["obs_enc_in"]
        
    # PROBLEM: How do we make the log_prob (batch * steps, 1) in mu_std?
    agent.step_in_episode(dummy_inputs)
        
        
        
# %%