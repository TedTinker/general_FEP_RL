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
            self.hp, self.hq, inner_state_dict, pred_obs_p, pred_obs_q = self.world_model(
                self.hq if posterior else self.hp, obs, self.prev_action, one_step = True)
            action, log_prob = self.actor(self.hq if posterior else self.hp) 
            self.prev_action = action
            values = []
            for i in range(len(self.critics)):
                value = self.critics[i](self.hq, action) 
                values.append(value)
        return {
            "obs" : obs,
            "action" : action,
            "log_prob" : log_prob,
            "values" : values,
            "inner_state_dict" : inner_state_dict,
            "pred_obs_p" : pred_obs_p,
            "pred_obs_q" : pred_obs_q}
        
        
        
    def epoch(self, batch_size):
        self.train()
    
        batch = self.buffer.sample(batch_size)
    
        obs = batch["obs"]          # dict[B, T+1]
        action = batch["action"]    # dict[B, T]
        reward = batch["reward"]    # [B,T,1]
        done = batch["done"]        # [B,T,1]
        mask = batch["mask"]        # [B,T,1]
    
        # World model uses obs[0..T-1] and action[0..T-1]
        obs_in = {k: v[:, :-1] for k, v in obs.items()}  # [B,T]
        obs_target = {k: v[:, 1:] for k, v in obs.items()}  # [B,T]
    
        hp, hq, inner_state_dict, pred_p, pred_q = \
            self.world_model(None, obs_in, action, one_step=False)
    
        # ---------- WORLD MODEL LOSS ----------
        accuracies = {}
        complexities = {}
    
        total_acc = torch.zeros(1, device=hq.device, requires_grad=True)
        total_cpx = torch.zeros(1, device=hq.device, requires_grad=True)
    
        for key in self.observation_dict:
    
            # accuracy
            loss_fn = self.observation_dict[key]["decoder"].loss_func
            scalar = self.observation_dict[key]["accuracy_scalar"]
    
            acc = loss_fn(pred_q[key], obs_target[key])    # [B,T,...]
            acc = acc.mean(dim=tuple(range(2, acc.ndim)))  # [B,T]
            acc = acc * mask.squeeze(-1) * scalar
    
            accuracies[key] = acc.mean().item()
            total_acc = total_acc + acc.mean()
    
            # complexity (KL)
            KL = inner_state_dict[key]["dkl"].mean(-1).unsqueeze(-1)  # [B,T,1]
            beta = self.observation_dict[key]["beta"]
    
            cpx = KL * mask * beta
            complexities[key] = cpx.mean().item()
            total_cpx = total_cpx + cpx.mean()
    
        self.world_model_opt.zero_grad()
        (total_acc + total_cpx).backward()
        self.world_model_opt.step()
    
        # ---------- CURIOSITY ----------
        curiosities = {}
        total_curi = 0
    
        for key in self.observation_dict:
            KL = inner_state_dict[key]["dkl"].mean(-1).unsqueeze(-1)  # [B,T,1]
            eta = self.observation_dict[key]["eta"]
    
            curi = torch.clamp(KL, 0, 1) * eta
            curiosities[key] = curi.mean().item()
            total_curi += curi
    
        total_reward = reward + total_curi
    
        # ---------- CRITIC TARGET (SAC) ----------
        with torch.no_grad():
            new_action, new_logpi = self.actor(hq.detach())  # [B,T+1]
    
            # align to [B,T]
            new_logpi_T = {k: v[:, :-1] for k, v in new_logpi.items()}
    
            Q_next_list = []
            for tgt in self.critic_targets:
                q = tgt(hq.detach(), new_action)  # [B,T+1,1]
                Q_next_list.append(q[:, 1:])     # [B,T,1]
    
            Q_next = torch.stack(Q_next_list, dim=0).min(dim=0)[0]
    
            entropy = torch.zeros_like(Q_next)
            for key in new_logpi_T:
                entropy += self.alphas[key] * new_logpi_T[key]
    
            Q_target = total_reward + self.gamma * (1 - done) * (Q_next - entropy)
    
        # ---------- CRITIC UPDATES ----------
        critic_losses = []
        for i, critic in enumerate(self.critics):
            Q = critic(hq[:, :-1].detach(), action)         # [B,T,1]
            loss = 0.5 * F.mse_loss(Q * mask, Q_target * mask)
    
            critic_losses.append(loss.item())
            self.critic_opts[i].zero_grad()
            loss.backward()
            self.critic_opts[i].step()
    
            # Soft update
            for p_tgt, p in zip(self.critic_targets[i].parameters(), critic.parameters()):
                p_tgt.data.copy_(self.tau * p.data + (1 - self.tau) * p_tgt.data)
    
        # ---------- ACTOR UPDATE ----------
        new_action, new_logpi = self.actor(hq[:, :-1].detach())
    
        Qs = [critic(hq[:, :-1].detach(), new_action) for critic in self.critics]
        Qs = torch.stack(Qs, dim=0)
        Q = torch.min(Qs, dim=0)[0]  # [B,T,1]
    
        entropy = torch.zeros_like(Q)
        for key in new_logpi:
            entropy += self.alphas[key] * new_logpi[key]
    
        actor_loss = (entropy - Q) * mask
        actor_loss = actor_loss.mean() / mask.mean()
    
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
    
        # ---------- ALPHA TUNING ----------
        alpha_losses = {}
        _, new_logpi = self.actor(hq[:, :-1].detach())
    
        for key, logpi in new_logpi.items():
            a_loss = -(self.log_alphas[key] * (logpi + self.action_dict[key]["target_entropy"])) * mask
            a_loss = a_loss.mean() / mask.mean()
    
            self.alpha_opt[key].zero_grad()
            a_loss.backward()
            self.alpha_opt[key].step()
    
            self.alphas[key] = torch.exp(self.log_alphas[key])
            alpha_losses[key] = a_loss.item()
    
        # ---------- RETURN EXACT ORIGINAL DICT ----------
        return {
            "total_reward": total_reward.mean().item(),
            "reward": reward.mean().item(),
            "critic_losses": critic_losses,
            "actor_loss": actor_loss.item(),
            "alpha_losses": alpha_losses,
            "accuracies": accuracies,
            "complexities": complexities,
            "curiosities": curiosities,
            "entropies": {k: entropy.mean().item() for k in new_logpi},
        }
                                
    

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
        
    agent.step_in_episode(dummy_inputs)
        
        
    agent.epoch(64)
# %%