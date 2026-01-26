#%% 
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
        self.time_scales = time_scales
        self.beta_hidden = beta_hidden
        self.eta_before_clamp = eta_before_clamp
        self.eta = eta
        self.tau = tau
        self.gamma = gamma

        # World model.
        self.world_model = World_Model(hidden_state_sizes, observation_dict, action_dict, time_scales)
        self.world_model_opt = optim.Adam(self.world_model.parameters(), lr = lr, weight_decay = weight_decay)
        self.world_model = torch.compile(self.world_model)
                           
        # Actor.
        self.actor = Actor(hidden_state_sizes[0], action_dict)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr = lr, weight_decay = weight_decay) 
        self.actor = torch.compile(self.actor)
        
        # Alpha values (entropy hyperparameter).
        self.alphas = {key : 1 for key in action_dict.keys()} 
        self.log_alphas = nn.ParameterDict({
            key: nn.Parameter(torch.zeros((1,)))
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
            self.critics[-1] = torch.compile(self.critics[-1])
            self.critic_targets[-1] = torch.compile(self.critic_targets[-1])
        
        # Recurrent replay buffer.
        self.buffer = RecurrentReplayBuffer(
            self.world_model.observation_model_dict, 
            self.world_model.action_model_dict, 
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
    def step_in_episode(self, obs, posterior = True, best_action = None, eval = False):
        with torch.no_grad():
            if best_action is not None:
                self.action = best_action
            if eval:
                self.set_eval()
            else:
                self.set_train()
            self.hp, self.hq, inner_state_dict = self.world_model(
                self.hq if posterior else self.hp, obs, self.action, one_step = True)
            self.hp = [h.detach() for h in self.hp]
            self.hq = [h.detach() for h in self.hq]
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
        complete_mask = torch.cat([torch.zeros_like(mask[:, :1]), mask], dim=1)


                                    
        # Train world model to minimize Free Energy.
        hp, hq, inner_state_dict, pred_obs_p, pred_obs_q = self.world_model(None, obs, complete_action)
        
        # hp and hq steps: 
        #   t = -1, 0, 1, ..., n+1
        
        
        
        # Accuracy of observation prediction.
        # Given T steps and i = 0, ..., n parts of observations,
        # Accuracy = E_{q(z_{0:T,i})}[log p(o_{1:T+1,i}|z_{0:T,i})]) * mask_{0:T}
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
            accuracy_loss = accuracy_loss + obs_accuracy_loss.mean()
            accuracy_losses[key] = obs_accuracy_loss.mean().item()
            
        # Complexity of predictions.
        # Given T steps and i = 0, ..., n parts of observations,
        # Complexity = beta_i DKL[q(z_{0:T,i}) || p(z_{0:T,i})] * mask_{0:T}
        complexity_losses = {}
        complexity_loss = 0
        for key, value in self.observation_dict.items():
            dkl = inner_state_dict[key]['dkl'].mean(-1).unsqueeze(-1) * complete_mask
            complexity = dkl * self.observation_dict[key]['beta_obs']
            complexity_loss = complexity_loss + complexity.mean() / mask.sum()
            complexity_losses[key] = complexity[:,1:]

        for i, beta in enumerate(self.beta_hidden):
            dkl = inner_state_dict[i+1]['dkl'].mean(-1).unsqueeze(-1) * complete_mask 
            complexity = dkl * beta
            complexity_loss = complexity_loss + complexity.mean() / mask.sum()
            complexity_losses[f'hidden_layer_{i+2}'] = complexity[:,1:]
                        
        
                                
        # Minimize Free Energy = Complexity - Accuracy
        self.world_model_opt.zero_grad()
        (accuracy_loss + complexity_loss).backward()
        self.world_model_opt.step()
                

        
        # Get curiosity values based on complexity of next step.
        curiosities = {}
        curiosity = torch.zeros_like(reward)
                
        for key, value in self.observation_dict.items():
            obs_curiosity = self.observation_dict[key]['eta'] * \
                torch.clamp(complexity_losses[key] * self.observation_dict[key]['eta_before_clamp'], min = 0, max = 1)
            curiosity = curiosity + obs_curiosity
            complexity_losses[key] = complexity_losses[key].mean().item() # Replace tensor with scalar for plotting.
            curiosities[key] = obs_curiosity.mean().item()
            
        for i in range(len(self.hidden_state_sizes) - 1):
            obs_curiosity = self.eta[i] * \
                torch.clamp(complexity_losses[f'hidden_layer_{i+2}'] * self.eta_before_clamp[i], min = 0, max = 1)
            curiosity = curiosity + obs_curiosity
            complexity_losses[f'hidden_layer_{i+2}'] = complexity_losses[f'hidden_layer_{i+2}'].mean().item() # Replace tensor with scalar for plotting.
            curiosities[f'hidden_layer_{i+2}'] = obs_curiosity.mean().item() 
            
            
            
        # The actor and critics are concerned with both extrinsic rewards and intrinsic rewards in Expected Free Energy.
        # G(o_t, a_t) = 
        #   -DKL[q(z_t | o_t, h_{t-1}) || p(z_t | h_{t-1})]     (Curiosity)
        #   -r(s_t, a_t)}                                       (Extrinsic Reward)
        #   -H(\pi_\phi(a_t | o_t))                             (Entropy)
        #   -E_{\pi_\phi(a_t | o_t)} [\log p(a_t^\ast | o_t)]   (Imitation)
        total_reward = (reward + curiosity).detach() * mask
        
        hq_all = hq[0].detach()                 # (B, T+2, H)
        h_t    = hq_all[:, 1:-1]                # (B, T,   H)  -> h_t
        h_tp1  = hq_all[:, 2:]                  # (B, T,   H)  -> h_{t+1}
        
        
        
        # Target critics make target Q-values.
        with torch.no_grad():

            # Next-step action and log-prob.
            a_tp1, logp_tp1 = self.actor(h_tp1.detach())
        
            # Target critics evaluate next-step value.
            Q_tp1_list = [critic_tgt(h_tp1.detach(), a_tp1) for critic_tgt in self.critic_targets]
            Q_tp1 = torch.min(torch.stack(Q_tp1_list, dim=0), dim=0)[0]
            # Q_tp1 shape: (B, T, 1)
        
            # Entropy bonus (rewarded entropy).
            entropy_bonus_tp1 = torch.zeros_like(Q_tp1)
            for k, lp in logp_tp1.items():
                # lp = log π(a|h) <= 0
                # -lp = entropy bonus >= 0
                entropy_bonus_tp1 += self.alphas[k] * (-lp)
        
            # Bellman target (EFE-style).
            Q_target = total_reward + self.gamma * (1.0 - done) * (Q_tp1 + entropy_bonus_tp1)
        
            # Mask invalid timesteps.
            Q_target = Q_target * mask
        
        
        # Train critics to match Q_target
        critic_losses = []
        
        for i, critic in enumerate(self.critics):
            Q_pred = critic(h_t.detach(), action)
            Q_pred = Q_pred * mask
            critic_loss = 0.5 * ((Q_pred - Q_target) ** 2).sum() / mask.sum()
            critic_losses.append(critic_loss.item())
        
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()
        
            # Soft-update target critic with polyak averaging.
            for tgt_p, p in zip(
                self.critic_targets[i].parameters(),
                critic.parameters()):
                tgt_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * tgt_p.data)
        
        
        
        # Train agent to minimize expected free energy.
        new_action_dict, new_log_pis_dict, imitation_loss = self.actor(h_t.detach(), best_action)
        
        Q_list = [critic(h_t.detach(), new_action_dict) for critic in self.critics]
        Q = torch.min(torch.stack(Q_list, dim=0), dim=0)[0]
        #Q = Q.mean(-1, keepdim=True)
        
        alpha_entropies = {}
        alpha_normal_entropies = {}
        total_entropies = {}
        entropy = torch.zeros_like(Q)
        
        for k in new_action_dict.keys():
            lp = new_log_pis_dict[k]
            alpha_entropy = self.alphas[k] * (-lp)
        
            flat_a = new_action_dict[k].flatten(start_dim=2)
            alpha_normal_entropy = (
                0.5 * self.action_dict[k]['alpha_normal']
                * (flat_a ** 2).sum(-1, keepdim=True))
        
            total_entropy = alpha_entropy - alpha_normal_entropy
            entropy += total_entropy
            
            alpha_entropies[k] = alpha_entropy.mean().item()
            alpha_normal_entropies[k] = alpha_normal_entropy.mean().item()  
            total_entropies[k] = total_entropy.mean().item()

        
        imitations = {}
        total_imitation_loss = torch.zeros_like(Q)
        
        for k in new_action_dict.keys():
            scalar = self.action_dict[k]['delta']
            il = imitation_loss[k].mean(-1, keepdim=True) * scalar * mask * best_action_mask
            total_imitation_loss += il
            imitations[k] = il.mean().item()
        
        actor_loss = (entropy - Q - total_imitation_loss) * mask
        actor_loss = actor_loss.sum() / mask.sum()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        
        
        # Train alpha values.
        alpha_losses = {}
        _, logp_t = self.actor(h_t.detach())
        
        for k, lp in logp_t.items():
            alpha_loss = -(self.log_alphas[k] * (lp + self.action_dict[k]['target_entropy']).detach()) * mask
            alpha_loss = alpha_loss.sum() / mask.sum()
        
            self.alpha_opt[k].zero_grad()
            alpha_loss.backward()
            self.alpha_opt[k].step()
        
            self.alphas[k] = torch.exp(self.log_alphas[k])
            alpha_losses[k] = alpha_loss.detach()
            
        
        
        return({
            'obs' : {k: v.detach().cpu() for k, v in obs.items()},
            'pred_obs_p': {k: v.detach().cpu() for k, v in pred_obs_p.items()},
            'pred_obs_q': {k: v.detach().cpu() for k, v in pred_obs_q.items()},
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
                                
    

    # These should also make actor use the mu without std.
    def set_eval(self):
        self.world_model.eval()
        self.world_model.use_sample = False
        
        self.actor.eval() 
        for key, module in self.actor.action_model_dict.items():            
            module["decoder"].mu_std.eval = True
                
        for i in range(len(self.critics)):
            self.critics[i].eval()
            self.critic_targets[i].eval()

    def set_train(self):
        self.world_model.train()
        self.world_model.use_sample = True
        
        self.actor.train()
        for key, module in self.actor.action_model_dict.items():            
            module["decoder"].mu_std.eval = False
        
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
            k: v.detach().cpu()
            for k, v in self.alphas.items()}
        state["log_alphas"] = {
            k: v.detach().cpu()
            for k, v in self.log_alphas.items()}
    
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
            for k in self.alphas:
                if k in state["alphas"]:
                    self.alphas[k] = state["alphas"][k]
    
        if "log_alphas" in keys and "log_alphas" in state:
            for k in self.log_alphas:
                if k in state["log_alphas"]:
                    self.log_alphas[k].data.copy_(state["log_alphas"][k])
    
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