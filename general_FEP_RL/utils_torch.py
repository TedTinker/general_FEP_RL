#------------------
# util_torchs.py provides some utilities for pytorch models. 
#------------------

from math import exp
import copy

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F



#------------------
# Repeat one tensor to the size of a batch.
#------------------

def tile_batch_dim(
        tensor, 
        batch_size):
    repeat_pattern = [batch_size] + [1] * (len(tensor.shape) - 1)
    return tensor.repeat(*repeat_pattern)
                    


#------------------
# Randomly initiate parameters of a model.
#------------------

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)



#------------------
# Some types of models don't work well with shape (batch_size, steps, ...).
# This function reshapes to (batch_size * steps, ...).
# Assumes tensors for cnn have channels as last dimension, and moves them to front.
#------------------

def model_start(model_input_list):
    new_model_inputs = []
    for model_input, layer_type in model_input_list:
        episodes, steps = model_input.shape[0], model_input.shape[1]
        if layer_type == 'lin':
            model_input = model_input.reshape(episodes * steps, model_input.shape[2])
        elif layer_type == 'cnn':
            model_input = model_input.reshape(episodes * steps, model_input.shape[2], model_input.shape[3], model_input.shape[4]).permute(0, -1, 1, 2)
        elif layer_type == 'recurrent':
            model_input = model_input.reshape(episodes * steps, model_input.shape[2], model_input.shape[3])
        new_model_inputs.append(model_input)
    return episodes, steps, new_model_inputs



#------------------
# This undoes the effects of model_start.
#------------------

def model_end(
        episodes, 
        steps, 
        model_output_list):
    new_model_outputs = []
    for model_output, layer_type in model_output_list:
        if layer_type == 'lin':
            model_output = model_output.reshape(episodes, steps, model_output.shape[-1])
        elif layer_type == 'cnn':
            model_output = model_output.reshape(episodes, steps, model_output.shape[1], model_output.shape[2], model_output.shape[3])
            # THIS IS SUPPOSED TO PERMUTE TOO!
        elif layer_type == 'recurrent':
            model_output = model_output.reshape(episodes, steps, model_output.shape[1], model_output.shape[2])
        new_model_outputs.append(model_output)
    return new_model_outputs
   
            
        
#------------------
# For the reparameterization trick in probabilistic models.
# Make a mean and standard deviation:
# \mu^p_{t,i}, \sigma^p_{t,i} = \textrm{MLP}_i(h^q_{t-1} || a_{t-1})
# Sample:
# z^q_{t,i} \sim q(z_{t,i}) = \mathcal{N}(\mu^q_{t,i},\sigma^q_{t,i})
#------------------
            
def parametrize_normal(
        x,
        mu_func, 
        std_func):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = exp(-20), max = exp(2))
    return mu, std

def sample(mu, std):
    epsilon = torch.randn_like(std)
    return mu + epsilon * std



#------------------
# Apply sampling for probabilistic layer outputs.
#------------------

def recurrent_logprob(
        mu, 
        std):
    if torch.is_tensor(mu):
        std = F.softplus(std)
        std = torch.clamp(std, min = exp(-20), max = exp(2))
        output = sample(mu, std)
        log_prob = Normal(mu, std).log_prob(output)
        log_prob = log_prob.mean(-1).unsqueeze(-1)
        return output, log_prob
    else:
        outputs = []
        log_probs = []
        for sub_mu, sub_std in zip(mu, std):
            output, log_prob = recurrent_logprob(sub_mu, sub_std)
            outputs.append(output)
            log_probs.append(log_prob)
        return outputs, log_probs



#------------------
# Probabilistic layer.
#------------------

class mu_std(nn.Module):
    def __init__(self, mu, entropy = False):
        super().__init__()

        self.entropy = entropy
        self.mu = mu 
        self.std = copy.deepcopy(mu)

    def forward(self, x):
        mu = self.mu(x)
        std = self.std(x)
        output, log_prob = recurrent_logprob(mu, std)
        # log_prob SHOULD BE AVERAGE BASED ON MODEL_START OR NOT!
        if not self.entropy:    # If deterministic, ignore std.
            output = mu
        return output, log_prob
    
    
    
#------------------
# Calculate Kullback-Leibler divergence between the prior and estimated posterior.
# D_{KL}[q(z_{t,i})||p(z_{t,i})]
#------------------

def calculate_dkl(
        mu_1, 
        std_1, 
        mu_2, 
        std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return out
    
    
    
#------------------
# Generate dummy inputs for testing models.
#------------------

def generate_dummy_inputs(
        observation_model_dict, 
        action_model_dict, 
        hidden_state_sizes, 
        batch = 8, 
        steps = 4):
    dummies = {}

    if isinstance(hidden_state_sizes, list):
        dummies['hidden'] = [torch.zeros((batch, steps, hidden_state_size)) for hidden_state_size in hidden_state_sizes]
    else:
        dummies['hidden'] = torch.zeros((batch, steps, hidden_state_sizes))

    if observation_model_dict is not None:
        dummies['obs_enc_in'] = {}
        dummies['obs_enc_out'] = {}
        dummies['obs_dec_in'] = {}
        dummies['obs_dec_out'] = {}

        for key, model in observation_model_dict.items():
            if 'encoder' in model:
                dummies['obs_enc_in'][key] = torch.zeros((batch, steps, *model['encoder'].example_input.shape[2:]))
                dummies['obs_enc_out'][key] = torch.zeros((batch, steps, *model['encoder'].example_output.shape[2:]))
            if 'decoder' in model:
                dummies['obs_dec_in'][key] = torch.zeros((batch, steps, *model['decoder'].example_input.shape[2:]))
                dummies['obs_dec_out'][key] = torch.zeros((batch, steps, *model['decoder'].example_output.shape[2:]))

    if action_model_dict is not None:
        dummies['act_enc_in'] = {}
        dummies['act_enc_out'] = {}
        dummies['act_dec_in'] = {}
        dummies['act_dec_out'] = {}

        for key, model in action_model_dict.items():
            if 'encoder' in model:
                dummies['act_enc_in'][key] = torch.zeros((batch, steps, *model['encoder'].example_input.shape[2:]))
                dummies['act_enc_out'][key] = torch.zeros((batch, steps, *model['encoder'].example_output.shape[2:]))
            if 'decoder' in model:
                dummies['act_dec_in'][key] = torch.zeros((batch, steps, *model['decoder'].example_input.shape[2:]))
                dummies['act_dec_out'][key] = torch.zeros((batch, steps, *model['decoder'].example_output.shape[2:]))

    return dummies