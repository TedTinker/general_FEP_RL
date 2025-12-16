from math import exp
import copy

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F



# Expanding tensor to batch size.
def tile_batch_dim(tensor, batch_size):
    shape = list(tensor.shape)
    repeat_pattern = [batch_size] + [1] * (len(shape) - 1)
    return tensor.repeat(*repeat_pattern)

def print_step_in_episode(step_dict):
        for key, value in step_dict.items():
            if(type(value) == dict):
                for sub_key, sub_value in value.items():
                    if(type(sub_value) == torch.Tensor):
                        print(f"{key}:\t{sub_key}:\t{list(sub_value.shape)}")
                    elif(type(sub_value) == dict):
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            print(f"{key}:\t{sub_key}:\t{sub_sub_key}:\t{list(sub_sub_value.shape)}")
                    else:
                        print(f"{key}:\t{type(sub_value)}")
            elif(type(value) == list):
                for sub_value in value:
                    print(f"{key}:\t{list(sub_value.shape)}")
                    
"""class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, antialias=False):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias
        )
    
# Add position layers.
def add_position_layers(x, learned_pos, scale = 1):
    pos = learned_pos.repeat(x.shape[0], 1, 1, 1)
    pos = F.interpolate(pos, scale_factor = scale, mode = "bilinear", align_corners = True)
    x = torch.cat([x, pos], dim = 1)
    return(x)"""
    
    

# Calculating Kullback-Leibler divergence.
def calculate_dkl(mu_1, std_1, mu_2, std_2):
    std_1 = std_1**2
    std_2 = std_2**2
    term_1 = (mu_2 - mu_1)**2 / std_2 
    term_2 = std_1 / std_2 
    term_3 = torch.log(term_2)
    out = (.5 * (term_1 + term_2 - term_3 - 1))
    out = torch.nan_to_num(out)
    return(out)



def model_start(model_input_list, recurrent = False):
    new_model_inputs = []
    for model_input, layer_type in model_input_list:
        episodes, steps = model_input.shape[0], model_input.shape[1]
        if(layer_type == "lin"):
            model_input = model_input.reshape(episodes * steps, model_input.shape[2])
        if(layer_type == "cnn"):
            model_input = model_input.reshape(episodes * steps, model_input.shape[2], model_input.shape[3], model_input.shape[4]).permute(0, -1, 1, 2)
        if(layer_type == "text"):
            model_input = model_input.reshape(episodes * steps, model_input.shape[2], model_input.shape[3])
        new_model_inputs.append(model_input)
    return episodes, steps, new_model_inputs



def model_end(episodes, steps, model_output_list, recurrent = False):
    new_model_outputs = []
    for model_output, layer_type in model_output_list:
        if(layer_type == "lin"):
            model_output = model_output.reshape(episodes, steps, model_output.shape[-1])
        if(layer_type == "cnn"):
            model_output = model_output.reshape(episodes, steps, model_output.shape[1], model_output.shape[2], model_output.shape[3])
        new_model_outputs.append(model_output)
    return new_model_outputs



def generate_dummy_inputs(obs_dict, act_dict, hidden_state_sizes, batch=8, steps=4):
    dummies = {}

    if(type(hidden_state_sizes) == list):
        dummies["hidden"] = [torch.zeros((batch, steps, hidden_state_size)) for hidden_state_size in hidden_state_sizes]
    else:
        dummies["hidden"] = torch.zeros((batch, steps, hidden_state_sizes))

    if obs_dict is not None:
        dummies["obs_enc_in"] = {}
        dummies["obs_enc_out"] = {}
        dummies["obs_dec_in"] = {}
        dummies["obs_dec_out"] = {}

        for key, value in obs_dict.items():
            if "encoder" in value:
                dummies["obs_enc_in"][key] = torch.zeros((batch, steps, *value["encoder"].example_input.shape[2:]))
                dummies["obs_enc_out"][key] = torch.zeros((batch, steps, *value["encoder"].example_output.shape[2:]))
            if "decoder" in value:
                dummies["obs_dec_in"][key] = torch.zeros((batch, steps, *value["decoder"].example_input.shape[2:]))
                dummies["obs_dec_out"][key] = torch.zeros((batch, steps, *value["decoder"].example_output.shape[2:]))

    if act_dict is not None:
        dummies["act_enc_in"] = {}
        dummies["act_enc_out"] = {}
        dummies["act_dec_in"] = {}
        dummies["act_dec_out"] = {}

        for key, value in act_dict.items():
            if "encoder" in value:
                dummies["act_enc_in"][key] = torch.zeros((batch, steps, *value["encoder"].example_input.shape[2:]))
                dummies["act_enc_out"][key] = torch.zeros((batch, steps, *value["encoder"].example_output.shape[2:]))
            if "decoder" in value:
                dummies["act_dec_in"][key] = torch.zeros((batch, steps, *value["decoder"].example_input.shape[2:]))
                dummies["act_dec_out"][key] = torch.zeros((batch, steps, *value["decoder"].example_output.shape[2:]))

    return dummies



# For starting neural networks.
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm2d)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
            
# How to use mean and standard deviation layers.
def var(x, mu_func, std_func):
    mu = mu_func(x)
    std = torch.clamp(std_func(x), min = exp(-20), max = exp(2))
    return(mu, std)

# How to sample from probability distributions.
def sample(mu, std):
    e = Normal(0, 1).sample(std.shape).to(std.device)
    return(mu + e * std)



def recurrent_logprob(mu, std):
    if(torch.is_tensor(mu)):
        std = F.softplus(std)
        std = torch.clamp(std, min = exp(-20), max = exp(2))
        output = sample(mu, std)
        log_prob = Normal(mu, std).log_prob(output)
        log_prob = log_prob.mean(-1).unsqueeze(-1)
        return(output, log_prob)
    else:
        outputs = []
        log_probs = []
        for sub_mu, sub_std in zip(mu, std):
            output, log_prob = recurrent_logprob(sub_mu, sub_std)
            outputs.append(output)
            log_probs.append(log_prob)
        return(outputs, log_probs)



# Make output with entropy.
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
        if(not self.entropy):
            output = mu
        if log_prob.ndim > 2:
            log_prob = log_prob.mean(dim=tuple(range(2, log_prob.ndim))).unsqueeze(-1)
        return output, log_prob