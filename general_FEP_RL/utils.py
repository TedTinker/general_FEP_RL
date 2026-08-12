#%%
#------------------
# utils.py provides some basic utilities.
#------------------

import datetime  
import random
import matplotlib
import numpy as np

import torch 
torch.set_default_device("cpu")



#------------------
# Set pytorch device. (Right now, only cpu is supported.)
#------------------

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('\n\nDevice: {}.\n\n'.format(device))



#------------------
# Set random seed. (Some seeds may be missing.)
#------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(777)



def sample(mu, std):
    epsilon = torch.randn_like(std)
    return mu + epsilon * std



#------------------
# Functions for durations.
#------------------

start_time = datetime.datetime.now()

def duration(start_time = start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return change_time

def estimate_total_duration(
        proportion_completed, 
        start_time = start_time):
    if proportion_completed != 0: 
        so_far = datetime.datetime.now() - start_time
        estimated_total = so_far / proportion_completed
        estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    else: 
        estimated_total = '?:??:??'
    return estimated_total



#------------------
# Randomly initiate parameters of a model.
#------------------
            
def init_weights(m):
    """Initialize weights of a neural network layer using Xavier normal and zero bias."""
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
    except:
        pass
    
    
    
#------------------
# Calculate Kullback-Leibler divergence between the prior and estimated posterior.
# DKL(Q||P) = .5 * ( (p_mu - q_mu)**2 / p_std**2 + q_std**2 / p_std**2 - log(q_std**2 / p_std**2) - 1 )
#------------------

def calculate_dkl(q_mu, q_std, p_mu, p_std):
    p_std = p_std ** 2
    q_std = q_std ** 2
    term_1 = (p_mu - q_mu) ** 2 / p_std
    term_2 = q_std / p_std
    term_3 = torch.log(term_2)
    out = 0.5 * (term_1 + term_2 - term_3 - 1)
    out = torch.nan_to_num(out)
    return out