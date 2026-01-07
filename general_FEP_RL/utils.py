#------------------
# utils.py provides some basic utilities.
#------------------

import datetime  
import random
import matplotlib
matplotlib.use('Agg')
import numpy as np

import torch 



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
