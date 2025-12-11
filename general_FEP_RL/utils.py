import datetime  
import random
import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse 
import builtins
from math import exp

import torch 

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("\n\nDevice: {}.\n\n".format(device))



# Use random seed.
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



# Some utilities.
def print(*args, **kwargs):
    kwargs["flush"] = True
    builtins.print(*args, **kwargs)
    
start_time = datetime.datetime.now()

def duration(start_time = start_time):
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)

def estimate_total_duration(proportion_completed, start_time=start_time):
    if(proportion_completed != 0): 
        so_far = datetime.datetime.now() - start_time
        estimated_total = so_far / proportion_completed
        estimated_total = estimated_total - datetime.timedelta(microseconds=estimated_total.microseconds)
    else: estimated_total = "?:??:??"
    return(estimated_total)





def print_shapes(obs, action, complete_action, best_action, reward, done, mask, complete_mask):
    
    rows = []
    
    for key, value in obs.items():
        rows.append(("obs", key, list(value.shape)))
    for key, value in action.items():
        rows.append(("action", key, list(value.shape)))
    for key, value in complete_action.items():
        rows.append(("complete_action", key, list(value.shape)))
    for key, value in best_action.items():
        rows.append(("best_action", key, list(value.shape)))
    
    rows.append(("", "reward", list(reward.shape)))
    rows.append(("", "done", list(done.shape)))
    rows.append(("", "mask", list(mask.shape)))
    rows.append(("", "complete_mask", list(complete_mask.shape)))
    
    label_width = max(len(label) for label, _, _ in rows)
    key_width = max(len(key) for _, key, _ in rows)
    
    print("\n\n")
    print(f"{'Section':<{label_width}}  {'Key':<{key_width}}  Shape")
    print("-" * (label_width + key_width + 10))
    for label, key, shape in rows:
        print(f"{label:<{label_width}}  {key:<{key_width}}  {shape}")
    print("\n\n")


