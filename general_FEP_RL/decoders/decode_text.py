import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from utils import default_args
from utils_torch import init_weights, model_start, model_end, mu_std



# Decode text (DT)
class Decode_Text(nn.Module):
    def __init__(self, args = default_args, encoded_action_size = 0, entropy = False, verbose = False):
        super(Decode_Text, self).__init__()
        
        self.args = args
        
        self.example_start = torch.zeros(self.args.batch_size, self.args.num_steps, self.args.fhs_size + encoded_action_size)
        episodes, steps, [example] = model_start([(self.example_start, "lin")])
        if(verbose):
            print("\nDT start:", self.example_start.shape)
            print("Start:", example.shape)
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = self.args.fhs_size + encoded_action_size,
                out_features = 128 * self.args.max_text_len),
            nn.PReLU())
        
        example = self.a(example)
        if(verbose):
            print("a:", example.shape)
        example = example.reshape(episodes * steps, self.args.max_text_len, 128)
        if(verbose):
            print("a2:", example.shape)

        mu = nn.GRU(
            input_size = 128,
            hidden_size = 128,
            batch_first = True)
        
        self.b = mu_std(mu, entropy = entropy)
        
        # WORK ON THIS: need to accomodate GRU output.
        (example_output, example_h), (example_log_prob_output, example_log_prob_h) = self.b(example)
        if(verbose):
            print(f"b: \n\t{example_output.shape}, {example_h.shape}, \n\t{example_log_prob_output.shape}, {example_log_prob_h.shape}")
            
        example_output = torch.argmax(example_output, dim = -1).int() 
        example_h = torch.argmax(example_h, dim = -1).int() 
        if(verbose):
            print(f"b2: {example_output.shape}, {example_h.shape}")
        
        example_output = example_output.reshape(episodes, steps, example_output.shape[-1])
        example_h = example_h.reshape(episodes, steps, 1)
        example_log_prob_output = example_log_prob_output.reshape(episodes, steps, example_log_prob_output.shape[-2])
        example_log_prob_h = example_log_prob_h.reshape(episodes, steps, example_log_prob_h.shape[-1])
        
        # PROBLEM: Model might end with multiple values.
        self.example_end = (example_output, example_h), (example_log_prob_output, example_log_prob_h)
        
        if(verbose):
            print(f"DT end: \n \t{example_output.shape}, {example_h.shape}, \n\t{example_log_prob_output.shape}, {example_log_prob_h.shape}")
        
        self.apply(init_weights)



    def forward(self, fhs):
        episodes, steps, [fhs] = model_start([(fhs, "lin")])
        a = self.a(fhs)
        a = a.reshape(episodes * steps, self.args.max_text_len, 128)
        (output, h), (log_prob_output, log_prob_h) = self.b(a)
        
        output = torch.argmax(output, dim = -1).int() 
        h = torch.argmax(h, dim = -1).int() 
        
        output = output.reshape(episodes, steps, output.shape[-1])
        h = h.reshape(episodes, steps, 1)
        log_prob_output = log_prob_output.reshape(episodes, steps, log_prob_output.shape[-2])
        log_prob_h = log_prob_h.reshape(episodes, steps, log_prob_h.shape[-1])
        
        return (output, h), (log_prob_output, log_prob_h)
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    args = default_args
    pto = Decode_Text(args, verbose = True)
    print("\n\n")
    print(pto)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(pto, input_data=(pto.example_start)))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    pto = Decode_Text(args, entropy = True, verbose = True)
    print("\n\n")
    print(pto)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(pto, input_data=(pto.example_start)))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%