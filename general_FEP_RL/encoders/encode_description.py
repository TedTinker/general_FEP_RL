import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, model_start, model_end, mu_std



# Encode Description (ed).
class Encode_Description(nn.Module):
    def __init__(self, verbose = False):
        super(Encode_Description, self).__init__()
        
        self.out_features = 64
                
        self.example_input = torch.zeros(99, 98, 128)
        if(verbose):
            print("\nED Start:", self.example_input.shape)

        episodes, steps, [example] = model_start([(self.example_input, "lin")])
        if(verbose): 
            print("\tReshaped:", example.shape)
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = example.shape[-1],
                out_features = 128),
            nn.PReLU())
        
        example = self.a(example)
        if(verbose): 
            print("\ta:", example.shape)
                
        self.b = nn.Sequential(
            nn.Linear(
                in_features = example.shape[-1],
                out_features = self.out_features),
            nn.PReLU())
                
        example = self.b(example)
        if(verbose): 
            print("\toutput:", example.shape)
        
        [example] = model_end(episodes, steps, [(example, "lin")])
        self.example_output = example
        if(verbose):
            print("ED End:")
            print("\toutput:", example.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, description):
        episodes, steps, [description] = model_start([(description, "lin")])
        a = self.a(description)
        output = self.b(a)
        [output] = model_end(episodes, steps, [(output, "lin")])
        return(output)
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    ed = Encode_Description()
    print("\n\n")
    print(ed)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(ed, ed.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
