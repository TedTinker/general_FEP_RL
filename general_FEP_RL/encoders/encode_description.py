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
                
        self.example_input = torch.zeros(99, 98, 16, 16, 3)
        if(verbose):
            print("\nDI Start:", self.example_input.shape)

        episodes, steps, [example] = model_start([(self.example_input, "cnn")])
        if(verbose): 
            print("\tReshaped:", example.shape)
        
        self.a = nn.Sequential(
            nn.Conv2d(
                in_channels = 3, 
                out_channels = 64, 
                kernel_size = 3, 
                stride=1, 
                padding=1, 
                dilation=1, 
                groups=1, 
                bias=True, 
                padding_mode='reflect', 
                device=None, 
                dtype=None),
            nn.PReLU())
        
        example = self.a(example)
        if(verbose): 
            print("\ta:", example.shape)
        example = example.reshape(example.shape[0], 64 * 16 * 16)
        if(verbose): 
            print("\tReshaped:", example.shape)
                
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
            print("EI End:")
            print("\toutput:", example.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, hidden_state):
        episodes, steps, [hidden_state] = model_start([(hidden_state, "cnn")])
        a = self.a(hidden_state)
        a = a.reshape(hidden_state.shape[0], 64 * 16 * 16)
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
    
    
