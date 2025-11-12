import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, model_start, model_end, mu_std



# Decode Description (dd).
class Decode_Description(nn.Module):
    def __init__(self, hidden_state_size, encoded_action_size = 0, entropy = False, verbose = False):
        super(Decode_Description, self).__init__()
                
        self.example_input = torch.zeros(32, 16, hidden_state_size + encoded_action_size)
        if(verbose): 
            print("\nDD Start:", self.example_input.shape)

        episodes, steps, [example] = model_start([(self.example_input, "lin")])
        if(verbose): 
            print("\tReshaped:", example.shape)
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = hidden_state_size,
                out_features = 128),
            nn.PReLU())
        
        example = self.a(example)
        if(verbose): 
            print("\ta:", example.shape)
                
        mu = nn.Sequential(
            nn.Linear(
                in_features = 128,
                out_features = 128),
            nn.PReLU())
        
        self.mu_std = mu_std(mu, entropy = entropy)
        
        example_output, example_log_prob = self.mu_std(example)
        if(verbose): 
            print("\toutput:", example_output.shape)
            print("\tlog_prob:", example_log_prob.shape)
        
        [example_output, example_log_prob] = model_end(episodes, steps, [(example_output, "lin"), (example_log_prob, "lin")])
        self.example_output = example_output
        if(verbose): 
            print("DD End:")
            print("\toutput:", example_output.shape)
            print("\tlog_prob:", example_log_prob.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, hidden_state):
        episodes, steps, [hidden_state] = model_start([(hidden_state, "lin")])
        a = self.a(hidden_state)
        output, log_prob = self.mu_std(a)
        [output, log_prob] = model_end(episodes, steps, [(output, "lin"), (log_prob, "lin")])
        return(output, log_prob)
    
    
    
    def loss_func(self, true_values, predicted_values):
        print(true_values.shape, predicted_values.shape)
        loss_value = F.mse_loss(predicted_values, true_values, reduction = None)
        return loss_value
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    dd = Decode_Description(hidden_state_size = 128)
    print("\n\n")
    print(dd)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dd, dd.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    

    dd = Decode_Description(hidden_state_size = 128, entropy = True)
    print("\n\n")
    print(dd)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dd, dd.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
