import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, model_start, model_end, mu_std



# Decode Image (di).
class Decode_Image(nn.Module):
    def __init__(self, hidden_state_size, encoded_action_size = 0, entropy = False, arg_dict = {}, verbose = False):
        super(Decode_Image, self).__init__()
                
        self.example_input = torch.zeros(32, 16, hidden_state_size + encoded_action_size)
        if(verbose): 
            print("\nDI Start:", self.example_input.shape)

        episodes, steps, [example] = model_start([(self.example_input, "lin")])
        if(verbose): 
            print("\tReshaped:", example.shape)
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = hidden_state_size,
                out_features = 16 * 28 * 28),
            nn.PReLU())
        
        example = self.a(example)
        if(verbose): 
            print("\ta:", example.shape)
        example = example.reshape(example.shape[0], 16, 28, 28)
        if(verbose): 
            print("\tReshaped:", example.shape)
                
        mu = nn.Sequential(
            nn.Conv2d(
                in_channels = 16, 
                out_channels = 1,
                kernel_size = 3,
                padding = 1,
                padding_mode = "reflect"),
            nn.Tanh())
        
        self.mu_std = mu_std(mu, entropy = entropy)
        
        example_output, example_log_prob = self.mu_std(example)
        if(verbose): 
            print("\toutput:", example_output.shape)
            print("\tlog_prob:", example_log_prob.shape)
        
        [example_output, example_log_prob] = model_end(episodes, steps, [(example_output, "cnn"), (example_log_prob, "lin")])
        example_output = example_output.reshape(episodes, steps, 28, 28, 1)
        self.example_output = example_output
        if(verbose): 
            print("DI End:")
            print("\toutput:", example_output.shape)
            print("\tlog_prob:", example_log_prob.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, hidden_state):
        episodes, steps, [hidden_state] = model_start([(hidden_state, "lin")])
        a = self.a(hidden_state)
        a = a.reshape(episodes * steps, 16, 28, 28)
        output, log_prob = self.mu_std(a)
        output = (output + 1) / 2
        [output, log_prob] = model_end(episodes, steps, [(output, "cnn"), (log_prob, "lin")])
        output = output.reshape(episodes, steps, 28, 28, 1)
        return(output, log_prob)
    
    
    
    @staticmethod
    def loss_func(true_values, predicted_values):
        print(true_values.shape, predicted_values.shape)
        loss_value = F.mse_loss(predicted_values, true_values, reduction = None)
        return loss_value
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    di = Decode_Image(hidden_state_size = 128)
    print("\n\n")
    print(di)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(di, di.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    

    di = Decode_Image(hidden_state_size = 128, entropy = True)
    print("\n\n")
    print(di)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(di, di.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
    example_dict = {
        "decoder" : di,
        "target_entropy" : 1,
        "accuracy_scaler" : 1,                               
        "complexity_scaler" : 1,                                 
        "eta" : 1                                   
        }
    