#%%
import torch 
from torch import nn 
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary as torch_summary

from utils_torch import init_weights



# One cell in an (MTRNN) Multiple Timescall Recurrent Neural Network.
# This is also a GRU (Gated Recurrent Unit).
class MTRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, time_constant):
        super(MTRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_constant = time_constant
        self.new = 1 / time_constant
        self.old = 1 - self.new

        self.r_x = nn.Sequential(
            nn.Linear(
                in_features = input_size, 
                out_features = hidden_size))
        self.r_h = nn.Sequential(
            nn.Linear(
                in_features = hidden_size, 
                out_features = hidden_size))
        
        self.z_x = nn.Sequential(
            nn.Linear(
                in_features = input_size, 
                out_features = hidden_size))
        self.z_h = nn.Sequential(
            nn.Linear(
                in_features = hidden_size, 
                out_features = hidden_size))
        
        self.n_x = nn.Sequential(
            nn.Linear(
                in_features = input_size, 
                out_features = hidden_size))
        self.n_h = nn.Sequential(
            nn.Linear(
                in_features = hidden_size, 
                out_features = hidden_size))
        
        self.apply(init_weights)

    def forward(self, x, h):
        r = torch.sigmoid(self.r_x(x) + self.r_h(h))
        z = torch.sigmoid(self.z_x(x) + self.z_h(h))
        new_h = torch.tanh(self.n_x(x) + r * self.n_h(h))
        new_h = new_h * (1 - z)  + h * z
        new_h = new_h * self.new + h * self.old
        if(len(new_h.shape) == 2):
            new_h = new_h.unsqueeze(1)
        return new_h
    
    
    
if __name__ == "__main__":
    
    episodes = 32 
    steps = 16
    
    cell = MTRNNCell(
        input_size = 16,
        hidden_size = 32,
        time_constant = 1)
    
    print("\n\n")
    print(cell)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(cell, 
                                ((episodes, steps, 16), 
                                (episodes, steps, 32))))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        
#%%
    
    

# An MTRNN, applying the cell. 
class MTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, time_constant):
        super(MTRNN, self).__init__()
        self.hidden_size = hidden_size
        self.mtrnn_cell = MTRNNCell(input_size, hidden_size, time_constant)
        
        self.apply(init_weights)

    def forward(self, x, h = None):
        if(h == None):
            h = torch.zeros((x.shape[0], 1, self.hidden_size))
        outputs = []
        for step in range(x.shape[1]):  
            h = self.mtrnn_cell(x[:, step], h[:, 0])
            outputs.append(h)
        outputs = torch.cat(outputs, dim = 1)
        return outputs
    


if __name__ == "__main__":
    mtrnn = MTRNN(
        input_size = 16,
        hidden_size = 32,
        time_constant = 1)
    
    print("\n\n")
    print(mtrnn)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(torch_summary(mtrnn, 
                                ((episodes, steps, 16), 
                                (episodes, steps, 32))))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%