#%%
#------------------
# mtrnn.py provides an RNN layer which has a custom timescale and is gated. 
#------------------

import torch 
from torch import nn 

from general_FEP_RL.utils_torch import init_weights



#------------------
# One cell of the recurrent neural network.
# r_t = σ(W^r_x x_t + W^r_h h^q_{t-1})
# z_t = σ(W^z_x x_t + W^z_h h^q_{t-1})
# \tilde{h}_t = tanh(W^n_x x_t + r_t ⊙ W^n_h h^q_{t-1})
# \hat{h}_t = (1 − z_t) ⊙ \tilde{h}_t + z_t ⊙ h^q_{t-1}
# h^q_t = (1/τ) \hat{h}_t + (1 − 1/τ) h^q_{t-1}
#------------------

class MTRNNCell(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            time_constant):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_constant = time_constant

        self.new = 1.0 / time_constant
        self.old = 1.0 - self.new

        # One fused linear:
        # (x || h) -> [r, z, n]
        self.linear = nn.Linear(
            input_size + hidden_size,
            3 * hidden_size
        )

        self.apply(init_weights)

    def forward(self, x, h):

        # Concatenate input and hidden state
        xh = torch.cat([x, h], dim=-1)

        # Single matmul
        gates = self.linear(xh)

        # Split into gates
        r, z, n = gates.chunk(3, dim=-1)

        # Apply nonlinearities
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)

        # GRU-style update
        new_h = n * (1.0 - z) + h * z

        # Time-scale update
        new_h = self.new * new_h + self.old * h

        # Match original shape contract
        return new_h.unsqueeze(1)
    
    
    
    
    
#------------------
# Example. 
#------------------

if __name__ == "__main__":
        
    from torch.profiler import profile, record_function, ProfilerActivity
    from torchinfo import summary as torch_summary
        
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
    
    

#------------------
# Application of the cell.
#------------------

class MTRNN(nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_size,
            time_constant):
        super(MTRNN, self).__init__()
        self.hidden_size = hidden_size
        self.mtrnn_cell = MTRNNCell(input_size, hidden_size, time_constant)
        
        self.apply(init_weights)
        
        

    def forward(self, x, h = None):
        if(h is None):
            h = torch.zeros(
                (x.shape[0], 1, self.hidden_size),
                device=x.device, dtype=x.dtype,)
        outputs = []
        for step in range(x.shape[1]):  
            h = self.mtrnn_cell(x[:, step], h[:, 0])
            outputs.append(h)
        outputs = torch.cat(outputs, dim = 1)
        return outputs
    
    
    
    
    
    
    
""" 
class MTRNN(nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_size,
            time_constant):
        super(MTRNN, self).__init__()
        self.hidden_size = hidden_size
        self.mtrnn_cell = MTRNNCell(input_size, hidden_size, time_constant)

        self.apply(init_weights)

    def forward(self, x, h=None):
        B, T, _ = x.shape

        if h is None:
            h = torch.zeros(
                (B, self.hidden_size),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            # remove time dimension
            h = h[:, 0]

        # scan expects:
        #   fn(carry, x_t) -> (new_carry, output)
        def step_fn(h_prev, x_t):
            h_next = self.mtrnn_cell(x_t, h_prev)
            # mtrnn_cell returns (B, 1, H), squeeze it
            h_next = h_next[:, 0]
            return h_next, h_next

        # torch.scan scans over dimension 0, so transpose time to front
        x_time_major = x.transpose(0, 1)  # (T, B, input_size)

        # Run scan
        h_T, outputs = torch.scan(
            step_fn,
            h,
            x_time_major,
        )
        # outputs: (T, B, hidden_size)

        # Back to (B, T, hidden_size)
        outputs = outputs.transpose(0, 1)

        return outputs
    """
    
    
    


#------------------
# Example. 
#------------------

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