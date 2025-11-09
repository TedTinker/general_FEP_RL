import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from utils import default_args
from utils_torch import init_weights, model_start, model_end



# Encode text (ET)
class Encode_Text(nn.Module):
    def __init__(self, args = default_args, verbose = False):
        super(Encode_Text, self).__init__()

        self.args = args
        
        self.example_start = torch.randint(
            low=0,
            high=128,
            size=(self.args.batch_size, self.args.num_steps, self.args.max_text_len, 16),
            dtype=torch.long)

        episodes, steps, [example] = model_start([(self.example_start, "lin")])            
        
        example = torch.argmax(example, dim = -1).int()
        
        if(verbose):
            print("\nET start:", self.example_start.shape)
            print("Start:", example.shape)
            print("argmax:", example.shape)
        
        self.a = nn.Sequential(
                nn.Embedding(
                    num_embeddings = 128,
                    embedding_dim = self.args.text_embedding),
                nn.PReLU())
        
        example = self.a(example)
        if(verbose):
            print("a:", example.shape)

        self.b = nn.GRU(
            input_size = example.shape[2],
            hidden_size = 72,
            batch_first = True)
        
        _, example = self.b(example)
        if(verbose):
            print("b:", example.shape)
            
        self.c = nn.Sequential(
                nn.PReLU(),
                nn.Linear(
                    in_features = 72, 
                    out_features = 127))
        
        example = self.c(example)
        if(verbose):
            print("c:", example.shape)
        
        [example] = model_end(episodes, steps, [(example, "lin")])
        self.example_end = example
        if(verbose):
            print("ET end:", example.shape, "\n")
                        
        self.apply(init_weights)
        
    def forward(self, text):       
        episodes, steps, [text] = model_start([(text, "text")])   
        text = torch.argmax(text, dim = -1).int()                    
        a = self.a(text)
        _, b = self.b(a)    
        c = self.c(b)
        [c] = model_end(episodes, steps, [(c, "lin")])
        return(c)

    

# Let's check it out!
if(__name__ == "__main__"):
    args = default_args
    et = Encode_Text(args, verbose = True)
    print("\n\n")
    print(et)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(et, input_data=(et.example_start)))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# %%