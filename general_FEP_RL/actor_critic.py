#%% 

from torch.profiler import profile, record_function, ProfilerActivity

import torch
from torch import nn 
from torchinfo import summary

from general_FEP_RL.utils_torch import init_weights, generate_dummy_inputs

        

# Actor or policy, generating motor commands from Forward Model Hidden stats, fhs.
class Actor(nn.Module):

    def __init__(
            self, 
            hidden_state_size, 
            action_dict, 
            verbose = False):
        super(Actor, self).__init__()
                        
        self.example_input = torch.zeros((32, 16, hidden_state_size))
        
        self.action_dict = nn.ModuleDict()
        for key in action_dict.keys():
            self.action_dict[key] = nn.ModuleDict()
            self.action_dict[key]["decoder"] = action_dict[key]["decoder"](hidden_state_size, entropy = True, arg_dict = action_dict[key]["decoder_arg_dict"], verbose = verbose)
            
        if(verbose):
            pass
        
        example_action = {}
        for key, value in self.action_dict.items():
            example_action[key] = self.action_dict[key]["decoder"](self.example_input)
            
        if verbose:
            pass                

        self.apply(init_weights)

    def forward(self, hidden_state, best_action = None):
        action = {}
        log_prob = {}
        for key, model in self.action_dict.items():
            a, lp = self.action_dict[key]["decoder"](hidden_state)
            action[key] = a
            log_prob[key] = lp
        if(best_action == None):
            return(action, log_prob)
        else:
            imitation_loss = {}
            for key, model in self.action_dict.items():
                loss_func = self.action_dict[key]["decoder"].loss_func
                print(key, best_action[key].shape, action[key].shape)
                loss_value = loss_func(best_action[key], action[key])
                imitation_loss[key] = loss_value
            return(action, log_prob, imitation_loss)
    
    
    
if __name__ == "__main__":
    
    from general_FEP_RL.encoders.encode_image import Encode_Image
    from general_FEP_RL.decoders.decode_image import Decode_Image
    
    hidden_state_size = 128

    action_dict = {
        "make_image" : {
            "encoder" : Encode_Image,
            "decoder" : Decode_Image,
            "encoder_dict" : {},
            "decoder_arg_dict" : {},
            "accuracy_scaler" : 1,                               
            "complexity_scaler" : 1,                                 
            "eta" : 1  
            }
        }
    
    actor = Actor(hidden_state_size, action_dict, verbose = True)
    
    print("\n\n")
    print(actor)
    print()
    
        
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(actor, input_data=torch.zeros(32, 16, hidden_state_size)))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
#%%

    
    
# Critic or Q-network for predicting Q-value.
class Critic(nn.Module): 
    
    def __init__(
            self, 
            hidden_state_size, 
            action_dict, 
            value_decoder = None, 
            verbose = False):
        super(Critic, self).__init__()
        
        self.example_input = torch.zeros((32, 16, hidden_state_size))
                
        self.action_dict = nn.ModuleDict()
        for key, model in action_dict.items():
            self.action_dict[key] = nn.ModuleDict()
            self.action_dict[key]["encoder"] = action_dict[key]["encoder"](arg_dict = action_dict[key]["encoder_arg_dict"], verbose = verbose)
            
        if(verbose):
            pass
                    
        example_full_encoding = sum([self.action_dict[key]["encoder"].arg_dict["encode_size"] for key in self.action_dict.keys()] + [hidden_state_size])
        
        if(verbose):
            pass

        if(value_decoder != None):
            self.value_decoder = value_decoder()
        else:
            self.value_decoder = nn.Sequential(
                nn.Linear(
                    example_full_encoding, 
                    1))
                        
        if(verbose):
            pass
        
        self.apply(init_weights)
        
    def forward(self, hidden_state, action):        
        encoded_action = []
        for key, model in self.action_dict.items():
            encoded_action.append(self.action_dict[key]["encoder"](action[key]))
        encoded_action = torch.cat(encoded_action + [hidden_state], dim=-1)
        value = self.value_decoder(encoded_action)
        return(value)
    


if __name__ == "__main__":
        
    action_dict = {
        "make_image" : {
            "encoder" : Encode_Image,
            "decoder" : Decode_Image,
            "encoder_arg_dict" : {
                "encode_size" : 128,
                "zp_zq_sizes" : [128]},
            "decoder_arg_dict" : {},
            "accuracy_scaler" : 1,                               
            "complexity_scaler" : 1,                                 
            "eta" : 1  
            }
        }
        
    critic = Critic(hidden_state_size, action_dict, verbose = True)
    
    print("\n\n")
    print(critic)
    print()
    
    
    
    dummies = generate_dummy_inputs(None, critic.action_dict, hidden_state_size)
    dummy_inputs = dummies["hidden"], dummies["act_enc_in"],
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(critic, input_data = dummy_inputs))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

# %%