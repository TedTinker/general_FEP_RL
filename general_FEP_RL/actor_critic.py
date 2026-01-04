# CONCERN: In equations here and in the paper, are the actor and critic 
# matching the correct hidden state / previous hidden state?



#------------------
# actor_critic.py provides a model for an actor (policy) and critic (Q-network).
#------------------

import torch
from torch import nn 
from torchinfo import summary

from utils_torch import init_weights, generate_dummy_inputs

        

#------------------
# Actor generates actions from World Model hidden stats.
# a_t = \pi_\phi(h^q_t)
#------------------

class Actor(nn.Module):

    def __init__(
            self, 
            hidden_state_size, 
            action_dict, 
            verbose = False):
        super(Actor, self).__init__()
                        
        self.example_input = torch.zeros(32, 16, hidden_state_size)
        
        if verbose:
            print('START ACTOR')
        
        self.action_model_dict = nn.ModuleDict()
        for key, model in action_dict.items():
            self.action_model_dict[key] = nn.ModuleDict()
            self.action_model_dict[key]['decoder'] = model['decoder'](
                hidden_state_size,
                entropy = True, 
                arg_dict = model['decoder_arg_dict'], 
                verbose = verbose)
            
        self.example_action = {}
        for key, model in self.action_model_dict.items():
            self.example_action[key] = model['decoder'](self.example_input)     
            
        if verbose:
            print('END ACTOR')

        self.apply(init_weights)
        
        

    def forward(self, hidden_state, best_action = None):
        action = {}
        log_prob = {}
        for key, model in self.action_model_dict.items():
            a, lp = model['decoder'](hidden_state)
            action[key] = a
            log_prob[key] = lp
        if best_action is None:
            return action
        else:
            imitation_loss = {}
            for key, model in self.action_model_dict.items():
                loss_func = model['decoder'].loss_func
                loss_value = loss_func(best_action[key], action[key])
                imitation_loss[key] = loss_value
            return action, log_prob, imitation_loss
    
    
    
#------------------
# Example. 
#------------------
    
if __name__ == '__main__':
    
    from torch.profiler import profile, record_function, ProfilerActivity

    from general_FEP_RL.encoders.encode_image import Encode_Image
    from general_FEP_RL.decoders.decode_image import Decode_Image
    
    hidden_state_size = 128

    action_dict = {
        'make_image' : {
            'encoder' : Encode_Image,
            'decoder' : Decode_Image,
            'encoder_dict' : {},
            'decoder_arg_dict' : {},
            'accuracy_scaler' : 1,                               
            'complexity_scaler' : 1,                                 
            'eta' : 1  
            }
        }
    
    actor = Actor(
        hidden_state_size,
        action_dict,
        verbose = True)
    
    print('\n\n')
    print(actor)
    print()
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function('model_inference'):
            print(summary(actor, input_data=torch.zeros(32, 16, hidden_state_size)))
    #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))
    
    
    
#%%

    
    
#------------------
# Critic predicts Q-values.
# \widehat{Q}_t = Q_\theta(h^q_t, a_t)
#------------------

class Critic(nn.Module): 
    
    def __init__(
            self, 
            hidden_state_size, 
            action_dict, 
            value_decoder = None, 
            verbose = False):
        super(Critic, self).__init__()
        
        self.example_input = torch.zeros(32, 16, hidden_state_size)
        
        if verbose:
            print('START CRITIC')
                
        self.action_model_dict = nn.ModuleDict()
        for key, model in action_dict.items():
            self.action_model_dict[key] = nn.ModuleDict()
            self.action_model_dict[key]['encoder'] = model['encoder'](
                arg_dict = model['encoder_arg_dict'], 
                verbose = verbose)
                    
        full_encoding_size = sum(
            [hidden_state_size] + 
            [self.action_model_dict[key]['encoder'].arg_dict['encode_size']
             for key in self.action_model_dict.keys()])
        
        example_encoding = torch.zeros(32, 16, full_encoding_size)

        if value_decoder is not None:
            self.value_decoder = value_decoder
        else:
            self.value_decoder = nn.Sequential(
                nn.Linear(
                    full_encoding_size, 
                    1))
                        
        self.example_output = self.value_decoder(example_encoding)
        
        if verbose:
            print('END CRITIC:', self.example_output.shape)
        
        self.apply(init_weights)
        
        
        
    def forward(self, hidden_state, action):        
        encoded_action = []
        for key, model in self.action_model_dict.items():
            encoded_action.append(model['encoder'](action[key]))
        encoded_action = torch.cat(encoded_action + [hidden_state], dim=-1)
        value = self.value_decoder(encoded_action)
        return value
    


#------------------
# Example. 
#------------------

if __name__ == '__main__':
        
    critic_dict = {
        'make_image' : {
            'encoder' : Encode_Image,
            'decoder' : Decode_Image,
            'encoder_arg_dict' : {
                'encode_size' : 128,
                'zp_zq_sizes' : [128]},
            'decoder_arg_dict' : {},
            'accuracy_scaler' : 1,                               
            'complexity_scaler' : 1,                                 
            'eta' : 1  
            }
        }
        
    critic = Critic(
        hidden_state_size, 
        critic_dict, 
        verbose = True)
    
    print('\n\n')
    print(critic)
    print()
    
    dummies = generate_dummy_inputs(None, critic.action_model_dict, hidden_state_size)
    dummy_inputs = (dummies['hidden'], dummies['act_enc_in'])
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function('model_inference'):
            print(summary(critic, input_data = dummy_inputs))
    #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))

# %%