#%%
#------------------
# encoder_decoder.py provides models convenient for world_models.
#------------------

import torch
from torch import nn
import torch.nn.functional as F

from general_FEP_RL.shape_to_shape_models import Shape_to_Shape_Model



# Function for sampling from Gaussian distributions.
def sample(mu, std):
    epsilon = torch.randn_like(std)
    return mu + epsilon * std



# Encoder for hidden_states and posterior_sample.
class Misc_Encoder(Shape_to_Shape_Model):
    
    def __init__(
        self,
        name,               # String. Should be unique.
        input_size,         # Input_shape will equal output_shape.
        verbose = False):
                    
        super().__init__(
            name = name,               
            input_shape = (input_size,),        
            output_shape = (input_size,),        
            verbose = verbose)
        
    def build_model(self, arg_dict):
        self.linear_layers = nn.Sequential(
            nn.Linear(
                in_features = self.input_shape[0], 
                out_features = self.input_shape[0]),
            nn.LeakyReLU(),
            nn.Linear(
                in_features = self.input_shape[0], 
                out_features = self.input_shape[0]),
            nn.LeakyReLU())
    
    def forward(self, value):
        return self.linear_layers(value)
    
    
    
# Decoder to predict a lower layer's posterior_sample, and decode hidden_state.
class Misc_Decoder(Shape_to_Shape_Model):

    def __init__(
            self, 
            name,               # String. Should be unique.
            input_size,         # Size of prior_sample, or encoded posterior_sample (and higher hidden state, if available).
            output_size,        # Size of lower_layer_posterior_sample, or hidden_state.
            bounded = True,     # Apply Tanh?
            verbose = False):
        
        super().__init__(
            name = name,                    
            input_shape = (input_size,),    
            output_shape = (output_size,),
            arg_dict = {'bounded' : bounded},
            verbose = verbose)

    def build_model(self, arg_dict):
        layers = [
            nn.Linear(self.input_shape[0], self.input_shape[0]),
            nn.LeakyReLU(),
            nn.Linear(self.input_shape[0], self.output_shape[0])]
        if arg_dict['bounded']:
            layers.append(nn.Tanh())
        self.linear_layers = nn.Sequential(*layers)

    def forward(self, value):
        return self.linear_layers(value)

    @staticmethod
    def loss_func(predicted_values, target_values):
        return F.mse_loss(predicted_values, target_values, reduction = 'none')
    
    

# Probabilistic decoder for prior_ and posterior_inner_states.
class Inner_State_Decoder(Shape_to_Shape_Model):
    
    def __init__(
        self,
        name,               # String. Should be unique.
        input_size,         # Size of encoded prior inputs or posterior inputs.
        output_size,        # Size of inner states.
        verbose = False):
                    
        super().__init__(
            name = name,               
            input_shape = (input_size,),        
            output_shape = (output_size,),        
            verbose = verbose)
        
    def build_model(self, arg_dict):
        
        # Mean.
        self.mu = nn.Sequential(
            nn.Linear(in_features = self.input_shape[0], out_features = self.output_shape[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features = self.output_shape[0], out_features = self.output_shape[0]))
        self.mu[-1].weight.data.mul_(0.1)
        self.mu[-1].bias.data.zero_()
        
        # Standard deviation.
        self.std = nn.Sequential(
            nn.Linear(in_features = self.input_shape[0], out_features = self.output_shape[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features = self.output_shape[0], out_features = self.output_shape[0]))
    
    def forward(self, value):
        mu = self.mu(value)
        std = 1e-2 + F.softplus(self.std(value))
        inner_state_sample = sample(mu, std)
        return {'mu' : mu, 'std' : std, 'sample' : inner_state_sample}



# Decoder for encoded posterior inputs. One Inner_State_Decoder per modality.
# This ensures that each modality's inner_state is independent from the others.
class Sliced_Inner_State_Decoder(Shape_to_Shape_Model):

    def __init__(
        self,
        name,                   # String. Should be unique.
        input_size,             # Width of the WHOLE encoding.
        output_size,            # Size of this modality's inner state decoding.
        columns,                # Which inputs may be read?
        verbose = False):

        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            arg_dict = {'columns' : columns},
            verbose = verbose)

    def build_model(self, arg_dict):
        columns = torch.as_tensor(arg_dict['columns'], dtype = torch.long)
        self.register_buffer('columns', columns)
        self.inner_state_decoder = Inner_State_Decoder(
            name = f'{self.name}_from_own_encoding',
            input_size = len(columns),
            output_size = self.output_shape[0])

    def forward(self, value):
        return self.inner_state_decoder(value.index_select(-1, self.columns))



