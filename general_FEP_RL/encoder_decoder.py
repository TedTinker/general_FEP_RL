import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from general_FEP_RL.shape_to_shape_models import Shape_to_Shape_Model



# I use this encoder for hidden_states and posterior_sample.
class Misc_Encoder(Shape_to_Shape_Model):
    
    def __init__(
        self,
        name,
        input_size,        
        verbose = False):
            
        self.input_size = input_size
        
        super().__init__(
            name = name,               
            input_shape = (input_size,),        
            output_shape = (input_size,),        
            verbose = verbose)
        
    def build_model(self, arg_dict):
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features = self.input_shape[0], out_features = self.input_shape[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features = self.input_shape[0], out_features = self.input_shape[0]),
            nn.LeakyReLU())
    
    def forward(self, value):
        return self.linear_layers(value)
    
    
    
# I use this to predict another layer's posterior_sample.
class Misc_Decoder(Shape_to_Shape_Model):

    def __init__(self, name, input_size, output_size, verbose = False):
        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            verbose = verbose)

    def build_model(self, arg_dict):
        self.linear_layers = nn.Sequential(
            nn.Linear(self.input_shape[0], self.input_shape[0]),
            nn.LeakyReLU(),
            nn.Linear(self.input_shape[0], self.output_shape[0]),
            nn.Tanh())

    def forward(self, value):
        return self.linear_layers(value)

    @staticmethod
    def loss_func(predicted_values, target_values):
        return F.mse_loss(predicted_values, target_values, reduction = 'none')
    
    

# I use this is decode prior_ and posterior_inner_states.
class Inner_State_Decoder(Shape_to_Shape_Model):
    
    def __init__(
        self,
        name,
        input_size,
        output_size,
        verbose = False):
                    
        super().__init__(
            name = name,               
            input_shape = (input_size,),        
            output_shape = (output_size,),        
            verbose = verbose)
        
    def build_model(self, arg_dict):
        self.mu = nn.Sequential(
            nn.Linear(in_features = self.input_shape[0], out_features = self.output_shape[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features = self.output_shape[0], out_features = self.output_shape[0]),
            nn.Tanh())
        
        self.std = nn.Sequential(
            nn.Linear(in_features = self.input_shape[0], out_features = self.output_shape[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features = self.output_shape[0], out_features = self.output_shape[0]))
    
    def forward(self, value):
        mu = self.mu(value)
        std = 1e-2 + F.softplus(self.std(value))      # We may want a larger minimum.
        inner_state_sample = sample(mu, std)
        return {'mu' : mu, 'std' : std, 'sample' : inner_state_sample}




# I use this so each modality's posterior sees only its own encoding.
#
# A Divider hands every sub-model the whole encoding it was given, and requires them
# all to declare the same input_shape. So this advertises the full encoding width to
# satisfy the Divider, and takes its slice inside forward: shared context (the
# previous hidden_state and the prior inputs) plus this modality's own encoding.
class Sliced_Inner_State_Decoder(Shape_to_Shape_Model):

    def __init__(
        self,
        name,
        input_size,             # Width of the WHOLE encoding.
        output_size,
        columns,                # Which columns of it this modality may read.
        verbose = False):

        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            arg_dict = {'columns' : columns},
            verbose = verbose)

    def build_model(self, arg_dict):
        columns = torch.as_tensor(arg_dict['columns'], dtype = torch.long)
        # A buffer rather than a plain attribute, so it follows .to(device).
        self.register_buffer('columns', columns)
        self.inner_state_decoder = Inner_State_Decoder(
            name = f'{self.name}_from_own_encoding',
            input_size = len(columns),
            output_size = self.output_shape[0])

    def forward(self, value):
        return self.inner_state_decoder(value.index_select(-1, self.columns))


class Hidden_State_Decoder(Shape_to_Shape_Model):
    
    def __init__(
        self,
        name,
        input_size,
        output_size,
        verbose = False):
                    
        super().__init__(
            name = name,               
            input_shape = (input_size,),        
            output_shape = (output_size,),        
            verbose = verbose)
        
    def build_model(self, arg_dict):
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features = self.input_shape[0], out_features = self.input_shape[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features = self.input_shape[0], out_features = self.output_shape[0]),
            nn.Tanh())
    
    def forward(self, value):
        return self.linear_layers(value)