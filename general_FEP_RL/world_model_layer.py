#%% 
#------------------
# world_model_layer.py provides architecture for predicting future observations
# in one mtrnn layer of a world_model.  
#------------------

import math 
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from general_FEP_RL.utils import calculate_dkl
from general_FEP_RL.shape_to_shape_models import Shape_to_Shape_Model, Combiner, Divider
from general_FEP_RL.encoder_decoder import Misc_Encoder, Misc_Decoder, Inner_State_Decoder, Sliced_Inner_State_Decoder, Hidden_State_Cell


# The model itself.
class World_Model_Layer(nn.Module):
    
    def __init__(
            self,
            prior_input_encoder,            # Combiner (encodes values, including previous hidden_state).
            prior_inner_state_decoder,      # Divider (makes prior inner_states for everything in prediction_decoder).
            
            posterior_input_encoder,        # Combiner (encodes values, including everything in prior_input_encoder and perhaps lower_layer_posterior_sample).
            posterior_inner_state_decoder,  # Divider (makes inner_states, perhaps including lower_layer_posterior_sample).
            
            prediction_decoder,             # Divider (prediction of posterior input values).
            
            hidden_state_input_encoder,     # Combiner (encodes posterior_sample, and perhaps higher_layer_hidden_state)
            hidden_state_decoder,           # Hidden_State_Cell (makes hidden_state from posterior sample and previous_hidden_state).
                        
            time_constant = 1,
            verbose = False):
        
        super().__init__()
        
        self.prior_input_encoder = prior_input_encoder
        self.prior_inner_state_decoder = prior_inner_state_decoder
        
        self.posterior_input_encoder = posterior_input_encoder
        self.posterior_inner_state_decoder = posterior_inner_state_decoder
        
        self.prediction_decoder = prediction_decoder
        
        self.hidden_state_input_encoder = hidden_state_input_encoder
        self.hidden_state_decoder = hidden_state_decoder
        
        # self.time_constant = time_constant     # Applied inside hidden_state_decoder, not here.
        
        
        
    def forward(self):
        # I leave this empty, as it is never used.
        raise NotImplementedError("Unused forward pass of world_model_layer.")
    
    
    
   # Prior only. A "dream" step needs nothing more, and has no observations to give a posterior.
    def make_prior_inner_states(self, prior_value_dict, deterministic = False):
        encoding = self.prior_input_encoder(prior_value_dict)                                           # Encodes values.   
        prior_inner_states = self.prior_inner_state_decoder(encoding, deterministic = deterministic)    # Decodes (mu, std, sample) for prior_value.
        return {
            name : {
                'prior_mu' : states['mu'],
                'prior_std' : states['std'],
                'prior_sample' : states['sample']}
            for name, states in prior_inner_states.items()}

    def make_inner_states(self, prior_value_dict, posterior_value_dict, deterministic = False):
        inner_states = self.make_prior_inner_states(prior_value_dict, deterministic = deterministic)
        encoding = self.posterior_input_encoder(posterior_value_dict)                                           # Encodes values.   
        posterior_inner_states = self.posterior_inner_state_decoder(encoding, deterministic = deterministic)    # Decodes (mu, std, sample) for posterior_value.

        for name, states in inner_states.items():
            posterior = posterior_inner_states[name]
            states['posterior_mu'] = posterior['mu']
            states['posterior_std'] = posterior['std']
            states['posterior_sample'] = posterior['sample']
            states['dkl'] = calculate_dkl(
                posterior['mu'], posterior['std'],
                states['prior_mu'], states['prior_std'])

        return inner_states
    
    
    
    # Concatenate modules.
    # Dividers sort at construction. 
    def combine_inner_state_samples(self, inner_states, prior_or_posterior):
        key = f'{prior_or_posterior}_sample'
        return torch.cat(
            [inner_states[name][key] for name in self.posterior_inner_state_decoder.models_dict.keys()],
            dim = -1)
        
        
    
    # Predictions based on inner_state_sample.
    def make_predictions(self, inner_state_sample):
        predictions = self.prediction_decoder(inner_state_sample)
        return predictions
    


    # Hidden state based on inner_state_sample, and higher_layer_hidden_state if there's a higher layer.
    # For MTRNN, hidden_state_decoder's update gate is capped by time_constant.
    def make_hidden_state(self, previous_hidden_state, inner_state_sample, higher_layer_hidden_state = None):
        value_dict = {'inner_state_sample' : inner_state_sample}
        if higher_layer_hidden_state is not None:
            value_dict['higher_layer_hidden_state'] = higher_layer_hidden_state
        encoding = self.hidden_state_input_encoder(value_dict)
        return self.hidden_state_decoder(encoding, previous_hidden_state)
        
        
        
######################



# Function to make the layer.
def make_world_model_layer(
    hidden_state_size,                                      # Size of this layer's hidden_state.
    
    dict_of_prior_input_encoder_class_dicts,                # Dictionary of dictionaries for prior_input encoders. (Inner state decoder is automatically generated.)
                                                            # Do NOT include hidden_state encoding. (This is automatically generated.)
                                                            # Keys:
                                                                # name.
                                                                # Keys:
                                                                    # class. (These must have fixed input_size and fixed output_size.)
                                                                    # (There is no decoding_output_size. Only inner_states shared with posterior_inner_states are decoded.)
                                                                
    dict_of_posterior_input_encoder_class_dicts,            # Dictionary of dictionaries for posterior_input encoders. (Inner state decoder is automatically generated.)
                                                            # Do NOT include hidden_state encoding or lower_layer_posterior_sample encoding. (These are automatically generated.)
                                                            # Do NOT include lower_layer_posterior_sample_output_size. (This is automatically generated.)
                                                            # Keys:
                                                                # name.
                                                                # Keys:
                                                                    # class. (These must have fixed input_shape and fixed output_shape.)
                                                                    # decoding_output_size.
                                                                
    dict_of_prediction_decoder_class_dicts,                 # Dictionary of dictionaries for prediction decoders.
                                                            # Must decode nothing more or less than everything in the list_of_posterior_input_encoder_class_dicts.
                                                            # Do NOT include lower_layer_posterior_sample. (This is automatically generated.)
                                                            # Keys:
                                                                # name.
                                                                # Keys:
                                                                    # decoding class. (These must have OPEN input_shape but fixed output_shape.
                                                                    # (They must also have loss-functions.)
                                                                
    lower_layer_posterior_sample_size = 0,                  # Size of lower_layer_posterior_sample.
    lower_layer_posterior_sample_decoding_output_size = 0,  # Width of THIS layer's inner state for the lower layer's sample.
                                                            # Needed whenever lower_layer_posterior_sample_size != 0.
    higher_layer_hidden_state_size = 0,                     # Size of hidden_state of higher_layer.
    time_constant = 1,
    verbose = False):
    
    # Make prior input encoder.
    list_of_prior_input_encoders = [Misc_Encoder('previous_hidden_state', hidden_state_size, verbose = verbose)]        # Start with encoder for previous hidden state.
    for prior_input_encoder_class_dict in dict_of_prior_input_encoder_class_dicts.values():                             # Add encoders for prior_input.
        list_of_prior_input_encoders.append(prior_input_encoder_class_dict['class']())                                            
    prior_input_encoder = Combiner('prior_input_encoder', list_of_prior_input_encoders, verbose = verbose)
    
    # Make posterior input encoder.
    list_of_posterior_input_encoders = [Misc_Encoder('previous_hidden_state', hidden_state_size, verbose = verbose)]    # Start with encoder for previous hidden state.
    for prior_input_encoder_class_dict in dict_of_prior_input_encoder_class_dicts.values():                             # Add encoders for prior_input.
        list_of_posterior_input_encoders.append(prior_input_encoder_class_dict['class']())        
    for posterior_input_encoder_class_dict in dict_of_posterior_input_encoder_class_dicts.values():                     # Add encoders for posterior_input.
        list_of_posterior_input_encoders.append(posterior_input_encoder_class_dict['class']())   
    if lower_layer_posterior_sample_size != 0:                                                                          # If available, add encoder for lower_layer_posterior_sample
        list_of_posterior_input_encoders.append(Misc_Encoder('lower_layer_posterior_sample', lower_layer_posterior_sample_size, verbose = verbose))                                    
    posterior_input_encoder = Combiner('posterior_input_encoder', list_of_posterior_input_encoders, verbose = verbose)
    
    # Everything the prediction decoder predicts needs an inner state to be decoded from
    # and that includes the lower layer's posterior sample when there is a lower layer.
    if set(dict_of_prediction_decoder_class_dicts) != set(dict_of_posterior_input_encoder_class_dicts):
        raise ValueError(
            "The prediction decoders must decode exactly the posterior input encoders."
            f"Only in prediction decoders: \t{set(dict_of_prediction_decoder_class_dicts) - set(dict_of_posterior_input_encoder_class_dicts)}"
            f"Only in posterior encoders: \t{set(dict_of_posterior_input_encoder_class_dicts) - set(dict_of_prediction_decoder_class_dicts)}")

    # Dictionaries of sizes.
    dict_of_inner_state_sizes = {
        name : dict_of_posterior_input_encoder_class_dicts[name]['decoding_output_size']
        for name in dict_of_prediction_decoder_class_dicts.keys()}

    if lower_layer_posterior_sample_size != 0:
        if lower_layer_posterior_sample_decoding_output_size == 0:
            raise ValueError(
                "A layer with a lower layer needs lower_layer_posterior_sample_decoding_output_size != 0.")
        dict_of_inner_state_sizes['lower_layer_posterior_sample'] = \
            lower_layer_posterior_sample_decoding_output_size

    if not dict_of_inner_state_sizes:
        raise ValueError(
            "This layer would have no inner state at all. Give it observations of its own, "
            "or a lower layer whose posterior sample it can summarise.")

    inner_state_size = sum(dict_of_inner_state_sizes.values())

    # Make prior inner state decoder.
    prior_input_encoding_size = prior_input_encoder.total_output_shape[-1]
    prior_inner_state_decoder = Divider(
        'prior_inner_state_decoder',
        [Inner_State_Decoder(name = name, input_size = prior_input_encoding_size, output_size = size)
         for name, size in dict_of_inner_state_sizes.items()],
        verbose = verbose)

    # Make posterior inner state decoder.
    posterior_input_encoding_size = posterior_input_encoder.total_output_shape[-1]

    dict_of_encoding_columns = {}
    column = 0
    for name, model in posterior_input_encoder.models_dict.items():
        width = model.output_shape[-1]
        dict_of_encoding_columns[name] = list(range(column, column + width))
        column += width

    shared_columns = [
        column
        for name, columns in dict_of_encoding_columns.items()
        if name not in dict_of_inner_state_sizes
        for column in columns]

    list_of_posterior_inner_state_decoders = []
    for name, size in dict_of_inner_state_sizes.items():
        list_of_posterior_inner_state_decoders.append(
            Sliced_Inner_State_Decoder(
                name = name,
                input_size = posterior_input_encoding_size,
                output_size = size,
                columns = sorted(shared_columns + dict_of_encoding_columns[name])))
    posterior_inner_state_decoder = Divider(
        'posterior_inner_state_decoder', list_of_posterior_inner_state_decoders, verbose = verbose)

    # Make prediction decoder.
    list_of_prediction_decoders = []   
    for name, prediction_decoder_class_dict in dict_of_prediction_decoder_class_dicts.items(): 
        list_of_prediction_decoders.append(
            prediction_decoder_class_dict['class'](
                input_size = inner_state_size,
                verbose = verbose))
    if lower_layer_posterior_sample_size != 0:
        list_of_prediction_decoders.append(  
            Misc_Decoder(
                name = 'lower_layer_posterior_sample',
                input_size = inner_state_size,
                output_size = lower_layer_posterior_sample_size,
                bounded = False,
                verbose = verbose))       
    prediction_decoder = Divider('prediction_decoder', list_of_prediction_decoders, verbose = verbose)
    
    # Make hidden_state encoder.
    list_of_hidden_state_input_encoders = [Misc_Encoder('inner_state_sample', inner_state_size, verbose = verbose)]
    if higher_layer_hidden_state_size != 0:
        list_of_hidden_state_input_encoders.append(Misc_Encoder('higher_layer_hidden_state', higher_layer_hidden_state_size, verbose = verbose))
    hidden_state_input_encoder = Combiner('hidden_state_input_encoder', list_of_hidden_state_input_encoders, verbose = verbose)
    
    # Make hidden_state decoder.
    hidden_state_input_encoding_size = hidden_state_input_encoder.total_output_shape[-1]
    hidden_state_decoder = Hidden_State_Cell(
        'hidden_state_decoder', hidden_state_input_encoding_size, hidden_state_size,
        time_constant = time_constant, verbose = verbose)
    
    # Put all of those things together in a world_model_layer.
    world_model_layer = World_Model_Layer(
        prior_input_encoder,            
        prior_inner_state_decoder,              
        
        posterior_input_encoder,       
        posterior_inner_state_decoder,              
        
        prediction_decoder,        
        
        hidden_state_input_encoder,     
        hidden_state_decoder,         
        
        time_constant = time_constant,
        verbose = verbose)
    
    world_model_layer.inner_state_size = inner_state_size
    world_model_layer.hidden_state_size = hidden_state_size
    world_model_layer.dict_of_encoding_columns = dict_of_encoding_columns

    return world_model_layer