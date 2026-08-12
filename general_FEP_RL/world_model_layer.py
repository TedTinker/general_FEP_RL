#%% 
#------------------
# world_model.py provides an architecture for creating predictions of future observations
# based on multi-layer mtrnn. Actor and Critic utilize its hidden states.  
#------------------

import math 
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary

from general_FEP_RL.shape_to_shape_models import Shape_to_Shape_Model, Combinor, Divider
from general_FEP_RL.encoder_decoder import Misc_Encoder, Misc_Decoder, Inner_State_Decoder, Sliced_Inner_State_Decoder, Hidden_State_Decoder

from general_FEP_RL.utils import calculate_dkl



class World_Model_Layer(nn.Module):
    
    def __init__(
            self,
            prior_input_encoder,            # Combinor (encodes values, including previous hidden_state).
            prior_inner_state_decoder,      # Divider (makes prior inner_states for everything in prediction_decoder).
            
            posterior_input_encoder,        # Combinor (entodes values, including everything in prior_input_encoder and perhaps lower_layer_posterior_sample).
            posterior_inner_state_decoder,  # Divider (makes inner_states, perhaps including lower_layer_posterior_sample).
            
            prediction_decoder,             # Divider (prediction of posterior input values).
            
            hidden_state_input_encoder,     # Combinor (encodes posterior_sample, and perhaps higher_layer_hidden_state)
            hidden_state_decoder,           # Shape_to_Shape_Model (makes hidden_state).
            
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
        
        self.new = 1.0 / time_constant
        self.old = 1.0 - self.new
        
        
        
    def forward(self):
        pass # I don't think anything is really needed here.
    
    
    
    def make_inner_states(self, prior_value_dict, posterior_value_dict):
        encoding = self.prior_input_encoder(prior_value_dict)                   # Encodes values.   
        prior_inner_states = self.prior_inner_state_decoder(encoding)           # Decodes (mu, std, sample) for prior_value.
        
        encoding = self.posterior_input_encoder(posterior_value_dict)           # Encodes values.   
        posterior_inner_states = self.posterior_inner_state_decoder(encoding)   # Decodes (mu, std, sample) for posterior_value.
        
        inner_states = {
            name : {
            'prior_mu' : prior_inner_states[name]['mu'],
            'prior_std' : prior_inner_states[name]['std'],
            'prior_sample' : prior_inner_states[name]['sample'],
            'posterior_mu' : posterior_inner_states[name]['mu'],
            'posterior_std' : posterior_inner_states[name]['std'],
            'posterior_sample' : posterior_inner_states[name]['sample'],
            'dkl' : calculate_dkl(
                posterior_inner_states[name]['mu'],
                posterior_inner_states[name]['std'],
                prior_inner_states[name]['mu'],
                prior_inner_states[name]['std'])}
            for name in prior_inner_states.keys()}
        
        return inner_states
    
    
    
    def combine_inner_state_samples(self, inner_states, prior_or_posterior):
        # No sorting here. Divider sorted once at construction; models_dict replays
        # that order, so this can never disagree with how the layer was sized.
        key = f'{prior_or_posterior}_sample'
        return torch.cat(
            [inner_states[name][key] for name in self.posterior_inner_state_decoder.models_dict.keys()],
            dim = -1)
        
        
    
    def make_predictions(self, inner_state_sample):
        predictions = self.prediction_decoder(inner_state_sample)
        return predictions
    
    
    
    def make_hidden_state(self, previous_hidden_state, inner_state_sample, higher_layer_hidden_state = None):
        value_dict = {'inner_state_sample' : inner_state_sample}
        if higher_layer_hidden_state is not None:
            value_dict['higher_layer_hidden_state'] = higher_layer_hidden_state
        encoding = self.hidden_state_input_encoder(value_dict)
        new_hidden_state = self.hidden_state_decoder(encoding)
        print(f"\n\n{new_hidden_state.shape, previous_hidden_state.shape}\n\n")
        new_hidden_state = self.new * new_hidden_state + self.old * previous_hidden_state
        return new_hidden_state
        
    
    
######################



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
    prior_input_encoder = Combinor('prior_input_encoder', list_of_prior_input_encoders, verbose = verbose)
    
    # Make posterior input encoder.
    list_of_posterior_input_encoders = [Misc_Encoder('previous_hidden_state', hidden_state_size, verbose = verbose)]    # Start with encoder for previous hidden state.
    for prior_input_encoder_class_dict in dict_of_prior_input_encoder_class_dicts.values():                             # Add encoders for prior_input.
        list_of_posterior_input_encoders.append(prior_input_encoder_class_dict['class']())        
    for posterior_input_encoder_class_dict in dict_of_posterior_input_encoder_class_dicts.values():                     # Add encoders for posterior_input.
        list_of_posterior_input_encoders.append(posterior_input_encoder_class_dict['class']())   
    if lower_layer_posterior_sample_size != 0:                                                                          # If available, add encoder for lower_layer_posterior_sample
        list_of_posterior_input_encoders.append(Misc_Encoder('lower_layer_posterior_sample', lower_layer_posterior_sample_size, verbose = verbose))                                    
    posterior_input_encoder = Combinor('posterior_input_encoder', list_of_posterior_input_encoders, verbose = verbose)
    
    # Which inner states does this layer have, and how wide is each?
    #
    # Everything the prediction decoder predicts needs an inner state to be decoded
    # from, and that includes the lower layer's posterior sample when there is a
    # lower layer. A layer with no senses of its own is therefore legal: its only
    # inner state summarises the layer below it.
    if set(dict_of_prediction_decoder_class_dicts) != set(dict_of_posterior_input_encoder_class_dicts):
        raise ValueError(
            f"""
The prediction decoders must decode exactly the posterior input encoders.
Only in prediction decoders: \t{set(dict_of_prediction_decoder_class_dicts) - set(dict_of_posterior_input_encoder_class_dicts)}
Only in posterior encoders: \t{set(dict_of_posterior_input_encoder_class_dicts) - set(dict_of_prediction_decoder_class_dicts)}
            """)

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
    #
    # The Combinor concatenates its sub-models' outputs in models_dict order, so the
    # posterior encoding is a run of known-width blocks and each modality's block can
    # be located by name. Anything that is not itself an inner state -- the previous
    # hidden_state and the prior inputs -- is shared context every modality may read.
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
        list_of_prediction_decoders.append(   # POTENTIAL ISSUE: the decoder outputs tanh, while the posterior_same is mu + e*std
            Misc_Decoder(
                name = 'lower_layer_posterior_sample',
                input_size = inner_state_size,
                output_size = lower_layer_posterior_sample_size,
                verbose = verbose))       
    prediction_decoder = Divider('prediction_decoder', list_of_prediction_decoders, verbose = verbose)
    
    # Make hidden_state encoder.
    list_of_hidden_state_input_encoders = [Misc_Encoder('inner_state_sample', inner_state_size, verbose = verbose)]
    if higher_layer_hidden_state_size != 0:
        list_of_hidden_state_input_encoders.append(Misc_Encoder('higher_layer_hidden_state', higher_layer_hidden_state_size, verbose = verbose))
    hidden_state_input_encoder = Combinor('hidden_state_input_encoder', list_of_hidden_state_input_encoders, verbose = verbose)
    
    # Make hidden_state decoder.
    hidden_state_input_encoding_size = hidden_state_input_encoder.total_output_shape[-1]
    hidden_state_decoder = Hidden_State_Decoder('hidden_state_decoder', hidden_state_input_encoding_size, hidden_state_size, verbose = verbose)

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



######################



if __name__ == '__main__':
    
    
    
    print("\n\n\n\n\n\n\n\n\n\n")
    
        
    
    ######################
    # Concrete encoders and decoders to hand to the class-dicts.
    #
    # Encoders need fixed input_shape AND fixed output_shape, and get built with no
    # arguments, so their sizes are bound ahead of time with functools.partial.
    # Prediction decoders need an OPEN input_shape (it depends on inner_state_size,
    # which make_world_model_layer computes), so only their output_shape is bound.
    ######################
    
    
    class Vector_Encoder(Shape_to_Shape_Model):
    
        def __init__(self, name, input_size, output_size, hidden_size = 32, verbose = False):
            super().__init__(
                name = name,
                input_shape = (input_size,),
                output_shape = (output_size,),
                arg_dict = {'hidden_size' : hidden_size},
                verbose = verbose)
    
        def build_model(self, arg_dict):
            hidden_size = arg_dict.get('hidden_size', 32)
            self.model = nn.Sequential(
                nn.Linear(self.input_shape[0], hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, self.output_shape[0]),
                nn.LeakyReLU())
    
        def forward(self, value):
            return self.model(value)
    
    
    class Image_Encoder(Shape_to_Shape_Model):
    
        def __init__(self, name, input_shape, output_size, hidden_channels = [16, 32], verbose = False):
            super().__init__(
                name = name,
                input_shape = input_shape,
                output_shape = (output_size,),
                arg_dict = {'hidden_channels' : hidden_channels},
                verbose = verbose)
    
        def build_model(self, arg_dict):
            hidden_channels = arg_dict.get('hidden_channels', [16, 32])
            in_channels, in_height, in_width = self.input_shape
            channels = [in_channels] + hidden_channels
    
            layers = []
            for in_ch, out_ch in zip(channels[:-1], channels[1:]):
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1))
                layers.append(nn.LeakyReLU())
            self.model = nn.Sequential(*layers)
    
            self.end_shape = (
                hidden_channels[-1],
                in_height // 2**len(hidden_channels),
                in_width // 2**len(hidden_channels))
            self.linear = nn.Linear(math.prod(self.end_shape), self.output_shape[0])
    
        def forward(self, value):
            batch_size, episode_length = value.shape[:2]
            value = value.reshape(batch_size * episode_length, *self.input_shape)
            value = self.model(value).reshape(batch_size * episode_length, -1)
            encoding = self.linear(value)
            return encoding.reshape(batch_size, episode_length, self.output_shape[0])
    
    
    class Vector_Decoder(Shape_to_Shape_Model):
    
        def __init__(self, name, input_size, output_size, hidden_size = 32, verbose = False):
            super().__init__(
                name = name,
                input_shape = (input_size,),
                output_shape = (output_size,),
                arg_dict = {'hidden_size' : hidden_size},
                verbose = verbose)
    
        def build_model(self, arg_dict):
            hidden_size = arg_dict.get('hidden_size', 32)
            self.model = nn.Sequential(
                nn.Linear(self.input_shape[0], hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, self.output_shape[0]))
    
        def forward(self, value):
            return self.model(value)
        
        @staticmethod
        def loss_func(predicted_values, target_values):
            loss_value = F.mse_loss(predicted_values, target_values, reduction = 'none')
            return loss_value
    
    
    class Image_Decoder(Shape_to_Shape_Model):
    
        def __init__(self, name, input_size, output_shape, hidden_size = 64, verbose = False):
            super().__init__(
                name = name,
                input_shape = (input_size,),
                output_shape = output_shape,
                arg_dict = {'hidden_size' : hidden_size},
                verbose = verbose)
    
        def build_model(self, arg_dict):
            hidden_size = arg_dict.get('hidden_size', 64)
            self.model = nn.Sequential(
                nn.Linear(self.input_shape[0], hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, math.prod(self.output_shape)))
    
        def forward(self, value):
            batch_size, episode_length = value.shape[:2]
            output = self.model(value)
            return output.reshape(batch_size, episode_length, *self.output_shape)

        @staticmethod
        def loss_func(predicted_values, target_values):
            loss_value = F.mse_loss(predicted_values, target_values, reduction = 'none')
            return loss_value
        
        

    hidden_state_size = 32

    action_shape = (4,)
    vision_shape = (3, 16, 16)
    touch_shape  = (6,)

    dict_of_prior_input_encoder_class_dicts = {
        'action'  : 
            {
                'class' : partial(
                    Vector_Encoder,
                     name = 'action',
                     input_size = action_shape[0],
                     output_size = 16)}}

    dict_of_posterior_input_encoder_class_dicts = {
        'vision' :
            {
                'class' : partial(
                     Image_Encoder,
                     name = 'vision',
                     input_shape = vision_shape,
                     output_size = 32),
                 'decoding_output_size' : 16},

        'touch' :
            {
                'class' : partial(
                     Vector_Encoder,
                     name = 'touch',
                     input_size = touch_shape[0],
                     output_size = 16),
                 'decoding_output_size' : 8}}

    dict_of_prediction_decoder_class_dicts = {
        'vision' : 
            {
                'class' : partial(
                     Image_Decoder,
                     name = 'vision',
                     output_shape = vision_shape)},

        'touch' : 
            {
                 'class' : partial(
                     Vector_Decoder,
                     name = 'touch',
                     output_size = touch_shape[0])}}

    world_model_layer = make_world_model_layer(
        hidden_state_size = hidden_state_size,
        dict_of_prior_input_encoder_class_dicts = dict_of_prior_input_encoder_class_dicts,
        dict_of_posterior_input_encoder_class_dicts = dict_of_posterior_input_encoder_class_dicts,
        dict_of_prediction_decoder_class_dicts =dict_of_prediction_decoder_class_dicts,
        lower_layer_posterior_sample_size = 0,       # Bottom layer, so no lower layer.
        higher_layer_hidden_state_size = 0,          # Only layer, so no higher layer.
        verbose = True)

    print('\n\n')
    print(world_model_layer)
    print()



    ######################
    # One time-step through the layer, following the diagram left to right.
    ######################

    batch_size = 2
    episode_length = 3

    previous_hidden_state = torch.zeros(batch_size, episode_length, hidden_state_size)

    prior_value_dict = {
        'previous_hidden_state' : previous_hidden_state,
        'action'                : torch.zeros(batch_size, episode_length, *action_shape)}

    posterior_value_dict = {
        **prior_value_dict,
        'vision' : torch.zeros(batch_size, episode_length, *vision_shape),
        'touch'  : torch.zeros(batch_size, episode_length, *touch_shape)}

    # Prior mu/std and post mu/std.
    all_inner_state_dicts = world_model_layer.make_inner_states(
        prior_value_dict, posterior_value_dict)

    for name, inner_state_dict in all_inner_state_dicts.items():
        print(f"'{name}':")
        for key, value in inner_state_dict.items():
            print(f"\t{key}: \t{list(value.shape)}")
            
    # Prior and Post Sample: the layer owns the ordering contract.
    inner_state_sample_prior = world_model_layer.combine_inner_state_samples(
        all_inner_state_dicts, 'prior')
    inner_state_sample_posterior = world_model_layer.combine_inner_state_samples(
        all_inner_state_dicts, 'posterior')
    print(f"\nprior inner_state_sample: \t{list(inner_state_sample_prior.shape)}")
    print(f"posterior inner_state_sample: \t{list(inner_state_sample_posterior.shape)}")

    # Predictions with the prior sample.
    predictions = world_model_layer.make_predictions(inner_state_sample_prior)
    for name, prediction in predictions.items():
        print(f"prior prediction '{name}': \t{list(prediction.shape)}")

    # Predictions with the posterior sample.
    predictions = world_model_layer.make_predictions(inner_state_sample_prior)
    for name, prediction in predictions.items():
        print(f"posterior prediction '{name}': \t{list(prediction.shape)}")

    # Hidden state for time t.
    new_hidden_state = world_model_layer.make_hidden_state(
        previous_hidden_state = previous_hidden_state,
        inner_state_sample = inner_state_sample_posterior,
        higher_layer_hidden_state = None)
    print(f"\nnew_hidden_state: \t{list(new_hidden_state.shape)}")



    ######################
    # The two red arrows in the diagram.
    ######################

    complexity = sum(
        inner_state_dict['dkl'].mean()
        for inner_state_dict in all_inner_state_dicts.values())

    accuracy = sum(
        nn.functional.mse_loss(predictions[name], posterior_value_dict[name])
        for name in predictions.keys())

    print(f"\ncomplexity (dkl): \t{complexity.item():.5f}")
    print(f"accuracy (mse): \t{accuracy.item():.5f}")