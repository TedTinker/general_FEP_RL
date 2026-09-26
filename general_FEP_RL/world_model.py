#%% 
#------------------
# world_model.py provides an architecture for creating predictions of future observations
# based on multi-layer mtrnn. Actor and Critic use its hidden states.  
#------------------

import torch
from torch import nn

from general_FEP_RL.world_model_layer import make_world_model_layer



# The model itself.
class World_Model(nn.Module):
    
    def __init__(
            self,
            list_of_world_model_layers):
        
        super().__init__()

        self.list_of_world_model_layers = nn.ModuleList(list_of_world_model_layers)
        
        
        
    def forward_one_step(
            self,
            list_of_previous_hidden_states,
            list_of_prior_values_dicts,
            list_of_posterior_values_dicts,
            use_posterior = True,
            deterministic = False):      # Not using the posterior is like "dreams:" the hierarchy runs on its own predictions.

        layers = self.list_of_world_model_layers
        num_layers = len(layers)

        list_of_inner_states = []
        list_of_prior_samples = []
        list_of_posterior_samples = []
        list_of_prior_prediction_dicts = []
        list_of_posterior_prediction_dicts = []

        list_of_driving_samples = (
            list_of_posterior_samples if use_posterior else list_of_prior_samples)

        # From bottom to top.
        for i, world_model_layer in enumerate(layers):
            prior_values = {
                **list_of_prior_values_dicts[i],
                'previous_hidden_state' : list_of_previous_hidden_states[i]}

            if use_posterior:
                posterior_values = {
                    **list_of_posterior_values_dicts[i],
                    'previous_hidden_state' : list_of_previous_hidden_states[i]}
                if i > 0:
                    lower_layer_sample = list_of_posterior_samples[i - 1].detach()
                    posterior_values['lower_layer_posterior_sample'] = lower_layer_sample

                inner_states = world_model_layer.make_inner_states(prior_values, posterior_values, deterministic = deterministic)
                posterior_sample = world_model_layer.combine_inner_state_samples(inner_states, 'posterior')
                posterior_predictions = world_model_layer.make_predictions(posterior_sample)
            else:
                inner_states = world_model_layer.make_prior_inner_states(prior_values, deterministic = deterministic)
                posterior_sample = None
                posterior_predictions = {}

            prior_sample = world_model_layer.combine_inner_state_samples(inner_states, 'prior')

            list_of_inner_states.append(inner_states)
            list_of_prior_samples.append(prior_sample)
            list_of_posterior_samples.append(posterior_sample)
            list_of_prior_prediction_dicts.append(world_model_layer.make_predictions(prior_sample))
            list_of_posterior_prediction_dicts.append(posterior_predictions)

        list_of_driving_samples = (list_of_posterior_samples if use_posterior else list_of_prior_samples)

        # From top to bottom.
        list_of_new_hidden_states = [None] * num_layers
        for i in range(num_layers - 1, -1, -1):
            list_of_new_hidden_states[i] = layers[i].make_hidden_state(
                previous_hidden_state = list_of_previous_hidden_states[i],
                inner_state_sample = list_of_driving_samples[i],
                higher_layer_hidden_state = None if i == num_layers - 1 else list_of_new_hidden_states[i + 1])

        return {
            'list_of_hidden_states' : list_of_new_hidden_states,
            'list_of_inner_states' : list_of_inner_states,
            'list_of_prior_samples' : list_of_prior_samples,
            'list_of_posterior_samples' : list_of_posterior_samples,
            'list_of_prior_predictions' : list_of_prior_prediction_dicts,
            'list_of_posterior_predictions' : list_of_posterior_prediction_dicts}



    # Initiate with 0s.
    def start_hidden_states(self, batch_size, device = None, dtype = None):
        example_parameter = next(self.parameters())
        device = example_parameter.device if device is None else device
        dtype = example_parameter.dtype if dtype is None else dtype
        return [
            torch.zeros(
                batch_size, 1, world_model_layer.hidden_state_decoder.output_shape[0],
                device = device, dtype = dtype)
            for world_model_layer in self.list_of_world_model_layers]



    # Episode of steps with bottom-to-top and then top-to-bottom.
    def forward(
            self,
            list_of_lists_of_prior_values_dicts,
            list_of_lists_of_posterior_values_dicts,
            list_of_previous_hidden_states = None,      # Pass this in to continue an episode.
            use_posterior = True):

        episode_length = len(list_of_lists_of_prior_values_dicts)
        example_value = next(iter(list_of_lists_of_posterior_values_dicts[0][0].values()))

        if list_of_previous_hidden_states is None:
            list_of_previous_hidden_states = self.start_hidden_states(
                example_value.shape[0],
                device = example_value.device,
                dtype = example_value.dtype)

        list_of_step_dicts = []
        for t in range(episode_length):
            list_of_step_dicts.append(self.forward_one_step(
                list_of_previous_hidden_states,
                list_of_lists_of_prior_values_dicts[t],
                list_of_lists_of_posterior_values_dicts[t],
                use_posterior = use_posterior))
            list_of_previous_hidden_states = list_of_step_dicts[-1]['list_of_hidden_states']

        return list_of_step_dicts



######################

        

# Function to make the whole world model.
def make_world_model(
    hidden_state_sizes,
    
    list_of_dict_of_prior_input_encoder_class_dicts,                # List of dictionaries of dictionaries for prior_input encoders. (Inner state decoder is automatically generated.)
                                                                    # Do NOT include hidden_state encoding. (This is automatically generated.)
                                                                    # Keys:
                                                                        # name.
                                                                        # Keys:
                                                                            # class. (These must have fixed input_size and fixed output_size.)
                                                                            # (There is no decoding_output_size. Only inner_states shared with posterior_inner_states are decoded.)
                                                                
    list_of_dict_of_posterior_input_encoder_class_dicts,            # List of dictionaries of dictionaries for posterior_input encoders. (Inner state decoder is automatically generated.)
                                                                    # Do NOT include hidden_state encoding or lower_layer_posterior_sample encoding. (These are automatically generated.)
                                                                    # Do NOT include lower_layer_posterior_sample_output_size. (This is automatically generated.)
                                                                    # Keys:
                                                                        # name.
                                                                        # Keys:
                                                                            # class. (These must have fixed input_shape and fixed output_shape.)
                                                                            # decoding_output_size.
                                                                
    list_of_dict_of_prediction_decoder_class_dicts,                 # List of dictionaries of dictionaries for prediction decoders.
                                                                    # Must decode nothing more or less than everything in the list_of_posterior_input_encoder_class_dicts.
                                                                    # Do NOT include lower_layer_posterior_sample. (This is automatically generated.)
                                                                    # Keys:
                                                                        # name.
                                                                        # Keys:
                                                                            # decoding class. (These must have OPEN input_shape but fixed output_shape.
                                                                            # (They must also have loss-functions.)
                                                                
    lower_layer_posterior_sample_decoding_output_sizes,             # Size of each layer's inner state, for inputs from the layer below.
                                                                    # Entry 0 is ignored: layer 0 has no lower layer.
    time_constants,
    verbose = False):
    
    
    
    # TEST: Are length for the layers consistent?
    all_same = all(len(l) == len(hidden_state_sizes) for l in [
        list_of_dict_of_prior_input_encoder_class_dicts, 
        list_of_dict_of_posterior_input_encoder_class_dicts,
        list_of_dict_of_prediction_decoder_class_dicts,
        lower_layer_posterior_sample_decoding_output_sizes,
        time_constants])
    
    if not all_same: 
        raise ValueError("Inputs of make_world_model need to share length.")
    
    list_of_world_model_layers = []
    
    # Each layer's inner_state_size is what the layer above sees,
    # so it is read off the new layer rather than trusted.
    lower_layer_posterior_sample_size = 0
    
    # For each layer:
    for i in range(len(hidden_state_sizes)):
        
        # Check for higher layer.
        higher_layer_hidden_state_size = 0
        if i < len(hidden_state_sizes)-1:
            higher_layer_hidden_state_size = hidden_state_sizes[i+1]
        
        # Make world_model_layer, as described in world_model_layer.py.
        world_model_layer = make_world_model_layer(
            hidden_state_sizes[i],                                      
            
            list_of_dict_of_prior_input_encoder_class_dicts[i],                
            list_of_dict_of_posterior_input_encoder_class_dicts[i],
            list_of_dict_of_prediction_decoder_class_dicts[i],                
                                                                        
            lower_layer_posterior_sample_size = lower_layer_posterior_sample_size,                  # Size of lower_layer_posterior_sample.
            lower_layer_posterior_sample_decoding_output_size = (
                0 if i == 0 else lower_layer_posterior_sample_decoding_output_sizes[i]),
            higher_layer_hidden_state_size = higher_layer_hidden_state_size,                        # Size of hidden_state of higher_layer.
            time_constant = time_constants[i],
            verbose = verbose)
    
        list_of_world_model_layers.append(world_model_layer)
        lower_layer_posterior_sample_size = world_model_layer.inner_state_size
    
    world_model = World_Model(list_of_world_model_layers)

    return world_model