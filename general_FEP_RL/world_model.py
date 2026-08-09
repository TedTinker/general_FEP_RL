#%% 
#------------------
# world_model.py provides an architecture for creating predictions of future observations
# based on multi-layer mtrnn. Actor and Critic utilize its hidden states.  
#------------------

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary

from world_model_layer import World_Model_Layer, make_world_model_layer

from general_FEP_RL.utils_torch import init_weights, parametrize_normal, sample, calculate_dkl, generate_dummy_inputs



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
            list_of_posterior_values_dicts):

        layers = self.list_of_world_model_layers
        num_layers = len(layers)

        list_of_inner_states = []
        list_of_posterior_samples = []
        list_of_prior_prediction_dicts = []
        list_of_posterior_prediction_dicts = []

        # From bottom to top.
        for i, world_model_layer in enumerate(layers):
            prior_values = {
                **list_of_prior_values_dicts[i],
                'previous_hidden_state' : list_of_previous_hidden_states[i]}
            posterior_values = {
                **list_of_posterior_values_dicts[i],
                'previous_hidden_state' : list_of_previous_hidden_states[i]}
            if i > 0:
                posterior_values['lower_layer_posterior_sample'] = list_of_posterior_samples[i - 1]

            inner_states = world_model_layer.make_inner_states(prior_values, posterior_values)
            prior_sample = world_model_layer.combine_inner_state_samples(inner_states, 'prior')
            posterior_sample = world_model_layer.combine_inner_state_samples(inner_states, 'posterior')

            list_of_inner_states.append(inner_states)
            list_of_posterior_samples.append(posterior_sample)
            list_of_prior_prediction_dicts.append(world_model_layer.make_predictions(prior_sample))             # WE SHOULD BE USING SOMETHING ELSE
            list_of_posterior_prediction_dicts.append(world_model_layer.make_predictions(posterior_sample))     # FOR PREDICTION INPUTS.

        # From top to bottom.
        list_of_new_hidden_states = [None] * num_layers
        for i in range(num_layers - 1, -1, -1):
            list_of_new_hidden_states[i] = layers[i].make_hidden_state(
                previous_hidden_state = list_of_previous_hidden_states[i],
                inner_state_sample = list_of_posterior_samples[i],
                higher_layer_hidden_state = None if i == num_layers - 1 else list_of_new_hidden_states[i + 1])

        return {
            'list_of_hidden_states' : list_of_new_hidden_states,
            'list_of_inner_states' : list_of_inner_states,
            'list_of_posterior_samples' : list_of_posterior_samples,
            'list_of_prior_predictions' : list_of_prior_prediction_dicts,
            'list_of_posterior_predictions' : list_of_posterior_prediction_dicts}



    def start_hidden_states(self, batch_size, device = None, dtype = None):
        example_parameter = next(self.parameters())
        device = example_parameter.device if device is None else device
        dtype = example_parameter.dtype if dtype is None else dtype
        return [
            torch.zeros(
                batch_size, 1, world_model_layer.hidden_state_decoder.output_shape[0],
                device = device, dtype = dtype)
            for world_model_layer in self.list_of_world_model_layers]



    def forward(
            self,
            list_of_lists_of_prior_values_dicts,
            list_of_lists_of_posterior_values_dicts,
            list_of_previous_hidden_states = None):     # Pass this in to continue an episode.

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
                list_of_lists_of_posterior_values_dicts[t]))
            list_of_previous_hidden_states = list_of_step_dicts[-1]['list_of_hidden_states']

        return list_of_step_dicts



######################

        

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
                                                                
    lower_layer_posterior_sample_decoding_output_sizes,             # Width of each layer's inner state for the layer below it.
                                                                    # Entry 0 is ignored: layer 0 has no lower layer.
    time_constants,
    isolate_modality_posteriors = True,                             # Each modality's posterior reads only shared context
                                                                    # plus its own encoding.
    verbose = False):
    
    
    
    all_same = all(len(l) == len(hidden_state_sizes) for l in [
        list_of_dict_of_prior_input_encoder_class_dicts, 
        list_of_dict_of_posterior_input_encoder_class_dicts,
        list_of_dict_of_prediction_decoder_class_dicts,
        lower_layer_posterior_sample_decoding_output_sizes,
        time_constants])
    
    if not all_same: 
        raise ValueError("Inputs of make_world_model need to share length.")
    
    list_of_world_model_layers = []
    
    # Each layer's inner_state_size is what the layer above sees, so it is read off
    # the layer just built rather than passed in and trusted.
    lower_layer_posterior_sample_size = 0
    
    for i in range(len(hidden_state_sizes)):
        
        higher_layer_hidden_state_size = 0
        if i < len(hidden_state_sizes)-1:
            higher_layer_hidden_state_size = hidden_state_sizes[i+1]
        
        world_model_layer = make_world_model_layer(
            hidden_state_sizes[i],                                      
            
            list_of_dict_of_prior_input_encoder_class_dicts[i],                
            list_of_dict_of_posterior_input_encoder_class_dicts[i],
            list_of_dict_of_prediction_decoder_class_dicts[i],                
                                                                        
            lower_layer_posterior_sample_size = lower_layer_posterior_sample_size,                  # Size of lower_layer_posterior_sample.
            lower_layer_posterior_sample_decoding_output_size = (
                0 if i == 0 else lower_layer_posterior_sample_decoding_output_sizes[i]),
            higher_layer_hidden_state_size = higher_layer_hidden_state_size,                        # Size of hidden_state of higher_layer.
            isolate_modality_posteriors = isolate_modality_posteriors,
            time_constant = time_constants[i],
            verbose = verbose)
    
        list_of_world_model_layers.append(world_model_layer)
        lower_layer_posterior_sample_size = world_model_layer.inner_state_size
    
    world_model = World_Model(list_of_world_model_layers)

    return world_model



######################



if __name__ == '__main__':



    import math
    from functools import partial

    import torch
    from torch import nn
    import torch.nn.functional as F

    from shape_to_shape_models import Shape_to_Shape_Model

    print("\n\n\n\n\n\n\n\n\n\n")



    ######################
    # Concrete encoders and decoders, same as in world_model_layer.py's example.
    #
    # Encoders need fixed input_shape AND fixed output_shape and are built with no
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
            return F.mse_loss(predicted_values, target_values, reduction = 'none')


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
            return F.mse_loss(predicted_values, target_values, reduction = 'none')



    ######################
    # A three-layer hierarchy.
    #
    #   layer 0 (fast):     acts, sees vision and touch.            time_constant 1
    #   layer 1 (medium):   no senses of its own, only layer 0.      time_constant 4
    #   layer 2 (slow):     sees a task command, plus layer 1.       time_constant 16
    #
    # Layer 0 has no lower layer, layer 2 has no higher layer, and layer 1 has both,
    # so every branch in make_world_model_layer gets exercised. Layers 1 and 2 have
    # empty prior input dicts: their priors are driven by their own hidden states
    # alone. Layer 1 shows a layer with no observations, layer 2 shows one that mixes
    # its own observation with a summary of the layer below.
    ######################

    action_shape  = (4,)
    vision_shape  = (3, 16, 16)
    touch_shape   = (6,)
    command_shape = (8,)

    hidden_state_sizes = [32, 24, 16]
    time_constants     = [1, 4, 16]

    list_of_dict_of_prior_input_encoder_class_dicts = [
        # Layer 0: acts on the world.
        {'action' : {
            'class' : partial(Vector_Encoder, name = 'action', input_size = action_shape[0], output_size = 16)}},
        # Layer 1 and 2: prior is driven by their own hidden state alone.
        {},
        {}]

    list_of_dict_of_posterior_input_encoder_class_dicts = [
        # Layer 0.
        {'vision' : {
            'class' : partial(Image_Encoder, name = 'vision', input_shape = vision_shape, output_size = 32),
            'decoding_output_size' : 16},
         'touch' : {
            'class' : partial(Vector_Encoder, name = 'touch', input_size = touch_shape[0], output_size = 16),
            'decoding_output_size' : 8}},
        # Layer 1: no senses of its own, only the layer below.
        {},
        # Layer 2: a task command, plus the layer below.
        {'command' : {
            'class' : partial(Vector_Encoder, name = 'command', input_size = command_shape[0], output_size = 16),
            'decoding_output_size' : 8}}]

    list_of_dict_of_prediction_decoder_class_dicts = [
        # Layer 0.
        {'vision' : {
            'class' : partial(Image_Decoder, name = 'vision', output_shape = vision_shape)},
         'touch' : {
            'class' : partial(Vector_Decoder, name = 'touch', output_size = touch_shape[0])}},
        # Layer 1: the lower_layer_posterior_sample decoder is generated for it.
        {},
        # Layer 2.
        {'command' : {
            'class' : partial(Vector_Decoder, name = 'command', output_size = command_shape[0])}}]

    # How wide each layer's summary of the layer below is. Entry 0 is unused.
    lower_layer_posterior_sample_decoding_output_sizes = [0, 12, 8]

    world_model = make_world_model(
        hidden_state_sizes,
        list_of_dict_of_prior_input_encoder_class_dicts,
        list_of_dict_of_posterior_input_encoder_class_dicts,
        list_of_dict_of_prediction_decoder_class_dicts,
        lower_layer_posterior_sample_decoding_output_sizes,
        time_constants,
        verbose = False)

    print("inner_state_size per layer:",
          [layer.inner_state_size for layer in world_model.list_of_world_model_layers])

    print(f"\nparameters: {sum(p.numel() for p in world_model.parameters()):,}\n")



    ######################
    # A dummy episode.
    #
    # forward expects list_of_lists_of_..._values_dicts[time][layer], and every tensor
    # is (batch, 1, ...) because the hidden state is recurrent and one call to
    # forward_one_step covers exactly one time-step.
    ######################

    batch_size = 2
    episode_length = 5

    def make_step_values():
        action  = torch.randn(batch_size, 1, *action_shape)
        vision  = torch.randn(batch_size, 1, *vision_shape)
        touch   = torch.randn(batch_size, 1, *touch_shape)
        command = torch.randn(batch_size, 1, *command_shape)
        prior_values_dicts = [
            {'action' : action},
            {},
            {}]
        posterior_values_dicts = [
            {'action' : action, 'vision' : vision, 'touch' : touch},
            {},
            {'command' : command}]
        return prior_values_dicts, posterior_values_dicts

    list_of_lists_of_prior_values_dicts = []
    list_of_lists_of_posterior_values_dicts = []
    for _ in range(episode_length):
        prior_values_dicts, posterior_values_dicts = make_step_values()
        list_of_lists_of_prior_values_dicts.append(prior_values_dicts)
        list_of_lists_of_posterior_values_dicts.append(posterior_values_dicts)



    ######################
    # One step, printing every shape the diagram names.
    ######################

    print("###\nOne step\n###\n")

    previous_hidden_states = [
        torch.zeros(batch_size, 1, hidden_state_size)
        for hidden_state_size in hidden_state_sizes]

    step_dict = world_model.forward_one_step(
        previous_hidden_states,
        list_of_lists_of_prior_values_dicts[0],
        list_of_lists_of_posterior_values_dicts[0])

    for i in range(len(hidden_state_sizes)):
        print(f"layer {i}:")
        print(f"\thidden state: \t\t{list(step_dict['list_of_hidden_states'][i].shape)}")
        for name, inner_state_dict in step_dict['list_of_inner_states'][i].items():
            for key, value in inner_state_dict.items():
                print(f"\tinner state '{name}' {key}: \t{list(value.shape)}")
        print(f"\tposterior sample: \t{list(step_dict['list_of_posterior_samples'][i].shape)}")
        for name, prediction in step_dict['list_of_prior_predictions'][i].items():
            print(f"\tprior prediction '{name}': \t{list(prediction.shape)}")
        for name, prediction in step_dict['list_of_posterior_predictions'][i].items():
            print(f"\tposterior prediction '{name}': \t{list(prediction.shape)}")
        print()



    ######################
    # A whole episode.
    ######################

    print("###\nWhole episode\n###\n")

    list_of_step_dicts = world_model(
        list_of_lists_of_prior_values_dicts,
        list_of_lists_of_posterior_values_dicts)

    print(f"steps returned: {len(list_of_step_dicts)} (expected {episode_length})")

    # The MTRNN leak is h(t) = (1/time_constant) * decoded + (1 - 1/time_constant) * h(t-1).
    # Calling make_hidden_state twice with the same inner_state_sample but different
    # previous hidden states isolates the leak weight exactly, which is a sharper test
    # than watching how much the hidden state happens to move.
    print("\nMTRNN leak check:")
    for i, (world_model_layer, time_constant) in enumerate(
            zip(world_model.list_of_world_model_layers, time_constants)):
        inner_state_sample = list_of_step_dicts[0]['list_of_posterior_samples'][i]
        previous_hidden_state = torch.ones(batch_size, 1, hidden_state_sizes[i])
        higher_layer_hidden_state = (
            None if i == len(time_constants) - 1
            else list_of_step_dicts[0]['list_of_hidden_states'][i + 1])
        with torch.no_grad():
            from_zero = world_model_layer.make_hidden_state(
                torch.zeros_like(previous_hidden_state), inner_state_sample, higher_layer_hidden_state)
            from_previous = world_model_layer.make_hidden_state(
                previous_hidden_state, inner_state_sample, higher_layer_hidden_state)
        measured = (from_previous - from_zero).mean().item()
        expected = 1 - 1 / time_constant
        print(f"\tlayer {i} (time_constant {time_constant:>2}): "
              f"measured old-weight {measured:.4f}, expected {expected:.4f}")

    ######################
    # The two red arrows in the diagram, summed over the episode.
    #
    # Accuracy compares predictions to the values the posterior actually saw. The
    # lower_layer_posterior_sample target is itself a network output, so it is
    # detached: without that, the lower layer learns to be predictable rather than
    # informative.
    ######################

    print("\n###\nFree energy\n###\n")

    accuracy = 0.
    complexity = 0.

    for t, step_dict in enumerate(list_of_step_dicts):
        for i, world_model_layer in enumerate(world_model.list_of_world_model_layers):

            for name, prediction in step_dict['list_of_prior_predictions'][i].items():
                if name == 'lower_layer_posterior_sample':
                    target = step_dict['list_of_posterior_samples'][i-1].detach()
                else:
                    target = list_of_lists_of_posterior_values_dicts[t][i][name]
                loss_func = world_model_layer.prediction_decoder.models_dict[name].loss_func
                accuracy = accuracy + loss_func(prediction, target).mean()

            for name, inner_state_dict in step_dict['list_of_inner_states'][i].items():
                complexity = complexity + inner_state_dict['dkl'].mean()

    free_energy = accuracy + complexity
    print(f"accuracy: \t{accuracy.item():.5f}")
    print(f"complexity: \t{complexity.item():.5f}")
    print(f"free energy: \t{free_energy.item():.5f}")



    ######################
    # Does every parameter actually receive a gradient? A parameter with no gradient
    # is a branch of the diagram that nothing is training.
    ######################

    print("\n###\nGradient check\n###\n")

    free_energy.backward()

    without_gradient = [
        name for name, parameter in world_model.named_parameters()
        if parameter.grad is None]

    if without_gradient:
        print(f"{len(without_gradient)} parameters received NO gradient:")
        for name in without_gradient:
            print(f"\t{name}")
    else:
        print("every parameter received a gradient.")