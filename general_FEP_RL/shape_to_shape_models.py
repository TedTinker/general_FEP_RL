import math
from collections import Counter

import torch
from torch import nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity



class Shape_to_Shape_Model(nn.Module):

    def __init__(
            self,      
            name,               # String.
            input_shape,        # Tuple like a torch shape, not including batch_size and episode_length. 
                                # Example: (64,).
                                # Example: (16, 4, 4). 
            output_shape,       # Another tuple like that.
            
            arg_dict = {},      # Anything the sub-class needs for building or running. 
            verbose = False):   # Add print-outs.
    
        super().__init__()
        
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.arg_dict = arg_dict
        self.build_model(arg_dict)
        
        if verbose:
            self.print_examples()
        
    # Using sub-models, change these functions.
    def build_model(self, arg_dict):
        raise NotImplementedError("Subclasses must implement this method")
    
    def forward(self, value):
        raise NotImplementedError("Subclasses must implement this method")
        
    def make_examples(self, batch_size = 1, episode_length = 1):
        return(
            torch.zeros(batch_size, episode_length, *self.input_shape),
            torch.zeros(batch_size, episode_length, *self.output_shape))
    
    def print_examples(self):
        example_input, example_output = self.make_examples()
        print(
            f"""
{self.name} Shape_to_Shape_Model:
\texample input: \t\t{list(example_input.shape)}
\texample output: \t{list(example_output.shape)}
               """)

    
    
if __name__ == '__main__':
    
    
    
    print("\n\n\n\n\n\n\n\n\n\n")
    
    
    
    class ExampleEncoder(Shape_to_Shape_Model):
        # input_shape is assumed to be (channels, height, width).
        def build_model(self, arg_dict = {'hidden_channels' : [32, 64, 128]}):
            in_channels, in_height, in_width = self.input_shape
            hidden_channels = arg_dict['hidden_channels']
            downsamples = len(hidden_channels)        # Each conv halves H and W.

            end_height = in_height // 2**downsamples
            end_width  = in_width  // 2**downsamples
            assert end_height >= 1 and end_width >= 1, \
                f"{self.input_shape} too small for {downsamples} downsamples."
            self.end_shape = (hidden_channels[-1], end_height, end_width)

            # Image -> feature map.
            channels = [in_channels] + hidden_channels
            layers = []
            for in_ch, out_ch in zip(channels[:-1], channels[1:]):
                layers.append(nn.Conv2d(
                    in_channels = in_ch,
                    out_channels = out_ch,
                    kernel_size = 4,
                    stride = 2,
                    padding = 1))
                layers.append(nn.LeakyReLU())
            self.model = nn.Sequential(*layers)

            # Flat feature map -> encoding vector.
            self.linear = nn.Linear(math.prod(self.end_shape), self.output_shape[0])

        def forward(self, value):
            batch_size, episode_length = value.shape[:2]
            value = value.reshape(batch_size * episode_length, *self.input_shape)
            value = self.model(value).reshape(batch_size * episode_length, -1)
            encoding = self.linear(value)
            return encoding.reshape(batch_size, episode_length, self.output_shape[0])

    example_encoder = ExampleEncoder(
        name = 'example',
        input_shape = (3, 32, 32),
        output_shape = (64,),
        arg_dict = {'hidden_channels' : [32, 64, 128]},
        verbose = True)
    print('\n\n')
    print(example_encoder)
    print()

    example_input, example_output = example_encoder.make_examples()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function('model_inference'):
            print(summary(
                example_encoder,
                input_data = example_input))
    #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))
            


######################



class Combinor(nn.Module):
    
    def __init__(
            self,
            name,
            list_of_models,
            verbose = False):
        
        super().__init__()
        
        self.name = name 
        
        name_counts = Counter(model.name for model in list_of_models)
        repeated_names = sorted(name for name, count in name_counts.items() if count > 1)
        if repeated_names:
            raise ValueError(
                f"""
These model names are used more than once: {repeated_names}
Every model in list_of_models needs its own name.
                """)
        
        self.list_of_output_shapes = []
        self.models_dict = nn.ModuleDict()
        for model in sorted(list_of_models, key=lambda model: model.name):
            self.models_dict[model.name] = model
            self.list_of_output_shapes.append(model.output_shape)
        leading_shape = self.list_of_output_shapes[0][:-1]
        if any(shape[:-1] != leading_shape for shape in self.list_of_output_shapes):
            raise ValueError(
                "All model output shapes must match except for their final dimension. "
                f"Received: {self.list_of_output_shapes}")
        
        self.total_output_shape = (
            *leading_shape,
            sum(shape[-1] for shape in self.list_of_output_shapes),)
            
        if(verbose):
            self.print_examples()
            
    def forward(self, value_dict):
        keys_only_in_models = self.models_dict.keys() - value_dict.keys()        
        keys_only_in_values = value_dict.keys() - self.models_dict.keys()
        if keys_only_in_models or keys_only_in_values:
            raise ValueError(
                f"""
These dictionaries aren't matched!
These keys are only in models_dict: \t{keys_only_in_models}
These keys are only in value_dict: \t{keys_only_in_values}
                """)
        outputs = [model(value_dict[name]) for name, model in self.models_dict.items()]
        return torch.cat(outputs, dim=-1)
    
    def make_examples(self, batch_size = 1, episode_length = 1):
        example_input_dict = {
            name : model.make_examples(batch_size, episode_length)[0]
            for name, model in self.models_dict.items()}
        return(
            example_input_dict,
            torch.zeros(batch_size, episode_length, *self.total_output_shape))
    
    def print_examples(self):
        example_input_dict, example_output = self.make_examples()
        example_inputs = ''.join(
            f"\n\t\t{name}: \t{list(example_input.shape)}"
            for name, example_input in example_input_dict.items())
        print(
            f"""
{len(self.models_dict)} models ({', '.join(self.models_dict.keys())}):
\texample inputs: {example_inputs}
\texample output: \t{list(example_output.shape)}
            """)
            
            
            
if __name__ == '__main__':
    
    
    
    print("\n\n\n\n\n\n\n\n\n\n")
    
    
    
    class ExampleBranchModel(Shape_to_Shape_Model):

        def build_model(self, arg_dict = {'hidden_size' : 32}):
            hidden_size = arg_dict['hidden_size']
            input_size = math.prod(self.input_shape)
            output_size = math.prod(self.output_shape)

            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, output_size))

        def forward(self, value):
            batch_size, episode_length = value.shape[:2]
            value = value.reshape(batch_size * episode_length, math.prod(self.input_shape))
            output = self.model(value)
            return output.reshape(batch_size, episode_length, *self.output_shape)

    image_encoder = ExampleBranchModel(
        name='image',
        input_shape=(3, 8, 8),
        output_shape=(4, 16),
        arg_dict = {'hidden_size' : 32})

    position_encoder = ExampleBranchModel(
        name='position',
        input_shape=(6,),
        output_shape=(4, 8),
        arg_dict = {'hidden_size' : 32})

    combinor = Combinor(
        name = 'example_combinor',
        list_of_models=[
            position_encoder,
            image_encoder],
        verbose=True)

    print('\n')
    print(combinor)
    print()

    example_input_dict, example_output = combinor.make_examples()

    print(summary(
        combinor,
        input_data=[example_input_dict],
        depth=4))
    
    

######################
            
    
    
class Divider(nn.Module):
    
    def __init__(
            self,
            name,
            list_of_models,
            verbose = False):
                
        super().__init__()
        
        self.name = name
        
        name_counts = Counter(model.name for model in list_of_models)
        repeated_names = sorted(name for name, count in name_counts.items() if count > 1)
        if repeated_names:
            raise ValueError(
                f"""These model names are used more than once: {repeated_names}
                Every model in list_of_models needs its own name.
                """)
        
        list_of_input_shapes = []
        self.models_dict = nn.ModuleDict()
        for model in sorted(list_of_models, key=lambda model: model.name):
            self.models_dict[model.name] = model
            list_of_input_shapes.append(model.input_shape)
            
        self.input_shape = list_of_input_shapes[0]
        if any(shape != self.input_shape for shape in list_of_input_shapes[1:]):
            raise ValueError(
                "Every model must have the same input_shape. "
                f"Received: {list_of_input_shapes}")
            
        if(verbose):
            self.print_examples()
            
    def forward(self, value):
        return {name: model(value) for name, model in self.models_dict.items()}
    
    def make_examples(self, batch_size=1, episode_length=1):
        example_input = torch.zeros(batch_size, episode_length, *self.input_shape)
        example_output_dict = {
            name: torch.zeros(batch_size, episode_length, *model.output_shape)
            for name, model in self.models_dict.items()}
        return example_input, example_output_dict
            
    def print_examples(self):
        example_input, example_output_dict = self.make_examples()
        example_outputs = ''.join(
            f"\n\t\t{name}: \t{list(example_output.shape)}"
            for name, example_output in example_output_dict.items())
        print(
            f"""
{len(self.models_dict)} models ({', '.join(self.models_dict.keys())}):
\texample input: {list(example_input.shape)}
\texample outputs: \t{example_outputs}
            """)
            


if __name__ == '__main__':
    
    
    
    print("\n\n\n\n\n\n\n\n\n\n")
    
    

    class ExampleOutputModel(Shape_to_Shape_Model):

        def build_model(self, arg_dict = {'hidden_size' : 32}):
            hidden_size = arg_dict['hidden_size']
            input_size = math.prod(self.input_shape)
            output_size = math.prod(self.output_shape)

            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, output_size))

        def forward(self, value):
            batch_size, episode_length = value.shape[:2]
            value = value.reshape(
                batch_size * episode_length,
                math.prod(self.input_shape))
            output = self.model(value)

            return output.reshape(batch_size, episode_length, *self.output_shape)


    position_model = ExampleOutputModel(
        name='position',
        input_shape=(64,),
        output_shape=(3,),
        arg_dict = {'hidden_size' : 32})

    image_model = ExampleOutputModel(
        name='image',
        input_shape=(64,),
        output_shape=(3, 8, 8),
        arg_dict = {'hidden_size' : 32})

    divider = Divider(
        name = 'example_divider',
        list_of_models=[
            position_model,
            image_model],
        verbose=True)

    print('\n')
    print(divider)
    print()
    
    example_input, example_output_dict = divider.make_examples()
    
    #print(summary(
    #    divider,
    #    input_data=example_input,
    #    depth=4))