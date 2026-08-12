#%%
#------------------
# shape_to_shape_models.py provides models convenient for world_models.
#------------------

import math
from collections import Counter

import torch
from torch import nn
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity



# Super-model for arbitrary shape-to-shape models.
class Shape_to_Shape_Model(nn.Module):

    def __init__(
            self,      
            name,               # String. Should be unique. 
            input_shape,        # Tuple like a torch shape, not including batch_size and episode_length. 
                                # Example: (64,).
                                # Example: (16, 4, 4). 
            output_shape,       # Another tuple like that.
            arg_dict = {},      # Anything sub-classes need for building or running. 
            verbose = False):   # Add print-outs.
    
        super().__init__()
        
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.arg_dict = arg_dict
        self.build_model(arg_dict)
        
        if verbose:
            self.print_examples()
        
    # Change these functions for sub-classes.
    def build_model(self, arg_dict):
        raise NotImplementedError("Subclasses must implement this method")
    
    def forward(self, value):
        raise NotImplementedError("Subclasses must implement this method")
        
    # Handy tool. Make example of input and output.
    def make_examples(self, batch_size = 1, episode_length = 1):
        return(
            torch.zeros(batch_size, episode_length, *self.input_shape),
            torch.zeros(batch_size, episode_length, *self.output_shape))
    
    def print_examples(self):
        example_input, example_output = self.make_examples()
        print(
            f"{self.name} Shape_to_Shape_Model:",
            f"\texample input: \t\t{list(example_input.shape)}",
            f"\texample output: \t{list(example_output.shape)}")

    
    
# Example.
if __name__ == '__main__':
    
    
    
    print("\n\n\n\n\n\n\n\n\n\n")
    
    
    
    class Example_Model(Shape_to_Shape_Model):
        def build_model(self, arg_dict):

            in_channels, in_height, in_width = self.input_shape

            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels, 
                    out_channels = 16, 
                    kernel_size = 3,
                    padding = 1))
            
            self.linear = nn.Linear(
                in_features = 16 * in_height * in_width, 
                out_features = self.output_shape[0])

        def forward(self, value):
            batch_size, episode_length = value.shape[:2]
            value = value.reshape(batch_size * episode_length, *self.input_shape)
            value = self.model(value).reshape(batch_size * episode_length, -1)
            encoding = self.linear(value)
            return encoding.reshape(batch_size, episode_length, self.output_shape[0])

    example_model = Example_Model(
        name = 'example',
        input_shape = (3, 32, 32),
        output_shape = (64,),
        verbose = True)
    
    print('\n\n')
    print(example_model)
    print()

    example_input, example_output = example_model.make_examples()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function('model_inference'):
            print(summary(
                example_model,
                input_data = example_input))
    #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))
            


######################



# Super-model for arbitrary list-of-shapes-to-one-shape models.
class Combiner(nn.Module):
    
    def __init__(
            self,
            name,               # String. Should be unique.
            list_of_models,     # List of shape_to_shape_models, with output_shapes matching except final dimension.
            verbose = False):
        
        super().__init__()
        
        self.name = name 
        
        # TEST: Are shape_to_shape_model names unique?
        name_counts = Counter(model.name for model in list_of_models)
        repeated_names = sorted(name for name, count in name_counts.items() if count > 1)
        if repeated_names:
            raise ValueError(
                f"These model names are used more than once: {repeated_names}",
                "Every model in list_of_models needs its own name.")
        
        # Make list of models while tracking output_shapes.
        self.list_of_output_shapes = []
        self.models_dict = nn.ModuleDict()
        for model in sorted(list_of_models, key=lambda model: model.name):
            self.list_of_output_shapes.append(model.output_shape)
            self.models_dict[model.name] = model
        
        # TEST: Can all output_shapes be concatenated along the last dimension?
        leading_shape = self.list_of_output_shapes[0][:-1]
        if any(shape[:-1] != leading_shape for shape in self.list_of_output_shapes):
            raise ValueError(
                "All model output shapes must match except for their final dimension.",
                f"Received: {self.list_of_output_shapes}")
        
        # Find size of output.
        self.total_output_shape = (
            *leading_shape,
            sum(shape[-1] for shape in self.list_of_output_shapes),)
            
        if(verbose):
            self.print_examples()
            
    def forward(self, value_dict):
        # TEST: Do names of models match names of values, and vice-versa?
        keys_only_in_models = self.models_dict.keys() - value_dict.keys()        
        keys_only_in_values = value_dict.keys() - self.models_dict.keys()
        if keys_only_in_models or keys_only_in_values:
            raise ValueError(
                "These dictionaries aren't matched!",
                f"These keys are only in models_dict: \t{keys_only_in_models}",
                f"These keys are only in value_dict: \t{keys_only_in_values}")
        # Use all models, combine outputs.
        outputs = [model(value_dict[name]) for name, model in self.models_dict.items()]
        return torch.cat(outputs, dim=-1)
    
    # Handy tool. Make example of inputs and output.
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
            f"{len(self.models_dict)} models ({', '.join(self.models_dict.keys())}):",
            f"\texample inputs: {example_inputs}",
            f"\texample output: \t{list(example_output.shape)}")
            
            
        
# Example.
if __name__ == '__main__':
    
    
    
    print("\n\n\n\n\n\n\n\n\n\n")
    
    
    
    # First, make shape_to_shape_models with concatable outputs. 
    class Example_Branch_Model(Shape_to_Shape_Model):

        def build_model(self, arg_dict = {'hidden_size' : 64}):
            input_size = math.prod(self.input_shape)
            output_size = math.prod(self.output_shape)

            self.model = nn.Sequential(
                nn.Linear(input_size, arg_dict['hidden_size']),
                nn.LeakyReLU(),
                nn.Linear(arg_dict['hidden_size'], output_size))

        def forward(self, value):
            batch_size, episode_length = value.shape[:2]
            value = value.reshape(batch_size * episode_length, math.prod(self.input_shape))
            output = self.model(value)
            return output.reshape(batch_size, episode_length, *self.output_shape)

    image_encoder = Example_Branch_Model(
        name='image',
        input_shape=(3, 8, 8),
        output_shape=(4, 16),
        arg_dict = {'hidden_size' : 64})

    position_encoder = Example_Branch_Model(
        name='position',
        input_shape=(6,),
        output_shape=(4, 8),
        arg_dict = {'hidden_size' : 32})

    # Make a Combiner with list of shape_to_shape_models.
    Combiner = Combiner(
        name = 'example_combinor',
        list_of_models=[
            position_encoder,
            image_encoder],
        verbose=True)

    print('\n')
    print(Combiner)
    print()

    example_input_dict, example_output = Combiner.make_examples()

    print(summary(
        Combiner,
        input_data=[example_input_dict],
        depth=4))
    
    

######################
            
    
    
# Super-model for arbitrary one-shape-to-list-of-shapes models.
class Divider(nn.Module):
    
    def __init__(
            self,
            name,               # String. Should be unique.
            list_of_models,     # List of shape_to_shape_models, with matching input_shapes.
            verbose = False):
                
        super().__init__()
        
        self.name = name
        
        # TEST: Are shape_to_shape_model names unique?
        name_counts = Counter(model.name for model in list_of_models)
        repeated_names = sorted(name for name, count in name_counts.items() if count > 1)
        if repeated_names:
            raise ValueError(
                f"These model names are used more than once: {repeated_names}",
                "Every model in list_of_models needs its own name.")
        
        # Make list of models while tracking input_shapes.
        list_of_input_shapes = []
        self.models_dict = nn.ModuleDict()
        for model in sorted(list_of_models, key=lambda model: model.name):
            list_of_input_shapes.append(model.input_shape)
            self.models_dict[model.name] = model
            
        # TEST: Are all input_shapes the same?
        self.input_shape = list_of_input_shapes[0]
        if any(shape != self.input_shape for shape in list_of_input_shapes[1:]):
            raise ValueError(
                "Every model must have the same input_shape. "
                f"Received: {list_of_input_shapes}")
            
        if(verbose):
            self.print_examples()
            
    def forward(self, value):
        # Use all models with the same input.
        return {name: model(value) for name, model in self.models_dict.items()}
    
    # Handy tool. Make example of input and outputs.
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
            f"{len(self.models_dict)} models ({', '.join(self.models_dict.keys())}):",
            f"\texample input: {list(example_input.shape)}",
            f"\texample outputs: \t{example_outputs}")
            


# Example.
if __name__ == '__main__':
    
    
    
    print("\n\n\n\n\n\n\n\n\n\n")
    
    
    
    # First, make shape_to_shape_models the same input_shape. 
    class Example_Output_Model(Shape_to_Shape_Model):

        def build_model(self, arg_dict = {'hidden_size' : 32}):
            input_size = math.prod(self.input_shape)
            output_size = math.prod(self.output_shape)

            self.model = nn.Sequential(
                nn.Linear(input_size, arg_dict["hidden_size"]),
                nn.LeakyReLU(),
                nn.Linear(arg_dict["hidden_size"], output_size))

        def forward(self, value):
            batch_size, episode_length = value.shape[:2]
            value = value.reshape(
                batch_size * episode_length,
                math.prod(self.input_shape))
            output = self.model(value)

            return output.reshape(batch_size, episode_length, *self.output_shape)


    position_model = Example_Output_Model(
        name='position',
        input_shape=(64,),
        output_shape=(3,),
        arg_dict = {'hidden_size' : 32})

    image_model = Example_Output_Model(
        name='image',
        input_shape=(64,),
        output_shape=(3, 8, 8),
        arg_dict = {'hidden_size' : 32})

    # Make a divider with list of shape_to_shape_models.
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
    
    print(summary(
        divider,
        input_data=example_input,
        depth=4))