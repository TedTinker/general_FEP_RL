#%% 
#------------------
# world_model.py provides an architecture for creating predictions of future observations
# based on multi-layer mtrnn. Actor and Critic utilize its hidden states.  
#------------------

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary

from general_FEP_RL.utils_torch import init_weights, parametrize_normal, sample, calculate_dkl, generate_dummy_inputs
from general_FEP_RL.mtrnn import MTRNN



#------------------
# A module for every prior p estimated posterior q inner state. 
# \mu^p_t, \sigma^p_t = MLP_i(h^q_{t-1} || action_{t-1})
# \mu^q_t, \sigma^q_t = MLP_i(h^q_{t-1} || action_{t-1} || o^{encoded}_t)

# q(z_t) = \mathcal{N}(\mu^q_t,\sigma^q_t))
# p(z_t) = \mathcal{N}(\mu^p_t,\sigma^p_t))

# DKL[q(z_t) || p(z_t)] is used to measure complexity and curiosity.
#------------------

class ZP_ZQ(nn.Module):
    
    def __init__(
            self, 
            zp_in_features,     # Size of MTRNN hidden state plus encoded actions.
            zq_in_features,     # Size of MTRNN hidden state plus encoded actions and encoded observations.
            zp_zq_sizes,        # Inner state sizes.
            verbose = False):
        super(ZP_ZQ, self).__init__()
                
        self.zp_zq_sizes = zp_zq_sizes 
        self.example_zp_start = torch.zeros((32, 16, zp_in_features))
        self.example_zq_start = torch.zeros((32, 16, zq_in_features))
        
        if verbose:
            print(f'\nZP_ZQ start: \n \t{self.example_zp_start.shape}, {self.example_zq_start.shape}')
            
        # An inner state may have multiple layers. 
        def build_network(
                in_features, 
                layer_sizes, 
                final_activation):
            layers = []
            current_in_features = in_features
            num_layers = len(layer_sizes)
            for i, out_size in enumerate(layer_sizes):
                layers.append(nn.Linear(current_in_features, out_size))
                if i < num_layers - 1:
                    layers.append(nn.PReLU())
                else:
                    if final_activation == 'tanh':
                        layers.append(nn.Tanh())
                    elif final_activation == 'softplus':
                        layers.append(nn.Softplus())
                    else:
                        raise ValueError('Invalid final_activation specified.')
                current_in_features = out_size
            return nn.Sequential(*layers)
            
        self.zp_mu  = build_network(zp_in_features, zp_zq_sizes, 'tanh')
        self.zp_std = build_network(zp_in_features, zp_zq_sizes, 'softplus')
        self.zq_mu  = build_network(zq_in_features, zp_zq_sizes, 'tanh')
        self.zq_std = build_network(zq_in_features, zp_zq_sizes, 'softplus')
        
        example_zp_mu, example_zp_std = parametrize_normal(self.example_zp_start, self.zp_mu, self.zp_std)
        example_zq_mu, example_zq_std = parametrize_normal(self.example_zq_start, self.zq_mu, self.zq_std)
        
        if verbose:
            print(f'ZP_ZQ end: \n \tZP Mu/Std shape: {example_zp_mu.shape} / {example_zp_std.shape}, \n \tZQ Mu/Std shape: {example_zq_mu.shape} / {example_zq_std.shape}, \n')

        self.apply(init_weights)
        
        
        
    def forward(
            self, 
            zp_inputs, 
            zq_inputs):                                    
        zp_mu, zp_std = parametrize_normal(zp_inputs, self.zp_mu, self.zp_std)
        zq_mu, zq_std = parametrize_normal(zq_inputs, self.zq_mu, self.zq_std)

        if(use_sample):
            zp = sample(zp_mu, zp_std)
            zq = sample(zq_mu, zq_std)
        else:
            zp = zp_mu 
            zq = zq_mu
        dkl = calculate_dkl(zq_mu, zq_std, zp_mu, zp_std)
        return {
            'zp' : zp, 
            'zq' : zq, 
            'dkl' : dkl}      
   
    

#------------------
# Example. 
#------------------
    
if __name__ == '__main__':
    zp_zq = ZP_ZQ(
        zp_in_features = 16, 
        zq_in_features = 32, 
        zp_zq_sizes = [128, 128], 
        verbose = True)
    print('\n\n')
    print(zp_zq)
    print()
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function('model_inference'):
            print(summary(
                zp_zq, 
                input_data=(zp_zq.example_zp_start, zp_zq.example_zq_start)))
    #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))
    
    
#%%



#------------------
# This makes q, p, and dkl for each module of the sensation or higher layers of longer-term memory.
#------------------

class World_Model_Layer(nn.Module):
    
    def __init__(
            self, 
            hidden_state_size,
            observation_model_dict,             # The lowest layer sees observations 
            action_model_dict,                  # and actions.
            bottom_layer,           
            top_layer,          
            layer_number = 0,
            lower_zp_zq_size = 0,               # Higher layers see lower priors or estimaters posteriors. 
            higher_hidden_state_size = 0,       # Layers below the top see higher hidden states.
            time_scale = 1,                     # Higher time-scalars mean slower memory. 
            verbose = False):
        super(World_Model_Layer, self).__init__()
        
        self.bottom_layer = bottom_layer 
        self.top_layer = top_layer
        self.layer_number = layer_number

        self.zp_zq_dict = nn.ModuleDict()
        if bottom_layer:                                                        # For bottom layer:
            total_action_size = sum(                                            # Consider action encoding size.
                action_model_dict[key]['encoder'].arg_dict['encode_size'] 
                for key in sorted(action_model_dict.keys()))
            for key, model in sorted(observation_model_dict.items()):           # Consider observation encoding size and inner state sizes
                zp_zq_size = model['encoder'].arg_dict['zp_zq_sizes']
                obs_size = model['encoder'].arg_dict['encode_size']

                self.zp_zq_dict[key] = ZP_ZQ(                                   # Make inner state models for actions and observations.
                    zp_in_features = hidden_state_size + total_action_size, 
                    zq_in_features = hidden_state_size + total_action_size + obs_size, 
                    zp_zq_sizes = zp_zq_size)
        else:                                                                   # For higher layers:
            self.zp_zq_dict['zq'] = ZP_ZQ(                                      # Make inner state models for lower inner states.
                zp_in_features = hidden_state_size, 
                zq_in_features = hidden_state_size + lower_zp_zq_size, 
                zp_zq_sizes = [hidden_state_size])
    
        self.mtrnn = MTRNN(
                input_size = sum(
                    zp_zq.zp_zq_sizes[-1] 
                    for key, zp_zq in sorted(self.zp_zq_dict.items())) + higher_hidden_state_size,
                hidden_size = hidden_state_size, 
                time_constant = time_scale)
            
        self.apply(init_weights)
        
        
        
    # Information traveling from lowest layer to highest layer. 
    def bottom_up(
            self,
            prev_hidden_state, 
            encoded_obs = None, 
            encoded_prev_action = None,
            lower_zp_zq = None,
            higher_hidden_state = None,
            use_sample = True):
        
        episodes, steps = prev_hidden_state.shape[0], prev_hidden_state.shape[1]
        
        def process_z_func_outputs(zp_inputs, zq_inputs, z_func):
            zp_inputs = zp_inputs.reshape(episodes * steps, zp_inputs.shape[2])
            zq_inputs = zq_inputs.reshape(episodes * steps, zq_inputs.shape[2])
            inner_states = z_func(zp_inputs, zq_inputs, use_sample = use_sample)
            return inner_states
                
                
        # WHEN I USE ITEMS, AM I STOPPING BACKPROP?
                
        if self.bottom_layer:
            zp_inputs = torch.cat([prev_hidden_state] + [v for k, v in sorted(encoded_prev_action.items())], dim=-1)
            zq_inputs_dict = {key: torch.cat([zp_inputs, obs_part], dim=-1) for key, obs_part in sorted(encoded_obs.items())}
        else:
            zp_inputs = prev_hidden_state
            zq_inputs_dict = {'zq': torch.cat([zp_inputs, lower_zp_zq], dim=-1)}
                
        inner_state_dict = {
            key: process_z_func_outputs(zp_inputs, zq_inputs_dict[key], self.zp_zq_dict[key])
            for key in self.zp_zq_dict.keys()}
                
        if self.top_layer:
            mtrnn_inputs_p = torch.cat([inner_state['zp'] for _, inner_state in sorted(inner_state_dict.items())], dim=-1)
            mtrnn_inputs_q = torch.cat([inner_state['zq'] for _, inner_state in sorted(inner_state_dict.items())], dim=-1)
        else:
            higher_hidden_state = higher_hidden_state.reshape(episodes * steps, higher_hidden_state.shape[2])
            mtrnn_inputs_p = torch.cat([inner_state['zp'] for _, inner_state in sorted(inner_state_dict.items())] + [higher_hidden_state], dim=-1)
            mtrnn_inputs_q = torch.cat([inner_state['zq'] for _, inner_state in sorted(inner_state_dict.items())] + [higher_hidden_state], dim=-1)
        
        mtrnn_inputs_p = mtrnn_inputs_p.reshape(episodes, steps, mtrnn_inputs_p.shape[1])
        mtrnn_inputs_q = mtrnn_inputs_q.reshape(episodes, steps, mtrnn_inputs_q.shape[1])
        
        return mtrnn_inputs_p, mtrnn_inputs_q, inner_state_dict
    
    
    
    # Information traveling from highest layer to lowest layer. 
    def top_down(
            self,
            inputs_p,
            inputs_q,
            prev_hidden_state):
        
        new_hidden_state_p = self.mtrnn(inputs_p, prev_hidden_state)
        new_hidden_state_q = self.mtrnn(inputs_q, prev_hidden_state)
        return new_hidden_state_p, new_hidden_state_q
        
        
    
    def forward(
            self, 
            prev_hidden_state, 
            encoded_obs = None, 
            encoded_prev_action = None,
            lower_zp_zq = None,
            higher_hidden_state = None,
            use_sample = True):
        mtrnn_inputs_p, mtrnn_inputs_q, inner_state_dict = self.bottom_up(
            prev_hidden_state,
            encoded_obs,
            encoded_prev_action,
            lower_zp_zq,
            higher_hidden_state,
            use_sample)        
        new_hidden_state_p, new_hidden_state_q = self.top_down(
            mtrnn_inputs_p, 
            mtrnn_inputs_q, 
            prev_hidden_state)
        return new_hidden_state_p, new_hidden_state_q, inner_state_dict
        
        
    
#------------------
# Example. 
#------------------

if __name__ == '__main__':
    
    from general_FEP_RL.encoders.encode_image import Encode_Image
    from general_FEP_RL.decoders.decode_image import Decode_Image
    
    hidden_state_size = 128
    
    observation_model_dict = {
        'see_image' : { 
            'encoder' : Encode_Image(
                arg_dict = {
                    'encode_size' : 128,
                    'zp_zq_sizes' : [128, 128]}, 
                verbose = True),
            'decoder' : Decode_Image(
                hidden_state_size, 
                verbose = True),
            'target_entropy' : 1,
            'accuracy_scalar' : 1,                               
            'complexity_scalar' : 1,                                 
            'eta' : 1                                   
            },
        'see_image_2' : { 
            'encoder' : Encode_Image(
                arg_dict = {
                    'encode_size' : 64,
                    'zp_zq_sizes' : [64, 64]}, 
                verbose = True),
            'decoder' : Decode_Image(
                hidden_state_size, 
                verbose = True),
            'target_entropy' : 1,
            'accuracy_scalar' : 1,                               
            'complexity_scalar' : 1,                                 
            'eta' : 1                                   
            }
        }
    
    action_model_dict = {
        'make_image' : {
            'encoder' : Encode_Image(verbose = True),
            'decoder' : Decode_Image(hidden_state_size, entropy = True, verbose = True),
            'target_entropy' : 1,
            'accuracy_scalar' : 1,                               
            'complexity_scalar' : 1,                                 
            'eta' : 1                                   
            }
        }
    
    wl = World_Model_Layer(
        hidden_state_size = hidden_state_size,
        observation_model_dict = observation_model_dict, 
        action_model_dict = action_model_dict, 
        bottom_layer = True,
        top_layer = True,
        layer_number = 0,
        time_scale = 1, 
        verbose = True)
    print('\n\n')
    print(wl)
    print()
    
    dummies = generate_dummy_inputs(
        observation_model_dict,
        action_model_dict, 
        hidden_state_size)
    dummy_inputs = dummies['hidden'], dummies['obs_enc_out'], dummies['act_enc_out'], 0
        
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function('model_inference'):
            print(summary(wl, input_data=(dummy_inputs)))
    #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))
    

    
#%%

        

#------------------
# Combining layers to make a model like a PVRNN.
#------------------

class World_Model(nn.Module):
    
    def __init__(
            self, 
            hidden_state_sizes,             # Sizes of the agent's short-term/long-term memories.
            observation_dict,               # Observations the agent observes.
            action_dict,                    # Actions the agent performs.
            time_scales,                    # Time-scales of the short-term/long-tem memories.
            verbose = False):
        super(World_Model, self).__init__()
                
        self.use_sample = True
        self.hidden_state_sizes = hidden_state_sizes 
        self.example_input = torch.zeros((32, 16, hidden_state_sizes[0]))
        
        # World model encodes actions. 
        self.action_model_dict = nn.ModuleDict()
        for key, model in sorted(action_dict.items()):
            self.action_model_dict[key] = nn.ModuleDict()
            self.action_model_dict[key]['encoder'] = model['encoder'](
                arg_dict = model['encoder_arg_dict'], verbose = verbose)
        
        encoded_action_size = 0 
        for key, value in sorted(self.action_model_dict.items()):
            encoded_action_size += value['encoder'].arg_dict['encode_size']
        
        # World model encodes observations and predicts future observations. 
        self.observation_model_dict = nn.ModuleDict()
        for key, model in sorted(observation_dict.items()):
            self.observation_model_dict[key] = nn.ModuleDict()
            self.observation_model_dict[key]['encoder'] = model['encoder'](
                arg_dict = model['encoder_arg_dict'], verbose = verbose)
            self.observation_model_dict[key]['decoder'] = model['decoder'](
                hidden_state_sizes[0] + encoded_action_size, arg_dict = model['decoder_arg_dict'], verbose = verbose)
               
        # Multiple layers of MTRNN produce short-term and long-term memory.
        # On the bottom layer, with an observation of n parts,
        # h^q_t = RNN(h^q_{t-1}, z^q_{t,0} || ... || z^q_{t,n}).
        self.world_layers = nn.ModuleList()
        first_layer_zp_zq_size = sum(
            self.observation_model_dict[key]['encoder'].arg_dict['zp_zq_sizes'][-1] 
            for key in sorted(self.observation_model_dict.keys()))
        for i, time_scale in enumerate(time_scales):
            self.world_layers.append(
                World_Model_Layer(
                    hidden_state_size = hidden_state_sizes[i],      
                    observation_model_dict = self.observation_model_dict if i == 0 else None,   
                    action_model_dict = self.action_model_dict if i == 0 else None,            
                    bottom_layer = i == 0,
                    top_layer = i + 1 == len(time_scales), 
                    layer_number = i,
                    lower_zp_zq_size = 0 if i == 0 else first_layer_zp_zq_size if i == 1 else hidden_state_sizes[i-1],
                    higher_hidden_state_size = 0 if i+1 == len(time_scales) else hidden_state_sizes[i+1],
                    time_scale = time_scale, 
                    verbose = verbose))

        self.apply(init_weights)
                
                
            
    # Encode incoming actions.
    def action_in(self, action):
        encoded_action = {}
        for key, value in sorted(action.items()):
            encoded_action[key] = self.action_model_dict[key]['encoder'](value)
        return encoded_action
            
            
            
    # Encode incoming observations.
    def obs_in(self, obs):
        encoded_obs = {}
        for key, value in sorted(obs.items()):
            encoded_obs[key] = self.observation_model_dict[key]['encoder'](value)
        return encoded_obs
    
    
    
    # Predict upcoming observations.
    def predict(self, hidden_state, encoded_action):
        hidden_state_and_action = torch.cat([hidden_state] + [v for k, v in sorted(encoded_action.items())], dim=-1)
        predicted_obs = {}
        for key, value in sorted(self.observation_model_dict.items()):
            prediction, _ = self.observation_model_dict[key]['decoder'](hidden_state_and_action)
            predicted_obs[key] = prediction
        return predicted_obs
    
    
    
    # Transfer information from bottom layer to top layer, then top layer to bottom layer.
    def bottom_to_top_step(self, prev_hidden_states, encoded_obs, encoded_prev_action):

        inner_state_dict_list = []
        mtrnn_inputs_p_list = []
        mtrnn_inputs_q_list = []
        first_layer_zp_zq = []
        for i, world_layer in enumerate(self.world_layers):
            mtrnn_inputs_p, mtrnn_inputs_q, inner_state_dict = world_layer.bottom_up(
                prev_hidden_state = prev_hidden_states[i], 
                encoded_obs = encoded_obs if i==0 else None, 
                encoded_prev_action = encoded_prev_action if i==0 else None,
                lower_zp_zq = None if i==0 else first_layer_zp_zq if i==1 else inner_state_dict_list[-1][i-1]['zq'].unsqueeze(1),
                higher_hidden_state = None if i+1 == len(self.world_layers) else prev_hidden_states[i+1],   
                use_sample = self.use_sample)
            if i==0:
                first_layer_zp_zq = torch.cat([value['zq'] for key, value in sorted(inner_state_dict.items())], dim = -1).unsqueeze(1)
            inner_state_dict_list.append(inner_state_dict)
            mtrnn_inputs_p_list.append(mtrnn_inputs_p)
            mtrnn_inputs_q_list.append(mtrnn_inputs_q)
        
        new_hidden_states_p = []
        new_hidden_states_q = []
        for i in reversed(range(len(self.world_layers))):
            world_layer = self.world_layers[i]
            new_hidden_state_p, new_hidden_state_q = world_layer.top_down(
                mtrnn_inputs_p_list[i],
                mtrnn_inputs_q_list[i],
                prev_hidden_states[i])
            new_hidden_states_p.append(new_hidden_state_p)
            new_hidden_states_q.append(new_hidden_state_q)
            
        new_hidden_states_p.reverse() 
        new_hidden_states_q.reverse()
            
        inner_state_dict = {}
        for d in inner_state_dict_list:
            inner_state_dict.update(d)
            
        return(
            new_hidden_states_p, 
            new_hidden_states_q, 
            inner_state_dict)
    
    
    
    # Update hidden states and predict next observations.
    def forward(self, prev_hidden_states, obs, prev_action, one_step = False):
                           
        first = next(iter(obs.values()))
        episodes, steps = first.shape[0], first.shape[1]
                                    
        if prev_hidden_states is None:
            prev_hidden_states = [
                torch.zeros((episodes, 1, hidden_state_size))
                for hidden_state_size in self.hidden_state_sizes]
                        
        encoded_obs = self.obs_in(obs)
        encoded_prev_action = self.action_in(prev_action)
        
        hidden_states_p_list = [prev_hidden_states]
        hidden_states_q_list = [prev_hidden_states]
        inner_state_dicts_list = []
                                    
        for step in range(steps):
                                    
            step_obs = {}
            for key, value in sorted(encoded_obs.items()):
                step_obs[key] = value[:,step].unsqueeze(1)
                
            step_prev_action = {}
            for key, value in sorted(encoded_prev_action.items()):
                step_prev_action[key] = value[:,step].unsqueeze(1)
                                        
            new_hidden_states_p, new_hidden_states_q, inner_state_dict = \
                self.bottom_to_top_step(prev_hidden_states, step_obs, step_prev_action)
                
            prev_hidden_states = new_hidden_states_q
            hidden_states_p_list.append(new_hidden_states_p)
            hidden_states_q_list.append(new_hidden_states_q)
            inner_state_dicts_list.append(inner_state_dict)
                                       
        hidden_states_p = [] 
        hidden_states_q = [] 
        for i in range(len(hidden_states_p_list[0])):
            hidden_states_p.append(torch.cat([h[i] for h in hidden_states_p_list], dim = 1))
            hidden_states_q.append(torch.cat([h[i] for h in hidden_states_q_list], dim = 1))
                        
        catted_inner_state_dicts = {}
        for key in inner_state_dicts_list[0].keys():
            zp  = torch.stack([d[key]['zp']  for d in inner_state_dicts_list], dim=1)
            zq  = torch.stack([d[key]['zq']  for d in inner_state_dicts_list], dim=1)
            dkl = torch.stack([d[key]['dkl'] for d in inner_state_dicts_list], dim=1)
            catted_inner_state_dicts[key] = {'zp': zp, 'zq': zq, 'dkl': dkl}

        if one_step:
            # If working with only one step, we cannot predict the next
            # observation, because we don't have the next action.
            return [h[:,1:] for h in hidden_states_p], [h[:,1:] for h in hidden_states_q], catted_inner_state_dicts
        else:
            # If working with a whole episode, predict future observations. 
            skip_non_action = {}
            for key, value in sorted(encoded_prev_action.items()):    
                skip_non_action[key] = value[:, 1:]
            
            pred_obs_p = self.predict(hidden_states_p[0][:, 1:-1], skip_non_action)
            pred_obs_q = self.predict(hidden_states_q[0][:, 1:-1], skip_non_action)
        
            return(
                hidden_states_p, 
                hidden_states_q, 
                catted_inner_state_dicts, 
                pred_obs_p, 
                pred_obs_q)
        
        
        
    # This function prints the complete architecture of the world model.
    def summary(self):
                
        """# Layers can only be summarized if there is just one layer.
        if len(self.world_layers) == 1:
            dummies = generate_dummy_inputs(self.observation_model_dict, self.action_model_dict, self.hidden_state_sizes)        
    
            for i, world_layer in enumerate(self.world_layers):
                print(f'\n\nWORLD_MODEL_LAYER {i}')
                if i == 0:
                    dummy_inputs = dummies['hidden'][0], dummies['obs_enc_out'], dummies['act_enc_out'], 0
                else:
                    dummy_inputs = dummies['hidden'][i], None, None
                    
                with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                    with record_function('model_inference'):
                        print(summary(
                            self.world_layers[i], 
                            input_data=(dummy_inputs)))
                #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))"""
        
        
        print('\n\nOBSERVATIONS')
        for key, value in sorted(self.observation_model_dict.items()):
            print(f'\n{key} ENCODER')
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function('model_inference'):
                    print(summary(
                        self.observation_model_dict[key]['encoder'], 
                        input_data=(self.observation_model_dict[key]['encoder'].example_input)))
            #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))
                    
            print(f'\n{key} DECODER')
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function('model_inference'):
                    print(summary(
                        self.observation_model_dict[key]['decoder'], 
                        input_data=(self.observation_model_dict[key]['decoder'].example_input)))
            #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))
            
        print('\n\nACTIONS')
        for key, value in sorted(self.action_model_dict.items()):
            print(f'\n{key} ENCODER')
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function('model_inference'):
                    print(summary(
                        self.action_model_dict[key]['encoder'], 
                        input_data=(self.action_model_dict[key]['encoder'].example_input)))
            #print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=100))
        
        
        
if __name__ == '__main__':
    
    observation_dict = {
        'see_image' : {
            'encoder' : Encode_Image,
            'encoder_arg_dict' : {
                'encode_size' : 256,
                'zp_zq_sizes' : [256]},
            'decoder' : Decode_Image,
            'decoder_arg_dict' : {},
            },
        'see_image_2' : {
            'encoder' : Encode_Image,
            'encoder_arg_dict' : {
                'encode_size' : 16,
                'zp_zq_sizes' : [16]},
            'decoder' : Decode_Image,
            'decoder_arg_dict' : {},
            }
        }
    
    action_dict = {
        'make_image' : {
            'encoder' : Encode_Image,
            'encoder_arg_dict' : {
                'encode_size' : 128},
            'decoder' : Decode_Image,
            'decoder_arg_dict' : {},
            }
        }
    
    hidden_state_sizes = [128]
    wm = World_Model(            
        hidden_state_sizes = hidden_state_sizes,
        observation_dict = observation_dict, 
        action_dict = action_dict,
        time_scales = [1],
        verbose = True)
    print('\n\n')
    print(wm)
    print()
    
    

    dummies = generate_dummy_inputs(
        wm.observation_model_dict, 
        wm.action_model_dict, 
        hidden_state_size)
    dummy_inputs = [dummies['hidden'], dummies['hidden']], dummies['obs_enc_in'], dummies['act_enc_in'], 0
    
    wm.summary()

# %%