#%%
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary

from general_FEP_RL.utils_torch import init_weights, var, sample, calculate_dkl, generate_dummy_inputs
from general_FEP_RL.mtrnn import MTRNN



# A module for every prior/estimated posterior inner state.
class ZP_ZQ(nn.Module):
    
    def __init__(
            self, 
            zp_in_features,     # Size of RNN hidden state plus encoded actions
            zq_in_features,     # Size of RNN hidden state plus encoded actions and encoded observations
            zp_zq_sizes,        # State size
            verbose = False):
        super(ZP_ZQ, self).__init__()
                
        self.zp_zq_sizes = zp_zq_sizes 
        self.example_zp_start = torch.zeros((32, 16, zp_in_features))
        self.example_zq_start = torch.zeros((32, 16, zq_in_features))
        
        if(verbose):
            print(f"\nZP_ZQ start: \n \t{self.example_zp_start.shape}, {self.example_zq_start.shape}")
            
        # Prior: Previous hidden state and action.  
        self.zp_mu = nn.Sequential(
                nn.Linear(
                    in_features = zp_in_features, 
                    out_features = zp_in_features),
                nn.PReLU(),
                nn.Linear(
                    in_features = zp_in_features, 
                    out_features = zp_zq_sizes),
                nn.Tanh())
        
        self.zp_std = nn.Sequential(
                nn.Linear(
                    in_features = zp_in_features, 
                    out_features = zp_in_features),
                nn.PReLU(),
                nn.Linear(
                    in_features = zp_in_features, 
                    out_features = zp_zq_sizes),
                nn.Softplus())
        
        # Posterior: Previous hidden state and action and observation.  
        self.zq_mu = nn.Sequential(
                nn.Linear(
                    in_features = zq_in_features, 
                    out_features = zq_in_features),
                nn.PReLU(),
                nn.Linear(
                    in_features = zq_in_features, 
                    out_features = zp_zq_sizes),
                nn.Tanh())
        
        self.zq_std = nn.Sequential(
                nn.Linear(
                    in_features = zq_in_features, 
                    out_features = zq_in_features),
                nn.PReLU(),
                nn.Linear(
                    in_features = zq_in_features, 
                    out_features = zp_zq_sizes),
                nn.Softplus())
        
        example_zp_mu, example_zp_std = var(self.example_zp_start, self.zp_mu, self.zp_std)
        example_zq_mu, example_zq_std = var(self.example_zq_start, self.zq_mu, self.zq_std)
        
        if(verbose):
            print(f"ZP_ZQ end: \n \t{example_zp_mu.shape}, {example_zp_std.shape}, \n \t{example_zq_mu.shape}, {example_zq_std.shape}, \n")

        self.apply(init_weights)
        
    def forward(self, zp_inputs, zq_inputs):                                    
        zp_mu, zp_std = var(zp_inputs, self.zp_mu, self.zp_std)
        zp = sample(zp_mu, zp_std)
        zq_mu, zq_std = var(zq_inputs, self.zq_mu, self.zq_std)
        zq = sample(zq_mu, zq_std)
        dkl = calculate_dkl(zp_mu, zp_std, zq_mu, zq_std)
        return({
            "zp" : zp, 
            "zq" : zq, 
            "dkl" : dkl})      
   
    

if(__name__ == "__main__"):
    zp_zq = ZP_ZQ(16, 32, 128, verbose = True)
    print("\n\n")
    print(zp_zq)
    print()
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(
                zp_zq, 
                input_data=(zp_zq.example_zp_start, zp_zq.example_zq_start)))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
#%%



# Making prior and posterior inner states for each portion of the sensation.
class World_Model_Layer(nn.Module):
    
    def __init__(
            self, 
            hidden_state_size,
            observation_dict, 
            action_dict, 
            time_scale = 1, 
            verbose = False):
        super(World_Model_Layer, self).__init__()
                
        total_action_size = sum(action_dict[key]["encoder"].args_dict["encode_size"] for key in action_dict.keys())

        self.zp_zq_dict = nn.ModuleDict()
        for key in observation_dict.keys():
            self.zp_zq_dict[key] = ZP_ZQ(
                zp_in_features = hidden_state_size + total_action_size, 
                zq_in_features = hidden_state_size + total_action_size + observation_dict[key]["encoder"].args_dict["encode_size"], 
                out_features = observation_dict[key]["encoder"].out_features)
    
        self.mtrnn = MTRNN(
                input_size = sum(zp_zq.out_features for zp_zq in self.zp_zq_dict.values()),
                hidden_size = hidden_state_size, 
                time_constant = time_scale)
            
        self.apply(init_weights)
        
        
            
    def forward(self, prev_hidden_state, encoded_obs, encoded_prev_action):
        
        def reshape(inputs, episodes, steps):
            inputs = inputs.reshape(episodes * steps, inputs.shape[2])
            return inputs
        
        def process_z_func_outputs(zp_inputs, zq_inputs, z_func, episodes, steps):
            zp_inputs = reshape(zp_inputs, episodes, steps)
            zq_inputs = reshape(zq_inputs, episodes, steps)
            inner_states = z_func(zp_inputs, zq_inputs)
            return(inner_states)
                
        print(prev_hidden_state.shape, [v.shape for v in encoded_prev_action.values()])
        zp_inputs = torch.cat([prev_hidden_state] + [v for v in encoded_prev_action.values()], dim=-1)
        zq_inputs_dict = {key : torch.cat([zp_inputs, obs_part], dim=-1) for key, obs_part in encoded_obs.items()}              
        episodes, steps = zp_inputs.shape[0], zp_inputs.shape[1]

        inner_state_dict = {key : process_z_func_outputs(zp_inputs, zq_inputs, z_func, episodes, steps) for (key, zq_inputs), z_func in zip(zq_inputs_dict.items(), self.zp_zq_dict.values())}
                
        mtrnn_inputs_p = torch.cat([inner_state["zp"] for inner_state in inner_state_dict.values()], dim = -1)
        mtrnn_inputs_p = mtrnn_inputs_p.reshape(episodes, steps, mtrnn_inputs_p.shape[1])

        mtrnn_inputs_q = torch.cat([inner_state["zq"] for inner_state in inner_state_dict.values()], dim = -1)
        mtrnn_inputs_q = mtrnn_inputs_q.reshape(episodes, steps, mtrnn_inputs_q.shape[1])
        
        new_hidden_state_p = self.mtrnn(mtrnn_inputs_p, prev_hidden_state)
        new_hidden_state_q = self.mtrnn(mtrnn_inputs_q, prev_hidden_state)
        
        return(new_hidden_state_p, new_hidden_state_q, inner_state_dict)
        
        
    
if __name__ == "__main__":
    
    from general_FEP_RL.encoders.encode_image import Encode_Image
    from general_FEP_RL.decoders.decode_image import Decode_Image
    
    hidden_state_size = 128
    
    observation_dict = {
        "see_image" : { 
            "encoder" : Encode_Image(verbose = True),
            "decoder" : Decode_Image(hidden_state_size, verbose = True),
            "target_entropy" : 1,
            "accuracy_scaler" : 1,                               
            "complexity_scaler" : 1,                                 
            "eta" : 1                                   
            }
        }
    
    action_dict = {
        "make_image" : {
            "encoder" : Encode_Image(verbose = True),
            "decoder" : Decode_Image(hidden_state_size, entropy = True, verbose = True),
            "target_entropy" : 1,
            "accuracy_scaler" : 1,                               
            "complexity_scaler" : 1,                                 
            "eta" : 1                                   
            }
        }
    
    wl = World_Model_Layer(
        hidden_state_size = hidden_state_size,
        observation_dict = observation_dict, 
        action_dict = action_dict, 
        time_scale = 1, 
        verbose = True)
    print("\n\n")
    print(wl)
    print()
    
    dummies = generate_dummy_inputs(observation_dict, action_dict, hidden_state_size)
    dummy_inputs = dummies["hidden"], dummies["obs_enc_out"], dummies["act_enc_out"]
        
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(wl, input_data=(dummy_inputs)))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
    
    
    
    
#%%

        

# Combining the above modules to make a model in the style of a PVRNN.
class World_Model(nn.Module):
    
    def __init__(
            self, 
            hidden_state_size,
            observation_dict, 
            action_dict,
            verbose = False):
        super(World_Model, self).__init__()
                
        self.hidden_state_size = hidden_state_size 
        self.example_input = torch.zeros((32, 16, hidden_state_size))
        
        self.action_dict = nn.ModuleDict()
        for key in action_dict.keys():
            self.action_dict[key] = nn.ModuleDict()
            self.action_dict[key]["encoder"] = action_dict[key]["encoder"](arg_dict = action_dict[key]["encoder_arg_dict"], verbose = verbose)
            self.action_dict[key]["decoder"] = action_dict[key]["decoder"](hidden_state_size, entropy = True, arg_dict = action_dict[key]["decoder_arg_dict"],  verbose = verbose)
        
        encoded_action_size = 0 
        for key, value in self.action_dict.items():
            encoded_action_size += self.action_dict[key]["encoder"].out_features
        
        self.observation_dict = nn.ModuleDict()
        for key in observation_dict.keys():
            self.observation_dict[key] = nn.ModuleDict()
            self.observation_dict[key]["encoder"] = observation_dict[key]["encoder"](arg_dict = observation_dict[key]["encoder_arg_dict"], verbose = verbose)
            self.observation_dict[key]["decoder"] = observation_dict[key]["decoder"](hidden_state_size + encoded_action_size, arg_dict = observation_dict[key]["decoder_arg_dict"], verbose = verbose)
               
        self.wl = World_Model_Layer(
            hidden_state_size = hidden_state_size,
            observation_dict = self.observation_dict, 
            action_dict = self.action_dict,
            time_scale = 1, 
            verbose = False)

        self.apply(init_weights)
                
                
            
    # Encode incoming observations. # PROBLEM HERE!
    def obs_in(self, obs):
        encoded_obs = {}
        for key, value in obs.items():
            encoded_obs[key] = self.observation_dict[key]["encoder"](value)
        return(encoded_obs)
    
    
    
    # Encode incoming actions.
    def action_in(self, action):
        encoded_action = {}
        for key, value in action.items():
            encoded_action[key] = self.action_dict[key]["encoder"](value)
        return(encoded_action)
    
    
    
    # Predict upcoming observations.
    def predict(self, hidden_state, encoded_action):
        hidden_states_and_action = torch.cat([hidden_state] + [v for v in encoded_action.values()], dim=-1)
        predicted_obs = {}
        for key, value in self.observation_dict.items():
            prediction, log_prob = self.observation_dict[key]["decoder"](hidden_states_and_action)
            predicted_obs[key] = prediction
        return(predicted_obs)
    
    
    
    # This was originally made to utilize multiple layers, which is not currently implemented.
    def bottom_to_top_step(self, prev_hidden_states, obs, prev_action):
        new_hidden_states_p, new_hidden_states_q, inner_state_dict = \
            self.wl(prev_hidden_states, obs, prev_action)       
        return(new_hidden_states_p, new_hidden_states_q, inner_state_dict)
    
    
    
    def forward(self, prev_hidden_state, obs, prev_action, one_step = False):
                   
        for key, value in obs.items():    
            episodes, steps = value.shape[0], value.shape[1]
                                    
        if(prev_hidden_state == None):
            prev_hidden_state = torch.zeros(episodes, 1, self.hidden_state_size)
            
        encoded_obs = self.obs_in(obs)
        encoded_prev_action = self.action_in(prev_action)
        
        hidden_state_p_list = [prev_hidden_state]
        hidden_state_q_list = [prev_hidden_state]
        inner_state_dict_list = []
                                    
        for step in range(steps):
                                    
            step_obs = {}
            for key, value in encoded_obs.items():
                step_obs[key] = value[:,step].unsqueeze(1)
                
            step_prev_action = {}
            for key, value in encoded_prev_action.items():
                step_prev_action[key] = value[:,step].unsqueeze(1)
                                        
            new_hidden_state_p, new_hidden_state_q, inner_state_dict = \
                self.bottom_to_top_step(prev_hidden_state, step_obs, step_prev_action)
                
            prev_hidden_state = new_hidden_state_q
            hidden_state_p_list.append(new_hidden_state_p)
            hidden_state_q_list.append(new_hidden_state_q)
            inner_state_dict_list.append(inner_state_dict)
                                            
        hidden_state_p = torch.cat(hidden_state_p_list, dim = 1)
        hidden_state_q = torch.cat(hidden_state_q_list, dim = 1)
                        
        catted_inner_state_dict = {}
        for key, inner_state_dict in inner_state_dict_list[0].items():
            zp = torch.stack([inner_state_dict[key]["zp"] for inner_state_dict in inner_state_dict_list], dim = 1)
            zq = torch.stack([inner_state_dict[key]["zq"] for inner_state_dict in inner_state_dict_list], dim = 1)
            dkl = torch.stack([inner_state_dict[key]["dkl"] for inner_state_dict in inner_state_dict_list], dim = 1)
            catted_inner_state_dict[key] = {"zp" : zp, "zq" : zq, "dkl" : dkl}

            
        if(one_step):
            # Cannot make prediction, because we need the next action.
            return(hidden_state_p[:, 1:], hidden_state_q[:, 1:], catted_inner_state_dict)
        else:
            # Make predictions for all steps.
            skip_non_action = {}
            for key, value in encoded_prev_action.items():    
                skip_non_action[key] = value[:, 1:]
            
            pred_obs_p = self.predict(hidden_state_p[:, 1:-1], skip_non_action)
            pred_obs_q = self.predict(hidden_state_q[:, 1:-1], skip_non_action)
        
            return(hidden_state_p[:, 1:], hidden_state_q[:, 1:], catted_inner_state_dict, pred_obs_p, pred_obs_q)
        
        
        
    def summary(self):
                
        print("\nWORLD_MODEL_LAYER")
        dummies = generate_dummy_inputs(self.observation_dict, self.action_dict, self.hidden_state_size)
        dummy_inputs = dummies["hidden"], dummies["obs_enc_out"], dummies["act_enc_out"]
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                print(summary(
                    self.wl, 
                    input_data=(dummy_inputs)))
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        
        print("\n\nOBSERVATIONS")
        for key, value in self.observation_dict.items():
            print(f"\n{key} ENCODER")
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    print(summary(
                        self.observation_dict[key]["encoder"], 
                        input_data=(self.observation_dict[key]["encoder"].example_input)))
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
                    
            print(f"\n{key} DECODER")
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    print(summary(
                        self.observation_dict[key]["decoder"], 
                        input_data=(self.observation_dict[key]["decoder"].example_input)))
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
            
        print("\n\nACTIONS")
        for key, value in self.action_dict.items():
            print(f"\n{key} ENCODER")
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    print(summary(
                        self.action_dict[key]["encoder"], 
                        input_data=(self.action_dict[key]["encoder"].example_input)))
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
            
            print(f"\n{key} DECODER")
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    print(summary(
                        self.action_dict[key]["decoder"], 
                        input_data=(self.action_dict[key]["decoder"].example_input)))
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        
        
        
if __name__ == "__main__":
    
    observation_dict = {
        "see_image" : {
            "encoder" : Encode_Image,
            "decoder" : Decode_Image,
            "encoder_arg_dict" : {},
            "decoder_arg_dict" : {},
            }
        }
    
    action_dict = {
        "make_image" : {
            "encoder" : Encode_Image,
            "decoder" : Decode_Image,
            "encoder_arg_dict" : {},
            "decoder_arg_dict" : {},
            }
        }
    
    fm = World_Model(            
        hidden_state_size = hidden_state_size,
        observation_dict = observation_dict, 
        action_dict = action_dict,
        verbose = True)
    print("\n\n")
    print(fm)
    print()
    
    

    dummies = generate_dummy_inputs(fm.observation_dict, fm.action_dict, hidden_state_size)
    dummies["hidden"] = dummies["hidden"][:,0].unsqueeze(1)
    dummy_inputs = dummies["hidden"], dummies["obs_enc_in"], dummies["act_enc_in"]
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(fm, input_data=dummy_inputs))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

            

# %%