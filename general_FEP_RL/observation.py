class Observation:
    
    def __init__(
        self, 
        name, 
        encoder, 
        encoder_arg_dict ,
        decoder, 
        decoder_arg_dict = {},
        upsilon_obs = 1,
        beta_obs = 1,
        eta_before_clamp = 1,
        eta = 1
        ),
    
    self.name = name
    self.encoder = encoder
    self.encoder_arg_dict = encoder_arg_dict
    self.decoder = decoder
    self.decoder_arg_dict = decoder_arg_dict
    self.upsilon_obs = upsilon_obs 
    self.beta_obs = beta_obs
    self.eta_before_clamp = 1
    self.eta = eta 
    
    
    
    


            
            action_dict,            # Keys: action_names
                                    # Values: 
                                        # encoder
                                            # example_input
                                            # example_output
                                        # encoder_arg_dict
                                            # encode_size
                                        # decoder
                                            # example_input
                                            # example_output
                                            # loss_func
                                        # decoder_arg_dict
                                        # target_entropy
                                        # alpha_normal
                                        # lr_alpha
                                        # initial_alpha
                                        # delta (imitation scalar)