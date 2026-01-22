#%%
#------------------
# buffer.py provides a recurrent replay buffer.
#------------------

import torch



#------------------
# Buffer for a single variable.
# Observations have a final element at the end of the episode. 
#------------------

class VariableBuffer:
    def __init__(
            self, 
            capacity, 
            max_steps, 
            shape=(1,), 
            observation=False):
        self.shape = shape
        self.observation = observation
        self.data = torch.zeros((capacity, max_steps + (1 if observation else 0)) + shape, dtype=torch.float32)

    def reset_episode(self, episode_ptr):
        self.data[episode_ptr] = 0.0

    def push(self, episode_ptr, time_ptr, value):
        value = torch.as_tensor(value, dtype=torch.float32)
        self.data[episode_ptr, time_ptr] = value

    def sample(self, indices):
        return self.data[indices]



#------------------
# Recurrent replay buffer.
#------------------

class RecurrentReplayBuffer:
    def __init__(
            self, 
            observation_model_dict,     # The models carry the shapes of example tensors.
            action_model_dict, 
            capacity, 
            max_steps):
        self.capacity = capacity
        self.max_episode_len = max_steps
        self.num_episodes = 0
        self.episode_ptr = 0
        self.time_ptr = 0

        # Observations
        self.observation_buffers = {}
        for key, model in observation_model_dict.items():
            self.observation_buffers[key] = VariableBuffer(
                capacity, 
                max_steps, 
                shape = model['encoder'].example_input.shape[2:],
                observation = True)

        # Actions
        self.action_buffers = {}
        for key, model in action_model_dict.items():
            self.action_buffers[key] = VariableBuffer(
                capacity, 
                max_steps, 
                shape = model['encoder'].example_input.shape[2:])
            
        # Best Actions
        self.best_action_buffers = {}
        for key, model in action_model_dict.items():
            self.best_action_buffers[key] = VariableBuffer(
                capacity, 
                max_steps, 
                shape=model['encoder'].example_input.shape[2:])

        # Scalars
        self.reward = VariableBuffer(capacity, max_steps)
        self.done = VariableBuffer(capacity, max_steps)
        self.mask = VariableBuffer(capacity, max_steps)
        self.best_action_mask = VariableBuffer(capacity, max_steps)



    def reset_episode(self):
        for buf in [*self.observation_buffers.values(), *self.action_buffers.values(), *self.best_action_buffers.values(),
                    self.reward, self.done, self.mask, self.best_action_mask]:
            buf.reset_episode(self.episode_ptr)



    def push(
            self, 
            observation_dict, 
            action_dict, 
            reward, 
            next_observation_dict, 
            done, 
            best_action_dict = None):
        if self.time_ptr == 0:
            self.reset_episode()
            
        observation_dict = {k: v.detach().cpu() for k, v in observation_dict.items()}
        action_dict = {k: v.detach().cpu() for k, v in action_dict.items()}
        next_observation_dict = {k: v.detach().cpu() for k, v in next_observation_dict.items()}

        for k, v in observation_dict.items():
            self.observation_buffers[k].push(self.episode_ptr, self.time_ptr, v)

        for k, v in action_dict.items():
            self.action_buffers[k].push(self.episode_ptr, self.time_ptr, v)
            
        if best_action_dict is None:
            for k, v in action_dict.items():
                self.best_action_buffers[k].push(self.episode_ptr, self.time_ptr, torch.zeros_like(v))
            self.best_action_mask.push(self.episode_ptr, self.time_ptr, 0)
        else:
            best_action_dict = {k: v.detach().cpu() for k, v in best_action_dict.items()}
            for k, v in best_action_dict.items():
                self.best_action_buffers[k].push(self.episode_ptr, self.time_ptr, v)
            self.best_action_mask.push(self.episode_ptr, self.time_ptr, 1)

        self.reward.push(self.episode_ptr, self.time_ptr, reward)
        self.done.push(self.episode_ptr, self.time_ptr, done)
        self.mask.push(self.episode_ptr, self.time_ptr, 1.0)
        self.time_ptr += 1

        if done or self.time_ptr >= self.max_episode_len:
            for k, v in next_observation_dict.items():
                self.observation_buffers[k].push(self.episode_ptr, self.time_ptr, v)

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0
            self.num_episodes = min(self.num_episodes + 1, self.capacity)
            
            

    def sample(self, batch_size, random_sample=True):
        if self.num_episodes == 0:
            return None

        if random_sample:
            indices = torch.randint(0, self.num_episodes, (batch_size,))
        else:
            indices = torch.arange(min(batch_size, self.num_episodes))

        batch = {
            'obs': {k: buf.sample(indices) for k, buf in self.observation_buffers.items()},
            'action': {k: buf.sample(indices) for k, buf in self.action_buffers.items()},
            'best_action' : {k: buf.sample(indices) for k, buf in self.best_action_buffers.items()},
            'reward': self.reward.sample(indices),
            'done': self.done.sample(indices),
            'mask': self.mask.sample(indices),
            'best_action_mask': self.best_action_mask.sample(indices)}
        return batch
