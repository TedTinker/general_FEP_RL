import torch



class VariableBuffer:
    def __init__(self, capacity, max_steps, shape=(1,), before_and_after=False):
        self.shape = shape
        self.before_and_after = before_and_after
        self.data = torch.zeros(
            (capacity, max_steps + (1 if before_and_after else 0)) + shape,
            dtype=torch.float32,
        )

    def reset_episode(self, episode_ptr):
        self.data[episode_ptr] = 0.0

    def push(self, episode_ptr, time_ptr, value):
        value = torch.as_tensor(value, dtype=torch.float32)
        self.data[episode_ptr, time_ptr] = value

    def sample(self, indices):
        return self.data[indices]


class RecurrentReplayBuffer:
    def __init__(self, obs_dict, act_dict, capacity, max_steps):
        self.capacity = capacity
        self.max_episode_len = max_steps
        self.num_episodes = 0
        self.episode_ptr = 0
        self.time_ptr = 0

        # Observations
        self.obs_buffers = {}
        for key, value in obs_dict.items():
            self.obs_buffers[key] = VariableBuffer(capacity, max_steps, shape=value["encoder"].example_input.shape[2:], before_and_after=True)

        # Actions
        self.action_buffers = {}
        for key, value in act_dict.items():
            self.action_buffers[key] = VariableBuffer(capacity, max_steps, shape=value["decoder"].example_output.shape[2:])

        # Scalars
        self.reward = VariableBuffer(capacity, max_steps)
        self.done = VariableBuffer(capacity, max_steps)
        self.mask = VariableBuffer(capacity, max_steps)

    def reset_episode(self):
        for buf in [*self.obs_buffers.values(), *self.act_buffers.values(),
                    self.reward, self.done, self.mask]:
            buf.reset_episode(self.episode_ptr)

    def push(self, obs, action, reward, next_obs, done):
        if self.time_ptr == 0:
            self.reset_episode()

        for k, v in obs.items():
            self.obs_buffers[k].push(self.episode_ptr, self.time_ptr, v)

        for k, v in action.items():
            self.act_buffers[k].push(self.episode_ptr, self.time_ptr, v)

        self.reward.push(self.episode_ptr, self.time_ptr, reward)
        self.done.push(self.episode_ptr, self.time_ptr, done)
        self.mask.push(self.episode_ptr, self.time_ptr, 1.0)
        self.time_ptr += 1

        if done or self.time_ptr >= self.max_episode_len:
            for k, v in next_obs.items():
                self.obs_buffers[k].push(self.episode_ptr, self.time_ptr, v)

            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0
            self.num_episodes = min(self.num_episodes + 1, self.capacity)

    def sample(self, batch_size, random_sample=True):
        if self.num_episodes == 0:
            return False

        if random_sample:
            indices = torch.randint(0, self.num_episodes, (batch_size,))
        else:
            indices = torch.arange(batch_size)

        batch = {
            "obs": {k: buf.sample(indices) for k, buf in self.obs_buffers.items()},
            "action": {k: buf.sample(indices) for k, buf in self.act_buffers.items()},
            "reward": self.reward.sample(indices),
            "done": self.done.sample(indices),
            "mask": self.mask.sample(indices),
        }
        return batch
