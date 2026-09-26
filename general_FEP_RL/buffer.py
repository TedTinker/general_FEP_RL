#%%
#------------------
# buffer.py provides a recurrent replay buffer.
#
# Episodes are stored whole, in slots. When the buffer is full, which slot is evicted depends on the eviction policy:
# oldest-first by default, or the caller's decision. We hope to eventually eject episodes based on how little curiosity they carry.
#------------------

import random
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from general_FEP_RL.shape_to_shape_models import Shape_to_Shape_Model
from general_FEP_RL.world_model import make_world_model



#------------------
# Buffer for a single variable.
# Each episode has an extra, final element for observations, because
# the accuracy term is based on predictions of the NEXT observation.
#------------------
class Variable_Buffer:

    def __init__(
            self,
            capacity,                   # How many episodes can be saved?
            max_steps,                  # How long can an episode be?
            shape = (1,),               # Shape of element.
            observation = False):       # Is this element an observation?

        self.shape = tuple(shape)
        self.observation = observation
        self.data = torch.zeros(
            (capacity, max_steps + (1 if observation else 0)) + self.shape,
            dtype = torch.float32)

    def reset_episode(self, slot):
        self.data[slot] = 0.0

    def push(self, slot, time_ptr, value):
        value = torch.as_tensor(value, dtype = torch.float32)
        # Values have shape (batch, episode_length, ...). Reshape.
        if value.shape != self.shape:
            if value.numel() != int(torch.tensor(self.shape).prod()):
                raise ValueError(
                    f"Expected a value of shape {self.shape} or something reshapeable "
                    f"to it, received {tuple(value.shape)}.")
            value = value.reshape(self.shape)
        self.data[slot, time_ptr] = value

    def sample(self, slots):
        return self.data[slots]



#------------------
# Recurrent replay buffer.
#------------------

class Recurrent_Replay_Buffer:

    def __init__(
            self,
            dict_of_observation_shapes,     # name : shape tuple, not including batch or time.
            dict_of_action_shapes,          # name : shape tuple, not including batch or time.
            capacity,                       # How many episodes?
            max_steps,                      # How long can an episode be?
            eviction_policy = None):        # If at capacity, how to choose which episode to evict? Default: first-in, first-out.

        self.capacity = capacity
        self.max_episode_len = max_steps
        self.eviction_policy = eviction_policy

        # Observations.
        self.observation_buffers = {
            name : Variable_Buffer(capacity, max_steps, shape = shape, observation = True)
            for name, shape in dict_of_observation_shapes.items()}

        # Actions.
        self.action_buffers = {
            name : Variable_Buffer(capacity, max_steps, shape = shape)
            for name, shape in dict_of_action_shapes.items()}

        # Best actions, for an imitation term.
        self.best_action_buffers = {
            name : Variable_Buffer(capacity, max_steps, shape = shape)
            for name, shape in dict_of_action_shapes.items()}

        # Scalars.
        self.reward = Variable_Buffer(capacity, max_steps)
        self.done = Variable_Buffer(capacity, max_steps)
        self.mask = Variable_Buffer(capacity, max_steps)                 # To ignore dummy padding, it is masked.
        self.best_action_mask = Variable_Buffer(capacity, max_steps)     # If "best actions" are provided, a mask reveals them.

        # Slot bookkeeping.
        #
        # An episode_id is a unique an unchanging handle, for consistency in choosing episodes.
        # A slot is a row of the tensors, and slots ids ARE changed and reused.
        self.set_of_free_slots = set(range(capacity))
        self.list_of_episode_ids_by_age = []         # Committed episodes, oldest first.
        self.dict_of_slots_by_episode_id = {}
        self.next_episode_id = 0

        # The current episode cannot be evicted mid-write.
        self.current_episode_id = None
        self.current_slot = None
        self.time_ptr = 0



    #------------------
    # Slots and eviction.
    #------------------

    @property
    def episode_ids(self):
        return list(self.list_of_episode_ids_by_age)

    def __len__(self):
        return len(self.list_of_episode_ids_by_age)

    def choose_episode_to_evict(self):
        if self.eviction_policy is not None:                # If the user provided another episode-deletion function, use it.
            episode_id = self.eviction_policy(self)
            if episode_id not in self.dict_of_slots_by_episode_id:
                raise ValueError(
                    f"The eviction policy chose episode {episode_id}, which this buffer does not hold."
                    f"Episodes currently held: {self.list_of_episode_ids_by_age}")
            return episode_id
        return self.list_of_episode_ids_by_age[0]           # Otherwise, the default is first-in, first-out.

    def evict(self, episode_id):
        if episode_id == self.current_episode_id:
            raise ValueError(
                f"Episode {episode_id} is still being written and cannot be evicted.")
        if episode_id not in self.dict_of_slots_by_episode_id:
            raise ValueError(
                f"This buffer does not hold episode {episode_id}."
                f"Episodes currently held: {self.list_of_episode_ids_by_age}")
        slot = self.dict_of_slots_by_episode_id.pop(episode_id)
        self.list_of_episode_ids_by_age.remove(episode_id)
        self.set_of_free_slots.add(slot)
        self.reset_slot(slot)                               # So stale data cannot leak.
        return slot

    def acquire_slot(self):
        if not self.set_of_free_slots:
            self.evict(self.choose_episode_to_evict())
        return self.set_of_free_slots.pop()

    def reset_slot(self, slot):
        for variable_buffer in self.all_variable_buffers():
            variable_buffer.reset_episode(slot)

    def all_variable_buffers(self):
        return [
            *self.observation_buffers.values(),
            *self.action_buffers.values(),
            *self.best_action_buffers.values(),
            self.reward, self.done, self.mask, self.best_action_mask]



    # Writing inputs.
    def push(
            self,
            observation_dict,
            action_dict,
            reward,
            next_observation_dict,
            done,
            best_action_dict = None):

        if self.time_ptr == 0:
            self.current_slot = self.acquire_slot()
            self.current_episode_id = self.next_episode_id
            self.next_episode_id += 1
            self.reset_slot(self.current_slot)

        observation_dict = {k : v.detach().cpu() for k, v in observation_dict.items()}
        action_dict = {k : v.detach().cpu() for k, v in action_dict.items()}
        next_observation_dict = {k : v.detach().cpu() for k, v in next_observation_dict.items()}

        for k, v in observation_dict.items():
            self.observation_buffers[k].push(self.current_slot, self.time_ptr, v)

        for k, v in action_dict.items():
            self.action_buffers[k].push(self.current_slot, self.time_ptr, v)

        if best_action_dict is None:
            for k, v in action_dict.items():
                self.best_action_buffers[k].push(self.current_slot, self.time_ptr, torch.zeros_like(v))
            self.best_action_mask.push(self.current_slot, self.time_ptr, 0)
        else:
            best_action_dict = {k : v.detach().cpu() for k, v in best_action_dict.items()}
            for k, v in best_action_dict.items():
                self.best_action_buffers[k].push(self.current_slot, self.time_ptr, v)
            self.best_action_mask.push(self.current_slot, self.time_ptr, 1)

        self.reward.push(self.current_slot, self.time_ptr, reward)
        self.done.push(self.current_slot, self.time_ptr, done)
        self.mask.push(self.current_slot, self.time_ptr, 1.0)
        self.time_ptr += 1

        if done or self.time_ptr >= self.max_episode_len:
            for k, v in next_observation_dict.items():
                self.observation_buffers[k].push(self.current_slot, self.time_ptr, v)
            self.commit_episode()

    def commit_episode(self):
        self.dict_of_slots_by_episode_id[self.current_episode_id] = self.current_slot
        self.list_of_episode_ids_by_age.append(self.current_episode_id)
        self.current_episode_id = None
        self.current_slot = None
        self.time_ptr = 0



    # Fetching a batch.
    def sample(self, batch_size, random_sample = True, device = None):
        if not self.list_of_episode_ids_by_age:
            return None

        # Sample the unique episode_ids, and look their slots up.
        count = min(batch_size, len(self.list_of_episode_ids_by_age))
        if random_sample:
            positions = torch.randperm(len(self.list_of_episode_ids_by_age))[:count]
        else:
            positions = torch.arange(count)
        episode_ids = [self.list_of_episode_ids_by_age[position] for position in positions]
        slots = torch.tensor([self.dict_of_slots_by_episode_id[i] for i in episode_ids])

        def get(variable_buffer):
            value = variable_buffer.sample(slots)
            return value if device is None else value.to(device)

        return {
            'episode_ids' : episode_ids,       
            'obs' : {k : get(b) for k, b in self.observation_buffers.items()},
            'action' : {k : get(b) for k, b in self.action_buffers.items()},
            'best_action' : {k : get(b) for k, b in self.best_action_buffers.items()},
            'reward' : get(self.reward),
            'done' : get(self.done),
            'mask' : get(self.mask),
            'best_action_mask' : get(self.best_action_mask)}



# Find the shapes of a world_model.
# The prior inputs (EXCEPT the hidden state) are labeled actions.
# The posterior inputs (EXCEPT those shared with the prior) are labeled observations.
def shapes_from_world_model(world_model):

    generated = {'previous_hidden_state', 'lower_layer_posterior_sample'}
    dict_of_action_shapes = {}
    dict_of_observation_shapes = {}

    def add(dict_of_shapes, name, shape, kind):
        if name in dict_of_shapes and dict_of_shapes[name] != shape:
            raise ValueError(
                f"'{name}' appears as a {kind} with shape {dict_of_shapes[name]} on one "
                f"layer and {shape} on another. One name means one variable.")
        dict_of_shapes[name] = shape

    for world_model_layer in world_model.list_of_world_model_layers:
        for name, model in world_model_layer.prior_input_encoder.models_dict.items():
            if name not in generated:
                add(dict_of_action_shapes, name, tuple(model.input_shape), 'action')

    for world_model_layer in world_model.list_of_world_model_layers:
        for name, model in world_model_layer.posterior_input_encoder.models_dict.items():
            if name not in generated and name not in dict_of_action_shapes:
                add(dict_of_observation_shapes, name, tuple(model.input_shape), 'observation')

    return dict_of_observation_shapes, dict_of_action_shapes