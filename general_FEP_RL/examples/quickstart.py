#%%
#------------------
# quickstart.py is a complete, runnable example of general_FEP_RL.
#
# The agent controls a dot in a small 2-D world learns to move to a goal and stay there. 
# The agent sees an 8x8 image and feels the walls. 
# It has a two-layer world model, and trains with Soft Actor Critic on extrinsic reward plus curiosity.
#
# Run from the repository root:
#     pip install -r requirements.txt
#     python examples/quickstart.py
#
# Figures are saved to examples/output/.
#------------------

import math
import random
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

# Lets this run from a fresh clone, without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from general_FEP_RL.agent import Agent
from general_FEP_RL.actor_critic import Action_Decoder
from general_FEP_RL.shape_to_shape_models import Shape_to_Shape_Model
from general_FEP_RL.plot_training_log import plot_training_log
from general_FEP_RL.utils import set_seed

OUTPUT_DIR = Path(__file__).resolve().parent / 'output'



#------------------
# Settings.
#------------------

SEED = 0
EPISODES = 300
EPOCHS_PER_EPISODE = 3
BATCH_SIZE = 32             
MAX_STEPS = 20              
EVALUATE_EVERY = 25
EVALUATION_EPISODES = 20

# The agent must find a goal in the image and stay there. 
# Add more goals, like (-0.9, 0.9), to make it read that goal from the image too. 
GOALS = [(0.9, 0.9)]

IMAGE_SHAPE = (2, 8, 8)     # (channels, height, width). Channel 0 is the dot, channel 1 is the goal.
TOUCH_SIZE = 4
ACTION_SIZE = 2



#------------------
# Encoders and decoders. Everything is a Shape_to_Shape_Model.
# Tensors have shape (batch, steps, ...).
#------------------

class Vector_Encoder(Shape_to_Shape_Model):

    def __init__(self, name, input_size, output_size, hidden_size = 32, verbose = False):
        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            arg_dict = {'hidden_size' : hidden_size},
            verbose = verbose)

    def build_model(self, arg_dict):
        self.model = nn.Sequential(
            nn.Linear(self.input_shape[0], arg_dict['hidden_size']),
            nn.LeakyReLU(),
            nn.Linear(arg_dict['hidden_size'], self.output_shape[0]),
            nn.LeakyReLU())

    def forward(self, value):
        return self.model(value)
    
    
    
class Vector_Decoder(Shape_to_Shape_Model):

    def __init__(self, name, input_size, output_size, hidden_size = 32, verbose = False):
        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = (output_size,),
            arg_dict = {'hidden_size' : hidden_size},
            verbose = verbose)

    def build_model(self, arg_dict):
        self.model = nn.Sequential(
            nn.Linear(self.input_shape[0], arg_dict['hidden_size']),
            nn.LeakyReLU(),
            nn.Linear(arg_dict['hidden_size'], self.output_shape[0]))

    def forward(self, value):
        return self.model(value)

    @staticmethod
    def loss_func(predicted_values, target_values):
        return F.mse_loss(predicted_values, target_values, reduction = 'none')



class Image_Encoder(Shape_to_Shape_Model):

    def __init__(self, name, input_shape, output_size, hidden_channels = (16, 32), verbose = False):
        super().__init__(
            name = name,
            input_shape = input_shape,
            output_shape = (output_size,),
            arg_dict = {'hidden_channels' : list(hidden_channels)},
            verbose = verbose)

    def build_model(self, arg_dict):
        hidden_channels = arg_dict['hidden_channels']
        in_channels, in_height, in_width = self.input_shape
        channels = [in_channels] + hidden_channels

        layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1))
            layers.append(nn.LeakyReLU())
        self.model = nn.Sequential(*layers)

        end_shape = (
            hidden_channels[-1],
            in_height // 2 ** len(hidden_channels),
            in_width // 2 ** len(hidden_channels))
        self.linear = nn.Linear(math.prod(end_shape), self.output_shape[0])

    def forward(self, value):
        batch_size, episode_length = value.shape[:2]
        value = value.reshape(batch_size * episode_length, *self.input_shape)
        value = self.model(value).reshape(batch_size * episode_length, -1)
        encoding = self.linear(value)
        return encoding.reshape(batch_size, episode_length, self.output_shape[0])



class Image_Decoder(Shape_to_Shape_Model):

    def __init__(self, name, input_size, output_shape, hidden_size = 64, verbose = False):
        super().__init__(
            name = name,
            input_shape = (input_size,),
            output_shape = output_shape,
            arg_dict = {'hidden_size' : hidden_size},
            verbose = verbose)

    def build_model(self, arg_dict):
        self.model = nn.Sequential(
            nn.Linear(self.input_shape[0], arg_dict['hidden_size']),
            nn.LeakyReLU(),
            nn.Linear(arg_dict['hidden_size'], math.prod(self.output_shape)))

    def forward(self, value):
        batch_size, episode_length = value.shape[:2]
        output = self.model(value)
        return output.reshape(batch_size, episode_length, *self.output_shape)

    @staticmethod
    def loss_func(predicted_values, target_values):
        return F.mse_loss(predicted_values, target_values, reduction = 'none')



#------------------
# A simple example environment. The dot in the square [-1, 1]^2 should move to the goal and stay there.
#
#   Observations:   'vision' (2, 8, 8)  a blob for the dot and a blob for the goal.
#                   'touch'  (4,)       1 while pressed against the left, right, bottom, or top wall.
#   Action:         'move'   (2,)       velocity in [-1, 1].
#   Reward:         negative of the distance to the goal.
#------------------

class Dot_World:

    def __init__(self, goals = GOALS, image_size = IMAGE_SHAPE[-1], step_size = 0.15, goal_radius = 0.3, blob_width = 0.3):
        self.goals = goals
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.blob_width = blob_width
        coordinates = torch.linspace(-1, 1, image_size)
        self.grid_y, self.grid_x = torch.meshgrid(coordinates, coordinates, indexing = 'ij')

    def blob(self, point):
        squared_distance = (self.grid_x - point[0]) ** 2 + (self.grid_y - point[1]) ** 2
        return torch.exp(-squared_distance / (2 * self.blob_width ** 2))

    def observe(self):
        vision = torch.stack([self.blob(self.position), self.blob(self.goal)])
        touch = torch.tensor([
            float(self.position[0] <= -1), float(self.position[0] >= 1),
            float(self.position[1] <= -1), float(self.position[1] >= 1)])
        # The agent expects (batch, steps, ...), which here is (1, 1, ...).
        return {'vision' : vision[None, None], 'touch' : touch[None, None]}

    def reset(self):
        self.position = torch.rand(2) * 2 - 1
        self.goal = torch.tensor(random.choice(self.goals))
        return self.observe()

    def distance(self):
        return torch.dist(self.position, self.goal).item()

    def step(self, action):
        self.position = (self.position + self.step_size * action.flatten()).clamp(-1, 1)
        reward = -self.distance()
        done = False                # See the PITFALL above.
        return self.observe(), reward, done



#------------------
# The agent.
#
# Each per-layer argument is a list with one entry per layer, bottom layer first.
#------------------

def make_agent():
    return Agent(
        hidden_state_sizes = [32, 16],
        time_constants = [1, 4],            # Layer 1 updates at most 1/4 of its hidden state per step.

        # Prior inputs: what the agent knows BEFORE seeing the next observation. Just its hidden states and its action.
        list_of_dict_of_prior_input_encoder_class_dicts = [
            {'move' : {'class' : partial(Vector_Encoder, name = 'move', input_size = ACTION_SIZE, output_size = 16)}},
            {}],

        # Posterior inputs: the prior inputs, and, also, observations. decoding_output_size is the width of that modality's inner state.
        list_of_dict_of_posterior_input_encoder_class_dicts = [
            {'vision' : {
                'class' : partial(Image_Encoder, name = 'vision', input_shape = IMAGE_SHAPE, output_size = 32),
                'decoding_output_size' : 16},
             'touch' : {
                'class' : partial(Vector_Encoder, name = 'touch', input_size = TOUCH_SIZE, output_size = 16),
                'decoding_output_size' : 4}},
            {}],                            # Layer 1 has no observations of its own, so it only summarises layer 0.
                                            # To give the layer observations, add them here and in its prediction decoders.

        # One prediction decoder per posterior input, with the same names.
        list_of_dict_of_prediction_decoder_class_dicts = [
            {'vision' : {'class' : partial(Image_Decoder, name = 'vision', output_shape = IMAGE_SHAPE)},
             'touch' : {'class' : partial(Vector_Decoder, name = 'touch', output_size = TOUCH_SIZE)}},
            {}],

        # Width of each layer's inner state for the layer below's posterior sample. Entry 0, regarding the first layer, is ignored.
        lower_layer_posterior_sample_decoding_output_sizes = [0, 8],

        # The actor decodes actions from layer 0's hidden state. The critics encode those actions.
        dict_of_action_decoder_class_dicts = {
            'move' : {'class' : partial(Action_Decoder, name = 'move', output_size = ACTION_SIZE)}},
        dict_of_action_encoder_class_dicts = {
            'move' : {'class' : partial(Vector_Encoder, name = 'move', input_size = ACTION_SIZE, output_size = 16)}},

        # Optional arguments. Any arguments left out are defaults. 
        # Layer 1's only inner state is named 'lower_layer_posterior_sample'.
        list_of_dict_of_inner_state_scalar_dicts = [
            {'vision' : {'upsilon_prior' : 20.0, 'upsilon_posterior' : 20.0, 'eta' : 0.2},
             'touch' : {'upsilon_prior' : 5.0, 'upsilon_posterior' : 5.0, 'eta' : 0.2}},
            {'lower_layer_posterior_sample' : {'eta' : 0.1}}],
        dict_of_action_scalar_dicts = {
            'move' : {
                'target_entropy' : -float(ACTION_SIZE),     # The usual SAC choice: negative of the action's size.
                'initial_alpha' : 0.1,
                'action_cost' : 0.1}},

        gamma = 0.9,                        
        lr = 1e-3,
        capacity = 128,
        max_steps = MAX_STEPS,
        verbose = True)                    



#------------------
# One episode: begin, then alternate step_in_episode and env.step.
#------------------

def run_episode(agent, env, train = True, deterministic = False):
    observation = env.reset()
    agent.begin()
    total_reward = 0.0
    for step in range(MAX_STEPS):
        step_dict = agent.step_in_episode(observation, deterministic = deterministic)
        next_observation, reward, done = env.step(step_dict['action']['move'])
        if train:
            agent.buffer.push(observation, step_dict['action'], reward, next_observation, done)
        total_reward += reward
        observation = next_observation
        if done:
            break
    return total_reward, step + 1, done



# Deterministic episodes for testing: how often does the dot end up at the goal, and how far away is it?
def evaluate(agent, env):
    at_goal, final_distances = 0, []
    for _ in range(EVALUATION_EPISODES):
        run_episode(agent, env, train = False, deterministic = True)
        final_distances.append(env.distance())
        at_goal += env.distance() < env.goal_radius
    return at_goal / EVALUATION_EPISODES, sum(final_distances) / EVALUATION_EPISODES



#------------------
# Figures: 
#   prior predictions against what actually happened, 
#   a "dream" in which the world model runs on its own prior, with no observations.
#------------------

def as_rgb(vision):
    dot, goal = vision[0], vision[1]
    return torch.stack([dot, goal, torch.zeros_like(dot)], dim = -1).clamp(0, 1).numpy()

def plot_predictions_and_dream(agent, env, path, real_steps = 6, dream_steps = 6):
    observation = env.reset()
    agent.begin()
    actual, predicted, dreamed = [], [], []
    for _ in range(real_steps):
        step_dict = agent.step_in_episode(observation, deterministic = True)
        actual.append(observation['vision'][0, 0])
        predicted.append(step_dict['prior_predictions'][0]['vision'][0, 0])
        observation, _, _ = env.step(step_dict['action']['move'])
    for _ in range(dream_steps):
        step_dict = agent.step_in_episode(use_posterior = False, deterministic = True)
        dreamed.append(step_dict['observation']['vision'][0, 0])

    rows = [('actual', actual), ('prior prediction', predicted), ('dream', dreamed)]
    columns = max(len(frames) for _, frames in rows)
    fig, axs = plt.subplots(3, columns, figsize = (1.6 * columns, 5.2), squeeze = False)
    for r, (label, frames) in enumerate(rows):
        for c in range(columns):
            ax = axs[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(frames):
                ax.imshow(as_rgb(frames[c]), origin = 'lower')
            else:
                ax.axis('off')
        axs[r][0].set_ylabel(label)
    fig.suptitle('Red: dot. Green: goal. The first prior prediction is made before any observation.', fontsize = 9)
    fig.tight_layout()
    fig.savefig(path, dpi = 120)
    plt.close(fig)



#------------------
# Train.
#------------------

if __name__ == '__main__':

    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok = True)

    env = Dot_World()
    agent = make_agent()

    at_goal, distance = evaluate(agent, env)
    print(f"Before training: ends at the goal {at_goal:.0%} of the time, {distance:.2f} away on average.\n")

    recent_rewards = []
    for episode in range(1, EPISODES + 1):
        episode_reward, _, _ = run_episode(agent, env)
        recent_rewards = (recent_rewards + [episode_reward / MAX_STEPS])[-EVALUATE_EVERY:]

        for _ in range(EPOCHS_PER_EPISODE):
            epoch_dict, epoch_dict_actor = agent.epoch(BATCH_SIZE)

        if episode % EVALUATE_EVERY == 0:
            at_goal, distance = evaluate(agent, env)
            prior = epoch_dict['accuracy_losses_prior']['layer_0']['vision']
            posterior = epoch_dict['accuracy_losses_posterior']['layer_0']['vision']
            print(
                f"episode {episode:4d}  |  "
                f"reward per step {sum(recent_rewards) / len(recent_rewards):5.2f}  |  "
                f"at goal {at_goal:4.0%}, {distance:.2f} away  |  "
                f"vision error: prior {prior:.4f}, posterior {posterior:.4f}  |  "
                f"curiosity {epoch_dict['curiosity']:.3f}  |  "
                f"alpha {agent.alpha('move').item():.3f}")

    figure = plot_training_log(agent)
    figure.savefig(OUTPUT_DIR / 'training_log.png', dpi = 100)
    plt.close(figure)
    plot_predictions_and_dream(agent, env, OUTPUT_DIR / 'predictions_and_dream.png')
    print(f"\nSaved figures to {OUTPUT_DIR}")