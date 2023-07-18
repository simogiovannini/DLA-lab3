import gymnasium as gym
import torch
from models.policy_network import PolicyNetwork
from utils.reinforce import reinforce
from torch.utils.tensorboard import SummaryWriter


EPISODES = 30000
MAX_STEPS = 1000
GAMMA = 0.99
LR = 1e-4
TEMPERATURE = 0.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter(f'runs/exercise-3_1')

env = gym.make("LunarLander-v2")
observation, info = env.reset()


num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

policy = PolicyNetwork(num_inputs, num_actions, hidden_size=64).to(DEVICE)

episodes_rewards = reinforce(policy, env, gamma=GAMMA, lr=LR, temperature=TEMPERATURE, num_episodes=EPISODES, max_steps=MAX_STEPS, device=DEVICE, writer=writer)
env.close()

torch.save(policy.state_dict(), f'trained_models/reinforce-lunar-lander')
