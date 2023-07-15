from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter
import time


writer = SummaryWriter('runs/exercise-3_3')

timesteps_sequence = [2500, 5000, 10000, 25000, 50000, 100000, 150000, 200000, 250000, 500000, 1000000, 2500000, 5000000]

for timesteps in timesteps_sequence:
    print(f'Training agent with {timesteps} timesteps')
    vec_env = make_vec_env('LunarLander-v2', n_envs=1)

    model = PPO('MlpPolicy', vec_env, verbose=1)
    start_training_time = time.time()
    model.learn(total_timesteps=timesteps)
    training_time = time.time() - start_training_time
    writer.add_scalar('Training time', training_time, timesteps)

    model.save(f'trained_models/ppo_lander_{timesteps}')

    max_steps = 1000
    episodes_rewards = []
    for i in range(10):
        obs = vec_env.reset()
        reward = 0
        dones = False
        for j in range(max_steps):
            action, _states = model.predict(obs)
            obs, r, dones, info = vec_env.step(action)
            reward += r

            if dones:
                break
        episodes_rewards.append(reward)

    avg_reward = sum(episodes_rewards) / len(episodes_rewards)
    print(f'{timesteps} timesteps ---- Average reward: {avg_reward}')
    writer.add_scalar('Average reward', avg_reward, timesteps)
