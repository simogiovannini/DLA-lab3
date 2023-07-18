import numpy as np
import torch


def reinforce(policy, env, gamma=0.99, lr=1e-4, temperature=0.8, num_episodes=10, max_steps=1000, device='cpu', writer=None):
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    episodes_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state[0], dtype=torch.float32, device=device)

        log_probs = []
        rewards = []
        for step in range(max_steps):
            action, log_prob = policy(state, temperature)
            state, reward, terminated, truncated, _ = env.step(action)
            state = torch.tensor(state, dtype=torch.float32, device=device)
            log_probs.append(log_prob)
            rewards.append(reward)

            if terminated or truncated:
                total_reward = np.sum(rewards)
                episodes_rewards.append(total_reward)
                if episode % 50 == 0:
                    avg_rew = sum(episodes_rewards[-50:]) / 50
                    writer.add_scalar('Average reward', avg_rew, episode)
                    writer.add_scalar('N steps', step, episode)
                    print(f'Episode {episode}/{num_episodes} ---- Total reward: {total_reward} ---- Avg reward: {avg_rew}')
                break

        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + gamma ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = np.array(discounted_rewards)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards))
        log_probs = torch.stack(log_probs)
        policy_gradient = -log_probs * discounted_rewards

        policy.zero_grad()
        policy_gradient.sum().backward()
        optimizer.step()
    return episodes_rewards
