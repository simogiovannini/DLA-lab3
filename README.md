# DLA-lab3
Deep Learning Applications laboratory on Reinforcement Learning


## Policy Network
In `policy_network.py` we define the two-linear-layers network that will be used by the Reinforce learning algorithm.

The class has a `get_action(actions, temperature)` that returns the ID of the action to take. The `temperature` parameter must be within the interval [0, 1] and allows the user to choose the proportion of times in which the agent will select the action with the highest estimated probability instead of sampling from the distribution compuetd by the network.

If `temperature == 1` the agent only selects the action with the highest probability, if `temperature == 0` the agent only samples from the distribution.


## Reinforce Algorithm

The implementation of the Reinforce algorithm is defined in `utils/reinforce.py`. It's a vanilla version with no particular improvement.


## Exercise 3.1: Lunar Lander
In `3_1.py` we applied Reinforce algorithm on gymnasium's [Lunar Lander environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

The algorithm was tested with 3 different values for the `temperature` parameter setting the number of episodes to 30K:
- `temperature = 0.0` (represented in green)
- `temperature = 0.8` (represemted in pink)
- `temperature = (episode/num_episodes) * 0.9` that increases linearly during learning (represemted in yellow)

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/423810aa-9660-4495-ba7f-91b1743d71e3)

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/078c4a7a-7cc1-4ea7-b4be-5d471b751d5e)

The first graph represents the average reward collected by the agent during the last 10 episodes while the second represents the number of steps of the last episode.

Unexpectedly the best performance are provided by the version that always samples from the actions' distribution. The other two runs does not converge at all and it's clear also from the second graph.

The best model reaches an average reward of around 130.


## Exercise 3.3: Proximal Policy Optimization
In `3_3.py` we applied PPO from [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) to Lunar Lander.

The algorithm was tested varying the number of timesteps using these set of values: `[2'500, 5'000, 10'000, 25'000, 50'000, 100'000, 150'000, 200'000, 250'000, 500'000, 1'000'000, 2'500'000, 5'000'000]`.

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/76594ca9-2940-43cb-841e-3e74d0031de7)

In this graph the average reward of PPO is represented with the blue line and it's compared to the green line seen before. It's clear how PPO overperforms Reinforce both in time and in reached reward.
15 minutes of training are enough to reach way better performances.


## Extra: the role of Discount Factor

"The discount factor γ determines the tradeoff between immediate and future rewards: small γ values immediate, large is more farsighted".

This graph shows the average rewards obtained by the agent during learning as the discount factor value varies.
The tested values were:
- `gamma = 0.9` (in purple)
- `gamma = 0.95` (in green)
- `gamma = 0.99` (in cyan)
- `gamma = 0.999` (in orange)

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/eabe99f7-101e-4d71-ba70-cf38c7d59a5e)

These data highlight how the choice of discount factor is determinant of convergence and not necessarily a higher or lower value causes better behavior on the part of the agent. The conclusion that can be drawn is that the optimal value of the discount factor depends on the environment and is a problem-specific feature.

For Lunar Lander, it is noticeable how it is much more relevant to give more weight to actions in the more distant future rather than focusing on the near future,

## Extra: solve Lunar Lander with Q-Learning
To get a more comprehensive overview of deep reinforcement learning methods, it was decided to try solving the lunar lander environment using Q Learning.

The implementation of the network and Replay Buffer needed to run the algorithm can be found in `models/q_network.py` and `utils/replay_buffer.py` while the algorithm itself is in `3_4.py`.

In the graph below see the results obtained by training the agent on 5k episodes.

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/ca5b0a23-831f-493e-bff3-4912935a30ec)

Note how the learning process is much faster than that of Reinforce but is still very unstable.

Now that we have a new method of solving the environment we want to test whether the value of the discount factor has the same influence as we found for Reinforce. We then run DQL using the various values tried for Reinforce (`0.9, 0.95, 0.99, 0.999`).

The lines represent:
- `gamma = 0.9` (in cyan)
- `gamma = 0.95` (in pink)
- `gamma = 0.99` (in gray)
- `gamma = 0.999` (in orange)

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/ae16ace1-72c3-4260-afd7-01c38bda2344)

What was seen for Reinforce is confirmed i.e., the optimal value for the discount factor seems to be the same:`0.99`
Even with Q Learning different values lead to nonconvergence and ineffective agent learning.

Thus, the intuition that the identification of the discount factor is a problem spefical to the problem and not dependent on the technique used is confirmed.

### Stabilize Q-Learning

Despite achieving good results with training it can be seen that the learning process is very unstable and oscillating.

This could be due to the fact that, in the implementation of the algorithm, we update the parameters of the target network at the end of each episode making it a moving target for the policy network.

In order to try to stabilize the learning, the implementation of the method was modified by controlling the updating of the weights so that it was done every certain number of steps, changing the `UPDATE_TARGET_STEP` parameter.

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/bc36cb3d-5580-4346-8f4e-90e3eadb82d2)




## Requirements
You can use the `requirements.txt` file to create the conda environment to run the code in this repository.
