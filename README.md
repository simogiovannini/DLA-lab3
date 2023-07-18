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


![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/9f29a980-44d9-488d-80d9-dd8d60c88aa3)


## Requirements
You can use the `requirements.txt` file to create the conda environment to run the code in this repository.
