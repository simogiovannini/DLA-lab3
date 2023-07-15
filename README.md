# DLA-lab3
Deep Learning Applications laboratory on Reinforcement Learning


## Policy Network
In `policy_network.py` we define the two-linear-layers network that will be used by the Reinforce learning algorithm.

The class has a `get_action(actions, temperature)` that returns the ID of the action to take. The `temperature` parameter must be within the interval [0, 1] and allows the user to choose the proportion of times in which the agent will select the action with the highest estimated probability instead of sampling from the distribution compuetd by the network.

If `temperature == 1` the agent only selects the action with the highest probability, if `temperature == 0` the agent only samples from the distribution.


## Reinforce Algorithm

The implementation of the Reinforce algorithm is defined in `utils/reinforce.py`. It's a vanilla version with no particular improvement.

## Exercise 3.1: Lunar Lander

## Exercise 3.3: Proximal Policy Optimization

## Requirements
You can use the `requirements.txt` file to create the conda environment to run the code in this repository.
