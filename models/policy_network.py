import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.layer1 = nn.Linear(num_inputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x, temperature):
        x = F.relu(self.layer1(x))
        actions = F.softmax(self.layer2(x))
        action = self.get_action(actions, temperature)
        log_prob_action = torch.log(actions.squeeze(0))[action]
        return action, log_prob_action

    def get_action(self, actions, temperature):
        if np.random.random() < temperature:
            return torch.argmax(actions).item()
        else:
            return np.random.choice(list(range(self.num_actions)), p=actions.squeeze(0).detach().cpu().numpy())
