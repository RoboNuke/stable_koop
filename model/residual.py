import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2, reduction="sum",
                 hidden_size=64, hidden_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std, reduction)

        layers = [nn.Linear(self.num_observations, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.net(inputs["states"])
        return self.mean_layer(x), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, hidden_size=64, hidden_layers=2):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        input_size = self.num_observations + self.num_actions
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}
