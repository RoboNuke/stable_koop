import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    def __init__(self, obs_size=3, action_size=1, latent_size=16, num_layers=3):
        super().__init__()
        input_size = obs_size + action_size
        layers = []
        layers.append(nn.Linear(input_size, latent_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(latent_size, latent_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(latent_size, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)
