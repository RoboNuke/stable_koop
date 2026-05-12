import torch
import torch.nn as nn

from torch.distributions import Normal
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
    
    def act(self, inputs, role=""):
        # get mean and std dev from env
        mean_actions, log_std, outputs = self.compute(inputs, role)
        
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)
        
        # sample
        self._g_distribution = Normal(mean_actions, log_std.exp())
        u = self._g_distribution.rsample()
        #tanh to ensure -1,1
        actions = torch.tanh(u)
        
        # Use u directly for fresh samples, 
        # recover via atanh for replay buffer actions
        taken = inputs.get("taken_actions", None)
        u_for_logprob = torch.atanh(
            torch.clamp(taken, -1 + 1e-6, 1 - 1e-6)
        ) if taken is not None else u
        
        # adjust log prob for the tanh
        log_prob = self._g_distribution.log_prob(u_for_logprob)
        log_prob -= torch.log(1 - torch.tanh(u_for_logprob).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        outputs["mean_actions"] = actions
        return actions, log_prob, outputs


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
