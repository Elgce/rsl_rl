# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_one_step_obs,
        num_one_step_critic_obs,
        actor_history_length,
        critic_history_length,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_critic_obs = num_one_step_critic_obs
        self.actor_history_length = actor_history_length
        self.critic_history_length = critic_history_length
        self.actor_proprioceptive_obs_length = self.actor_history_length * self.num_one_step_obs
        self.critic_proprioceptive_obs_length = self.critic_history_length * self.num_one_step_critic_obs
        self.num_height_points = self.num_actor_obs - self.actor_proprioceptive_obs_length
        self.num_critic_height_points = self.num_critic_obs - self.critic_proprioceptive_obs_length
        self.actor_use_height = True if self.num_height_points > 0 else False
        self.num_actions = num_actions
        
        self.history_latent_dim = 32
        self.terrain_latent_dim = 32

        if self.actor_use_height:
            mlp_input_dim_a = num_one_step_obs + self.history_latent_dim + self.terrain_latent_dim
        else:
            mlp_input_dim_a = num_one_step_obs + self.history_latent_dim
        mlp_input_dim_c = num_critic_obs
        
        self.history_encoder = nn.Sequential(
            nn.Linear(self.num_one_step_obs * self.actor_history_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.history_latent_dim),
        )
        
        if self.actor_use_height:
            self.terrain_encoder = nn.Sequential(
                nn.Linear(self.num_one_step_obs + self.num_height_points, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.terrain_latent_dim),
            )
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'History Encoder: {self.history_encoder}')
        if self.actor_use_height:
            print(f'Terrain Encoder: {self.terrain_encoder}')

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        # compute mean
        history_latent = self.history_encoder(obs_history[:, :-self.num_height_points])
        
        if self.actor_use_height:
            terrain_latent = self.terrain_encoder(obs_history[:,-(self.num_height_points+self.num_one_step_obs):])
            # terrain_latent = self.terrain_encoder(obs_history[:,-(self.num_height_points):].reshape(-1, 1, 32, 32))
            actor_input = torch.cat((obs_history[:,-(self.num_height_points + self.num_one_step_obs):-self.num_height_points], history_latent, terrain_latent), dim=-1)
        else:
            actor_input = torch.cat((obs_history[:,-(self.num_one_step_obs):], history_latent), dim=-1)
        action_mean = self.actor(actor_input)
        if torch.any(torch.isnan(action_mean)) or torch.any(torch.isnan(self.std)):
            import ipdb; ipdb.set_trace()
        self.distribution = Normal(action_mean, action_mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history):
        history_latent = self.history_encoder(obs_history[:, :-self.num_height_points])
        if self.actor_use_height:
            terrain_latent = self.terrain_encoder(obs_history[:,-(self.num_height_points+self.num_one_step_obs):])
            # terrain_latent = self.terrain_encoder(obs_history[:,-(self.num_height_points):].reshape(-1, 1, 32, 32))
            actor_input = torch.cat((obs_history[:,-(self.num_height_points + self.num_one_step_obs):-self.num_height_points], history_latent, terrain_latent), dim=-1)
        else:
            actor_input = torch.cat((obs_history[:,-self.num_one_step_obs:], history_latent), dim=-1)
        action_mean = self.actor(actor_input)
        return action_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
