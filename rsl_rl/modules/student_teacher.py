# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.depthencoder import DepthEncoder

class StudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_one_step_student_obs,
        num_one_step_teacher_obs,
        student_history_length,
        teacher_history_length,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        self.num_student_obs = num_student_obs
        self.num_teacher_obs = num_teacher_obs
        self.num_one_step_student_obs = num_one_step_student_obs
        self.num_one_step_teacher_obs = num_one_step_teacher_obs
        self.student_history_length = student_history_length
        self.teacher_history_length = teacher_history_length
        self.student_propriceptive_obs_length = self.num_one_step_student_obs * self.student_history_length
        self.teacher_propriceptive_obs_length = self.num_one_step_teacher_obs * self.teacher_history_length
        self.num_student_height_points = self.num_student_obs - self.student_propriceptive_obs_length
        self.num_teacher_height_points = self.num_teacher_obs - self.teacher_propriceptive_obs_length
        self.num_actions = num_actions

        self.history_latent_dim = 32
        self.terrain_latent_dim = 32

        # student
        mlp_input_dim_s = self.num_one_step_student_obs + self.history_latent_dim + self.terrain_latent_dim
        self.student_history_encoder = nn.Sequential(
            nn.Linear(self.num_one_step_student_obs * self.student_history_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.history_latent_dim),
        )
        
        self.student_terrain_encoder = DepthEncoder(latent_dim=self.terrain_latent_dim)
        
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)
        print("====================================== Student Nerwork ======================================")
        print(f"Student MLP: {self.student}")
        print(f"Student History Encoder: {self.student_history_encoder}")
        print(f"Student Terrain Encoder: {self.student_terrain_encoder}")

        # teacher
        mlp_input_dim_t = self.num_one_step_teacher_obs + self.history_latent_dim + self.terrain_latent_dim
        self.teacher_history_encoder = nn.Sequential(
            nn.Linear(self.num_one_step_teacher_obs * self.teacher_history_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.history_latent_dim),
        )
        self.teacher_terrain_encoder = nn.Sequential(
            nn.Linear(self.num_one_step_teacher_obs + self.num_teacher_height_points, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.terrain_latent_dim),
        )
        
        teacher_layers = []
        teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        print("====================================== Teacher Nerwork ======================================")
        print(f"Teacher MLP: {self.teacher}")
        print(f"Teacher History Encoder: {self.teacher_history_encoder}")
        print(f"Teacher Terrain Encoder: {self.teacher_terrain_encoder}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
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

    def update_distribution(self, observations):
        history_latent = self.student_history_encoder(observations[:, :-self.num_student_height_points])
        terrain_latent = self.student_terrain_encoder(observations[:, -(self.num_student_height_points):].reshape(-1, 1, 128, 128))
        student_input = torch.cat((observations[:, -(self.num_student_height_points + self.num_one_step_student_obs):-self.num_student_height_points], history_latent, terrain_latent), dim=-1)
        action_mean = self.student(student_input)
        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        history_latent = self.student_history_encoder(observations[:, :-self.num_student_height_points])
        terrain_latent = self.student_terrain_encoder(observations[:, -(self.num_student_height_points):].reshape(-1, 1, 128, 128))
        student_input = torch.cat((observations[:, -(self.num_student_height_points + self.num_one_step_student_obs):-self.num_student_height_points], history_latent, terrain_latent), dim=-1)
        action_mean = self.student(student_input)
        return action_mean

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            history_latent = self.teacher_history_encoder(teacher_observations[:, :-self.num_teacher_height_points])
            terrain_latent = self.teacher_terrain_encoder(teacher_observations[:, -(self.num_teacher_height_points + self.num_one_step_teacher_obs):])
            teacher_input = torch.cat((teacher_observations[:, -(self.num_teacher_height_points + self.num_one_step_teacher_obs):-self.num_teacher_height_points], history_latent, terrain_latent), dim=-1)
            actions = self.teacher(teacher_input)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            # teacher_state_dict = {}
            # for key, value in state_dict.items():
            #     if "actor." in key:
            #         teacher_state_dict[key.replace("actor.", "")] = value
            # self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # # also load recurrent memory if teacher is recurrent
            # if self.is_recurrent and self.teacher_recurrent:
            #     raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            # # set flag for successfully loading the parameters
            # self.loaded_teacher = True
            # self.teacher.eval()
            # return False
            
            history_encoder_params = {
                key.replace("history_encoder.", ""): value
                for key, value in state_dict.items()
                if key.startswith("history_encoder")
            }
            self.teacher_history_encoder.load_state_dict(history_encoder_params)


            terrain_encoder_params = {
                key.replace("terrain_encoder.", ""): value
                for key, value in state_dict.items()
                if key.startswith("terrain_encoder")
            }
            self.teacher_terrain_encoder.load_state_dict(terrain_encoder_params)

            teacher_params = {
                key.replace("actor.", ""): value
                for key, value in state_dict.items()
                if key.startswith("actor")
            }
            self.teacher.load_state_dict(teacher_params)
            
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_history_encoder.eval()
            self.teacher_terrain_encoder.eval()
            return False
            
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
