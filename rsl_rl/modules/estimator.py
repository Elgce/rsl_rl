import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class Estimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 estimate_dim,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(Estimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.estimate_dim = estimate_dim
        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + estimate_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        est_state, z = self.encode(obs_history)
        return est_state.detach(), z.detach()

    def forward(self, obs_history):
        parts = self.encoder(obs_history.detach())
        est_state, z = parts[..., :self.estimate_dim], parts[..., self.estimate_dim:]
        z = F.normalize(z, dim=-1, p=2)
        return est_state.detach(), z.detach()

    def encode(self, obs_history):
        parts = self.encoder(obs_history.detach())
        est_state, z = parts[..., :self.estimate_dim], parts[..., self.estimate_dim:]
        z = F.normalize(z, dim=-1, p=2)
        return est_state, z

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        state = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+self.estimate_dim].detach()
        next_obs = next_critic_obs.detach()[:, self.estimate_dim:self.num_one_step_obs+self.estimate_dim]

        z_s = self.encoder(obs_history[:, :self.temporal_steps * self.num_one_step_obs])
        z_t = self.target(next_obs)
        pred_state, z_s = z_s[..., :self.estimate_dim], z_s[..., self.estimate_dim:]

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_state, state)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss, swap_loss


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None