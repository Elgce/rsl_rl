import torch.nn as nn
import torch

from rsl_rl.utils import unpad_trajectories

# class DepthEncoder(nn.Module):
#     def __init__(self, latent_dim=64):
#         super().__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#             nn.ELU(),

#             nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=4),
#             nn.Conv2d(16, 32, kernel_size=1),
#             nn.ELU(),

#             nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=4),
#             nn.Conv2d(32, 64, kernel_size=1),
#             nn.ELU(),
            
#             nn.Flatten()
#         )
#         # self.fc = nn.Linear(4096, latent_dim)
#         self.fc = nn.Linear(384, latent_dim)

#     def forward(self, x):
#         x = self.conv_layers(x)
#         # x = x.view(x.size(0), -1)
#         return self.fc(x)

class DepthEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1, groups=4),
            nn.Conv3d(16, 32, kernel_size=1),
            nn.ELU(),

            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, groups=4),
            nn.Conv3d(32, 64, kernel_size=1),
            nn.ELU(),
            
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 16, 16, 16)
            n_feat = self.conv_layers(dummy).shape[1]
        # self.fc = nn.Linear(4096, latent_dim)
        self.fc = nn.Linear(n_feat, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        # x = x.view(x.size(0), -1)
        return self.fc(x)
    
class DepthGRUEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.depth_encoder = DepthEncoder(latent_dim=latent_dim)
        self.gru_cell = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)
        self.hidden_states = None
        
    def forward(self, x, hidden_states=None):
        B = x.shape[0]
        device = x.device
        if hidden_states is None:
            h0 = self.hidden_states
        else:
            h0 = hidden_states
        if h0 is None:
            h0 = torch.zeros(B, self.depth_encoder.fc.out_features, device=device)
        z = self.depth_encoder(x)
        try:
            h1 = self.gru_cell(z, h0)
        except:
            import ipdb; ipdb.set_trace()
        if hidden_states is None:
            self.hidden_states = h1
        return h1
            
    def reset(self, dones: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None):
        if dones is None:  # reset all hidden states
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        else:
            # reset hidden states of done environments
            if self.hidden_states is not None:
                self.hidden_states[dones, :] = 0.0
    
    def detach_hidden_states(self, dones: torch.Tensor | None = None):
        if self.hidden_states is None:
            return
        if dones is None:  # detach all hidden states
            self.hidden_states = self.hidden_states.detach()
        else:  # detach hidden states of done environments
            hs = self.hidden_states
            hs[dones == 1, :] = hs[dones == 1, :].detach()
            self.hidden_states = hs
            