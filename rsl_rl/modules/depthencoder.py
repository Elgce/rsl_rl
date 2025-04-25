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
        
    def forward(self, x, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            if hidden_states is not None:
                raise ValueError("Hiddent states not passed to memory module during polcy update")
            out, _ = self.
            # batch mode: needs saved hidden states
        z = self.depth_encoder(x)
        h_prev = h_prev.to(z.device)
        h_next = self.gru_cell(z, h_prev)
        return h_next