import torch.nn as nn

class DepthEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=4),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ELU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=4),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ELU(),
            
            nn.Flatten()
        )
        self.fc = nn.Linear(4096, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        # x = x.view(x.size(0), -1)
        return self.fc(x)