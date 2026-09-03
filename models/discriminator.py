# The discriminator network architecture

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, cond_days=3, gen_days=10, n_maturities=13):
        super().__init__()
        self.total_days = cond_days + gen_days
        self.n_maturities = n_maturities

        # maturity-axis only, no day-mixing, no downsampling
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LayerNorm([32, self.total_days, n_maturities]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LayerNorm([16, self.total_days, n_maturities]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LayerNorm([8, self.total_days, n_maturities]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LayerNorm([1, self.total_days, n_maturities]),
            nn.LeakyReLU(0.2),
        )

        flat_dim = 1 * self.total_days * n_maturities

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, seq, cond):
        x = torch.cat([cond, seq], dim=1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)