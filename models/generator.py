# This is the generator network architecture

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim=13, cond_days=3, n_maturities=13, gen_days=10):
        super().__init__()
        self.cond_days = cond_days
        self.n_maturities = n_maturities
        self.gen_days = gen_days

        # maturity/day feature extraction, no spatial downsampling
        self.feature_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 3), stride=1, padding='same'),
            nn.LayerNorm([16, cond_days, n_maturities]),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.LayerNorm([32, cond_days, n_maturities]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.LayerNorm([64, cond_days, n_maturities]),
            nn.ReLU(),
        )

        # grow day axis only, maturity axis untouched
        self.day_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 1), stride=(2, 1),
                                 padding=(1, 0), output_padding=(1, 0)),
            nn.LayerNorm([32, 6, n_maturities]),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(5, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, z, cond):
        z_expanded = z.unsqueeze(1).expand(-1, self.cond_days, -1)
        x = torch.stack([cond, z_expanded], dim=1)

        x = self.feature_conv(x)
        x = self.day_upsample(x)

        return x.squeeze(1)