# This is a simple statistical test to investigate mode collapse

import torch
import numpy as np
from config import NOISE_DIM


def compute_sample_std(G, cond_sample, device, n_samples=50):
    cond_tensor = torch.tensor(cond_sample, dtype=torch.float32).unsqueeze(0).to(device)
    cond_repeated = cond_tensor.repeat(n_samples, 1, 1)

    z_batch = torch.randn(n_samples, NOISE_DIM, device=device)

    G.eval()
    with torch.no_grad():
        samples = G(z_batch, cond_repeated).cpu().numpy()

    return samples.std(axis=0)