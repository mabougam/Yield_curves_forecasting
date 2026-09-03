# This contains using the generator to create 10 future paths

import torch
import numpy as np
from config import NOISE_DIM


def generate_samples(G, cond_sample, device, n_generated=10):
    cond_tensor = torch.tensor(cond_sample, dtype=torch.float32).unsqueeze(0).to(device)

    G.eval()
    with torch.no_grad():
        generated_paths = []
        for _ in range(n_generated):
            z = torch.randn(1, NOISE_DIM, device=device)
            fake_seq = G(z, cond_tensor).cpu().numpy().squeeze(0)
            generated_paths.append(fake_seq)

    return np.array(generated_paths)