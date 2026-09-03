# This contains the logic for the training loop for both generator and discriminator 

import torch
import torch.optim as optim
import torch.nn as nn
from config import NOISE_DIM, BATCH_SIZE, N_EPOCHS, LR_G, LR_D


def train(G, D, X_train_n, Y_train_n, device,
           batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr_g=LR_G, lr_d=LR_D):

    g_opt = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train_n, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_n, dtype=torch.float32)
    n_samples = X_train_t.size(0)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            cond_batch = X_train_t[idx].to(device)
            real_batch = Y_train_t[idx].to(device)
            bs = cond_batch.size(0)

            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # train discriminator
            d_opt.zero_grad()
            d_real = D(real_batch, cond_batch)
            loss_d_real = criterion(d_real, real_labels)

            z = torch.randn(bs, NOISE_DIM, device=device)
            fake_batch = G(z, cond_batch).detach()
            d_fake = D(fake_batch, cond_batch)
            loss_d_fake = criterion(d_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            d_opt.step()

            # train generator
            g_opt.zero_grad()
            z = torch.randn(bs, NOISE_DIM, device=device)
            fake_batch = G(z, cond_batch)
            d_fake_for_g = D(fake_batch, cond_batch)
            loss_g = criterion(d_fake_for_g, real_labels)
            loss_g.backward()
            g_opt.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss_d={loss_d.item():.4f}, loss_g={loss_g.item():.4f}")

    return G, D