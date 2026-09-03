# Test the network training forward pass.

import unittest
import torch
import torch.optim as optim
import torch.nn as nn
from models.generator import Generator
from models.discriminator import Discriminator
from config import NOISE_DIM, COND_DAYS, GEN_DAYS, N_MATURITIES


class TestTrainingMechanics(unittest.TestCase):

    def setUp(self):
        self.G = Generator()
        self.D = Discriminator()
        self.criterion = nn.BCEWithLogitsLoss()

    def test_discriminator_weights_update(self):
        d_opt = optim.Adam(self.D.parameters(), lr=1e-4)
        before = [p.clone() for p in self.D.parameters()]

        seq = torch.randn(4, GEN_DAYS, N_MATURITIES)
        cond = torch.randn(4, COND_DAYS, N_MATURITIES)
        loss = self.criterion(self.D(seq, cond), torch.ones(4, 1))
        loss.backward()
        d_opt.step()

        after = list(self.D.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(changed)

    def test_generator_gradients_not_none(self):
        z = torch.randn(4, NOISE_DIM)
        cond = torch.randn(4, COND_DAYS, N_MATURITIES)
        fake = self.G(z, cond)
        loss = self.criterion(self.D(fake, cond), torch.ones(4, 1))
        loss.backward()

        for p in self.G.parameters():
            self.assertIsNotNone(p.grad)

    def test_one_training_step_runs(self):
        g_opt = optim.Adam(self.G.parameters(), lr=1e-4)
        d_opt = optim.Adam(self.D.parameters(), lr=1e-4)

        cond_batch = torch.randn(4, COND_DAYS, N_MATURITIES)
        real_batch = torch.randn(4, GEN_DAYS, N_MATURITIES)

        d_opt.zero_grad()
        d_real = self.D(real_batch, cond_batch)
        loss_d_real = self.criterion(d_real, torch.ones(4, 1))
        z = torch.randn(4, NOISE_DIM)
        fake_batch = self.G(z, cond_batch).detach()
        d_fake = self.D(fake_batch, cond_batch)
        loss_d_fake = self.criterion(d_fake, torch.zeros(4, 1))
        (loss_d_real + loss_d_fake).backward()
        d_opt.step()

        g_opt.zero_grad()
        z = torch.randn(4, NOISE_DIM)
        fake_batch = self.G(z, cond_batch)
        d_fake_for_g = self.D(fake_batch, cond_batch)
        loss_g = self.criterion(d_fake_for_g, torch.ones(4, 1))
        loss_g.backward()
        g_opt.step()


if __name__ == '__main__':
    unittest.main()