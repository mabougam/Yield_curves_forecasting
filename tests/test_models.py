# Test the generator and discriminator for correct array shapes and NaNs.

import unittest
import torch
from models.generator import Generator
from models.discriminator import Discriminator
from config import NOISE_DIM, COND_DAYS, GEN_DAYS, N_MATURITIES


class TestGenerator(unittest.TestCase):

    def setUp(self):
        self.G = Generator()

    def test_output_shape(self):
        z = torch.randn(4, NOISE_DIM)
        cond = torch.randn(4, COND_DAYS, N_MATURITIES)
        out = self.G(z, cond)
        self.assertEqual(out.shape, (4, GEN_DAYS, N_MATURITIES))

    def test_batch_size_one(self):
        z = torch.randn(1, NOISE_DIM)
        cond = torch.randn(1, COND_DAYS, N_MATURITIES)
        out = self.G(z, cond)
        self.assertEqual(out.shape, (1, GEN_DAYS, N_MATURITIES))

    def test_no_nans_in_output(self):
        z = torch.randn(8, NOISE_DIM)
        cond = torch.randn(8, COND_DAYS, N_MATURITIES)
        out = self.G(z, cond)
        self.assertFalse(torch.isnan(out).any())


class TestDiscriminator(unittest.TestCase):

    def setUp(self):
        self.D = Discriminator()

    def test_output_shape(self):
        seq = torch.randn(4, GEN_DAYS, N_MATURITIES)
        cond = torch.randn(4, COND_DAYS, N_MATURITIES)
        out = self.D(seq, cond)
        self.assertEqual(out.shape, (4, 1))

    def test_no_nans_in_output(self):
        seq = torch.randn(8, GEN_DAYS, N_MATURITIES)
        cond = torch.randn(8, COND_DAYS, N_MATURITIES)
        out = self.D(seq, cond)
        self.assertFalse(torch.isnan(out).any())


if __name__ == '__main__':
    unittest.main()