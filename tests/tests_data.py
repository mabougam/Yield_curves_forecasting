# These are unit tests to check that the data used in the training is valid.
# The train-test split can't be random and must be chornological to avoid data leakage.
# This includes tests to verify the data standardisation and its reverse are correct.

import unittest
import numpy as np
from data.windowing import build_windows, train_test_split, normalize, denormalize
from config import COND_DAYS, GEN_DAYS, N_MATURITIES


class TestWindowing(unittest.TestCase):

    def setUp(self):
        self.n_days = 50
        self.data = np.random.randn(self.n_days, N_MATURITIES)

    def test_window_shapes(self):
        X_cond, Y_gen = build_windows(self.data)
        self.assertEqual(X_cond.shape[1:], (COND_DAYS, N_MATURITIES))
        self.assertEqual(Y_gen.shape[1:], (GEN_DAYS, N_MATURITIES))
        self.assertEqual(X_cond.shape[0], Y_gen.shape[0])

    def test_no_nans_in_windows(self):
        X_cond, Y_gen = build_windows(self.data)
        self.assertFalse(np.isnan(X_cond).any())
        self.assertFalse(np.isnan(Y_gen).any())

    def test_train_test_split_chronological(self):
        X_cond, Y_gen = build_windows(self.data)
        X_train, Y_train, X_test, Y_test = train_test_split(X_cond, Y_gen)
        self.assertEqual(len(X_train) + len(X_test), len(X_cond))

    def test_normalize_denormalize_are_inverses(self):
        arr = np.random.randn(5, 10, N_MATURITIES) * 3 + 2
        mean, std = arr.mean(), arr.std()
        arr_norm = normalize(arr, mean, std)
        arr_recovered = denormalize(arr_norm, mean, std)
        np.testing.assert_allclose(arr, arr_recovered, rtol=1e-5)
    
    def test_zero_std_raises_value_error(self):
        arr = np.ones((5, 3, N_MATURITIES))   # constant data -> std = 0
        mean, std = arr.mean(), arr.std()
        with self.assertRaises(ValueError):
            normalize(arr, mean, std)

    def test_normalize_output_has_no_nans_or_infs(self):
        arr = np.random.randn(10, COND_DAYS, N_MATURITIES) * 5 + 3
        mean, std = arr.mean(), arr.std()
        normalized = normalize(arr, mean, std)
        self.assertFalse(np.isnan(normalized).any())
        self.assertFalse(np.isinf(normalized).any())


if __name__ == '__main__':
    unittest.main()