# This file contains the global variables used in the project. 
# Details on some of those variables will be outlined in the README.

MATURITY_ORDER = ['1 Mo','2 Mo','3 Mo','4 Mo','6 Mo','1 Yr','2 Yr','3 Yr',
                   '5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']

BASELINE_START_DATE = '2006-02-09'

COND_DAYS = 3
GEN_DAYS = 10
N_MATURITIES = 13
COND_DIM = COND_DAYS * N_MATURITIES
GEN_DIM = GEN_DAYS * N_MATURITIES
NOISE_DIM = 13

TRAIN_SPLIT = 0.8

BATCH_SIZE = 64
N_EPOCHS = 301
LR_G = 1e-4
LR_D = 1e-4

DATA_PATH = 'merged_yield_curve.csv'