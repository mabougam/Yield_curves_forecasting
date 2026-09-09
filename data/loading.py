# This file is for loading and cleaning the data.

import pandas as pd

def load_merged_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    return df


def drop_all_nan_rows(df):
    maturity_cols = df.columns.drop('Date')
    mask = df[maturity_cols].isna().all(axis=1)
    return df[~mask].reset_index(drop=True)