# This file is for preprocessing the data

from config import MATURITY_ORDER, BASELINE_START_DATE

def filter_baseline_period(df, start_date=BASELINE_START_DATE):
    df_baseline = df[df['Date'] >= start_date].reset_index(drop=True)
    df_baseline = df_baseline.sort_values('Date').reset_index(drop=True)
    return df_baseline


def interpolate_maturities(df, maturity_order=MATURITY_ORDER):
    df[maturity_order] = df[maturity_order].interpolate(axis=1, method='linear')
    return df