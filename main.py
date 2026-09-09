import torch
from config import DATA_PATH, MATURITY_ORDER, N_EPOCHS
from data.loading import load_merged_data, drop_all_nan_rows
from data.preprocessing import filter_baseline_period, interpolate_maturities
from data.windowing import (build_windows, train_test_split,
                              compute_normalization_stats, normalize)
from models.generator import Generator
from models.discriminator import Discriminator
from training.train import train
from Evaluation.visualisation import plot_generated_paths, plot_curve_shapes_grid
from Evaluation.diversity import compute_sample_std


def main():
    df = load_merged_data(DATA_PATH)
    df = drop_all_nan_rows(df)
    df = filter_baseline_period(df)
    df = interpolate_maturities(df)

    data = df[MATURITY_ORDER].values
    X_cond, Y_gen = build_windows(data)
    X_train, Y_train, X_test, Y_test = train_test_split(X_cond, Y_gen)

    global_mean, global_std = compute_normalization_stats(X_train)
    X_train_n = normalize(X_train, global_mean, global_std)
    Y_train_n = normalize(Y_train, global_mean, global_std)
    X_test_n = normalize(X_test, global_mean, global_std)
    Y_test_n = normalize(Y_test, global_mean, global_std)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)
    D = Discriminator().to(device)

    G, D = train(G, D, X_train_n, Y_train_n, device, n_epochs=N_EPOCHS)

    plot_generated_paths(G, X_test_n, Y_test_n, global_mean, global_std, device)
    plot_curve_shapes_grid(G, X_test_n, Y_test_n, global_mean, global_std, device)

    std_result = compute_sample_std(G, X_test_n[4], device)
    print(std_result)


if __name__ == '__main__':
    main()