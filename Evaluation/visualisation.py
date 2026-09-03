# Creating some visuals

import matplotlib.pyplot as plt
from config import MATURITY_ORDER
from evaluation.inference import generate_samples


def plot_generated_paths(G, X_test_n, Y_test_n, global_mean, global_std, device,
                            sample_idx=0, n_generated=10,
                            maturities_to_show=('1 Mo', '2 Yr', '10 Yr', '30 Yr')):

    cond_sample = X_test_n[sample_idx]
    generated_paths = generate_samples(G, cond_sample, device, n_generated)

    cond_real = cond_sample * global_std + global_mean
    generated_real = generated_paths * global_std + global_mean
    real_future = Y_test_n[sample_idx] * global_std + global_mean

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, mat in zip(axes.flat, maturities_to_show):
        m_idx = MATURITY_ORDER.index(mat)

        hist_days = list(range(-3, 0))
        ax.plot(hist_days, cond_real[:, m_idx], color='black', marker='o',
                 linewidth=2, label='3-day history (real)')

        gen_days = list(range(0, 10))
        for i in range(n_generated):
            path = generated_real[i, :, m_idx]
            ax.plot([hist_days[-1]] + gen_days, [cond_real[-1, m_idx]] + list(path),
                     alpha=0.5, linewidth=1.2)

        ax.plot([hist_days[-1]] + gen_days, [cond_real[-1, m_idx]] + list(real_future[:, m_idx]),
                 color='red', linewidth=2, linestyle='--', label='Actual future (real)')

        ax.axvline(x=-1, color='gray', linestyle=':', alpha=0.7)
        ax.set_title(mat)
        ax.set_xlabel('Days (0 = condition end)')
        ax.set_ylabel('Yield (%)')

    axes[0, 0].legend(fontsize=8)
    fig.suptitle('10 Generated 10-Day Paths, Conditioned on Preceding 3 Days', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_curve_shapes_grid(G, X_test_n, Y_test_n, global_mean, global_std, device,
                              sample_indices=(4, 5, 6, 7), days_to_check=(1, 4, 9),
                              n_generated=10):

    fig, axes = plt.subplots(len(sample_indices), len(days_to_check),
                                figsize=(16, 4 * len(sample_indices)), sharey='row')

    for row, sample_idx in enumerate(sample_indices):
        cond_sample = X_test_n[sample_idx]
        generated_paths = generate_samples(G, cond_sample, device, n_generated)

        generated_real = generated_paths * global_std + global_mean
        real_future = Y_test_n[sample_idx] * global_std + global_mean

        for col, day in enumerate(days_to_check):
            ax = axes[row, col]
            for i in range(n_generated):
                ax.plot(MATURITY_ORDER, generated_real[i, day], alpha=0.5, marker='o', markersize=3)
            ax.plot(MATURITY_ORDER, real_future[day], color='black', linewidth=2, marker='o', label='Actual')
            ax.tick_params(axis='x', rotation=45)

            if row == 0:
                ax.set_title(f'Day {day + 1} ahead')
            if col == 0:
                ax.set_ylabel(f'Sample {sample_idx}\nYield (%)')

    axes[0, 0].legend()
    fig.suptitle('Generated Curve Shapes (10 samples) at Selected Future Days — Multiple Test Windows',
                  fontsize=14)
    plt.tight_layout()
    plt.show()