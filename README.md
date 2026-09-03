# Conditional GAN for US Treasury Yield Curve Generation

## Overview

A yield curve plots interest rates across bonds of comparable credit quality but differing maturities — most commonly US Treasuries. Its shape (upward-sloping, flat, or inverted) is one of the more closely watched signals in fixed income markets, since it reflects the market's collective view on future rates, inflation, and growth, and feeds directly into bond pricing and interest-rate-risk management.

This project designs, implements, and trains a conditional GAN (cGAN) that generates 10 plausible 10-day paths of the US Treasury yield curve, conditioned on the preceding 3 days of the curve.

---

## Data

Source: [US Treasury Yield Curve Rates, 1990–2024](https://home.treasury.gov/interest-rates-data-csv-archive)

Columns: date, and maturities from 1 month to 30 years.

### Data quality findings and merged dataset

The originally provided file was missing the 2000–2003 period. A separate CSV covering this gap was sourced and merged in, producing `merged_yield_curve.csv`, which this project uses as its input.

Inspecting the merged data by maturity availability over time surfaced the following gaps:

| Period | Missing maturities |
|---|---|
| Latest date → 19/10/2022 | None — all maturities available |
| 18/10/2022 → 16/10/2018 | 4 Mo |
| 15/10/2018 → 9/2/2006 | 2 Mo, 4 Mo |
| 8/2/2006 → 19/2/2002 | 2 Mo, 4 Mo, 30 Yr |
| 15/2/2002 → 31/7/2001 | 2 Mo, 4 Mo |
| 30/7/2001 → 1/10/1993 | 1 Mo, 2 Mo, 4 Mo |
| 30/9/1993 → 2/1/1990 | 1 Mo, 2 Mo, 4 Mo, 20 Yr |

**Decision: restrict training data to 9 Feb 2006 → present.**

Missing maturities fall into two categories:
- **Interior maturities** (2 Mo, 4 Mo) — bounded on both sides by observed maturities (e.g. 1 Mo and 3 Mo), so linear interpolation is a reasonable, low-risk way to fill them in.
- **Edge maturities** (1 Mo, 30 Yr, and in one stretch 20 Yr) — sit at either end of the curve, so filling them in would require extrapolation rather than interpolation, which is a meaningfully weaker assumption.

To avoid extrapolating edge maturities altogether, the dataset was restricted to the period where only interior maturities are missing (9 Feb 2006 onward). This sacrifices roughly 16 years of earlier data but avoids introducing fabricated edge values into the training set.

**Interpolation assumption.** Missing interior maturities (2 Mo, 4 Mo) were filled via linear interpolation *across the maturity axis*, not across time. This is a reasonable assumption for two reasons: the interpolated points sit only a month or two from their known neighbours, so large jumps are unlikely, and short-to-medium maturities (1 Mo through roughly 10 Yr) tend to move together fairly tightly, since they're driven by similar macroeconomic and Fed-policy factors. Note that this same logic does not extend to the 30 Yr point, which behaves somewhat independently of the rest of the curve — driven more by auction dynamics, its own demand base, and supply/duration considerations — which is part of the reasoning for excluding, rather than interpolating, periods where 30 Yr was missing.

---

## Architecture and Design Choices

### Problem framing

- **Input (condition):** 3 consecutive days of the yield curve, 13 maturities each — shape `(3, 13)`.
- **Output:** 10 generated future days, 13 maturities each — shape `(10, 13)`.
- **Noise:** a 13-dimensional random vector, giving the generator a source of randomness so that a single condition can map to many distinct, plausible futures.

### Normalization

Data was standardised using a single **global mean and standard deviation, computed from the training set only** (not per-maturity, and not per-curve).

This choice was deliberate, and two alternatives were explicitly rejected:
- **Per-maturity normalization** (a separate mean/std for each of the 13 columns) removes each maturity's absolute level, which distorts the *shape* of the curve once visualised — the curve appears artificially flat across most maturities before dropping sharply at the long end, since 30 Yr behaves as a relative outlier under this scheme.
- **Per-curve (per-day) normalization** was also considered infeasible, since it strips out the curve's baseline yield level entirely — a feature that matters, since "where the curve sits" (a high-rate regime vs a low-rate regime) is itself meaningful information, not noise to be normalised away.

A single global scalar mean/std preserves the real shape and relative level of the curve, at the cost of not equalising volatility across maturities — judged to be the more defensible trade-off for this task.

### Model architecture

Both networks were arrived at through iterative experimentation; the reasoning below reflects what was actually tried, not just the final result.

**Baseline:** a simple MLP for both generator and discriminator, taking flattened, concatenated inputs. This trained but showed some day-to-day noisiness and limited curve realism.

**Generator — final architecture:**
1. Noise is concatenated with the 3-day condition **along the channel dimension** (not added element-wise, and not stacked as a "4th day" — see below), giving an input of shape `(batch, 2, 3, 13)`.
2. A stack of `Conv2d` layers with `(3,3)` kernels and `stride=1` extracts joint day-and-maturity features, without changing the spatial size.
3. A `ConvTranspose2d` stage grows the day axis only (3 → 6 → 10) using asymmetric kernels (e.g. `(3,1)`, `(5,1)`) so the maturity axis is never mixed or resized during upsampling.

Two earlier design choices for injecting noise were tried and discarded:
- Treating the noise vector as a literal "4th day" stacked onto the 3-day condition — discarded as physically arbitrary, since noise is not a real observed day.
- Adding the noise element-wise to each of the 3 real days — this caused the network to collapse during training, so it was replaced with channel-wise concatenation, which trained stably and preserved output diversity.

**Discriminator — final architecture:** a stack of `Conv1d`/`Conv2d`-equivalent layers using `(1,3)`-style kernels (maturity axis only, `stride=1`, no downsampling), followed by `Linear` layers producing a single real/fake logit.

The asymmetry between generator (2D convolution) and discriminator (1D, maturity-only convolution) was a deliberate final choice, arrived at after testing several combinations:

| Variant tried | Outcome |
|---|---|
| 1D discriminator (maturity-only) + 2D generator | Best-performing combination: stable training, confirmed sample diversity, reasonable curve shapes |
| 2D discriminator (`(3,3)`, day+maturity mixing) + 2D generator | Did not resolve day-to-day noise as hoped, and gave the discriminator disproportionate power over the generator |
| Fully convolutional discriminator using strided downsampling to shrink the input to a single value | Consistently produced mode collapse |

The fully-convolutional, strided-downsampling discriminator's failure is treated as an informative negative result: strided convolution is a tool for efficiently building large receptive fields over *long* sequences, where it trades resolution for context cheaply. With a sequence length of only 13, a large receptive field is already achievable in 2–3 layers with `stride=1`; aggressive striding instead discards positional resolution the network didn't need to give up, and plausibly let the discriminator become too confident, too quickly, starving the generator of a useful gradient signal.

Given the generator's more demanding task (jointly modelling day and maturity structure to produce something coherent along both axes), while the discriminator's simpler binary classification task performed well without that extra capacity, keeping the discriminator comparatively lightweight and the generator comparatively expressive was the most efficient configuration found.

### Loss function

`BCEWithLogitsLoss` was used for both networks. This computes the standard GAN binary cross-entropy loss on raw logits (no `sigmoid` applied manually), using a numerically stable formulation that avoids the `log(0)` instability that can occur with a naive `sigmoid()` + `BCELoss()` implementation, particularly as the discriminator becomes more confident over training.

### Evaluation

Three complementary checks were used, since GAN loss curves alone are not a reliable indicator of sample quality — a stable, converged loss can equally indicate a well-trained generator or a collapsed one producing safe, low-diversity output.

1. **Path evolution plot** — for a given 3-day condition, all 10 generated 10-day paths are plotted alongside the real historical continuation, across several representative maturities (1 Mo, 2 Yr, 10 Yr, 30 Yr).
2. **Curve shape grid** — for several different 3-day conditions (rows), the 10 generated curves at selected future days (day 2, 5, and 10 ahead) are plotted against the actual realised curve (columns).
3. **Diversity check** — the standard deviation across 50 generated samples for a fixed condition is computed per day and per maturity, to directly confirm the generator is using the noise input meaningfully (i.e. not exhibiting mode collapse) rather than relying on visual inspection alone.

---

## How to Run

### Requirements

```
pip install -r requirements.txt
```

### Project structure

```
config.py              # shared constants (window sizes, maturity list, hyperparameters)
data/                   # loading, cleaning, interpolation, windowing, normalization
models/                 # Generator and Discriminator definitions
training/               # training loop
evaluation/             # inference, diversity check, plotting
tests/                  # unittest suite
main.py                 # end-to-end pipeline
```

### Running the pipeline

```
python main.py
```

This loads `merged_yield_curve.csv`, applies the cleaning/interpolation/windowing steps described above, trains the GAN, and produces the evaluation plots and diversity statistics.

### Running the tests

```
python -m unittest discover tests
```

Covers: data windowing and normalization correctness, model input/output shapes (including batch-size and malformed-input edge cases), and core training mechanics (weights updating, gradients flowing correctly between generator and discriminator).

---

## Assumptions and Constraints

- Only data from 9 Feb 2006 onward is used, to avoid extrapolating edge maturities (1 Mo, 30 Yr).
- Interior missing maturities (2 Mo, 4 Mo) are linearly interpolated across the maturity axis.
- Normalization uses a single global mean/std, computed from the training split only, to avoid data leakage.
- The train/test split is chronological (not shuffled), since this is time-series data and a random split would leak future information into training.
- No validation set is used for this baseline; only train and test.

---

## Limitations and Future Work

**Current limitations:**
- The generator's handling of the maturity dimension is noticeably more robust than its handling of the temporal (day-to-day) dimension. Day-to-day generated paths are somewhat noisier than real Treasury movements typically are, despite this improving with the move to 2D convolution in the generator.
- The test suite covers the core mechanics well but could be extended with further edge cases.

**Future work considered worth investigating:**
1. Using a model such as XGBoost to estimate the missing edge maturities (e.g. 1 Mo and 2 Mo in the early 1990s), which would allow the full 1990–2024 dataset to be used for training rather than restricting to post-2006 data.
2. A U-Net-style generator with skip connections between the feature-extraction and upsampling stages, which may help preserve fine-grained temporal continuity that the current architecture doesn't explicitly encourage.
3. Exploring transformer-based generator and discriminator architectures, which may be better suited to capturing longer-range temporal dependencies than convolution alone.
4. Incorporating a financial model (e.g. a no-arbitrage or term-structure model) as a supplementary term in the loss function, to explicitly encourage economically plausible curve dynamics rather than relying solely on the adversarial signal.

Items 2 and 3 in particular are the most direct candidates for addressing the temporal-smoothness limitation noted above.
