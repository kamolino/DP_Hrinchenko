"""Statistics for the ablations: Wilson interval, paired difference (Newcombe), exact McNemar, seed bootstrap."""
from math import comb, sqrt

import numpy as np

Z = 1.959964


def wilson(k, n, z=Z):
    """95 % Wilson score interval for k successes out of n (fractions)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def paired_counts(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    return int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum()), int((~x & ~y).sum())


def newcombe_paired(x, y, z=Z):
    """Difference p(x) - p(y) on the same scenes with the 95 % interval of Newcombe (1998), method 10."""
    a, b, c, d = paired_counts(x, y)
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    numerator = a * d - b * c
    numerator = max(numerator - n / 2, 0) if numerator > 0 else numerator
    phi = numerator / sqrt(denominator) if denominator else 0.0
    theta = p1 - p2
    low = theta - sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    high = theta + sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return theta, low, high


def mcnemar_exact(x, y):
    """Two-sided exact McNemar p-value from the discordant scenes (binomial, p = 0.5)."""
    _, b, c, _ = paired_counts(x, y)
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def compare(x, y):
    """x vs y on identical scenes: difference in percentage points, paired 95 % CI, McNemar p, discordant counts."""
    theta, low, high = newcombe_paired(x, y)
    _, b, c, _ = paired_counts(x, y)
    return dict(diff_pp=100 * theta, ci_pp=(100 * low, 100 * high), p_mcnemar=mcnemar_exact(x, y),
                only_first=b, only_second=c)


def seed_bootstrap(per_seed, reps=10000, rng=0):
    """per_seed: list of 0/1 arrays (one per training seed, same scenes). Resamples seeds, then scenes.
    Returns the mean success and its 95 % interval, which includes the training randomness."""
    rng = np.random.default_rng(rng)
    data = np.asarray(per_seed, float)
    seeds, scenes = data.shape
    means = np.empty(reps)
    for i in range(reps):
        s = rng.integers(seeds, size=seeds)
        c = rng.integers(scenes, size=scenes)
        means[i] = data[np.ix_(s, c)].mean()
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seed_bootstrap_diff(per_seed_x, per_seed_y, reps=10000, rng=0):
    """Difference of two trained variants (independent seeds, shared scenes): seeds resampled per arm,
    scenes resampled jointly for both arms."""
    rng = np.random.default_rng(rng)
    x, y = np.asarray(per_seed_x, float), np.asarray(per_seed_y, float)
    scenes = x.shape[1]
    diffs = np.empty(reps)
    for i in range(reps):
        c = rng.integers(scenes, size=scenes)
        sx = rng.integers(len(x), size=len(x))
        sy = rng.integers(len(y), size=len(y))
        diffs[i] = x[np.ix_(sx, c)].mean() - y[np.ix_(sy, c)].mean()
    return float(x.mean() - y.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
