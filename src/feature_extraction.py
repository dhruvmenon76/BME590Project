import numpy as np
import matplotlib.pyplot as plt

def _lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation (Pearson corr between x[t] and x[t+1])."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    x0 = x[:-1]
    x1 = x[1:]
    x0 = x0 - np.mean(x0)
    x1 = x1 - np.mean(x1)
    denom = (np.std(x0) * np.std(x1))
    if denom == 0:
        return np.nan
    return float(np.mean(x0 * x1) / denom)

def demo_feature_distributions(
    n: int = 200,
    window_points: int = 500,
    upper_q: float = 99.5,
    log_transform: bool = True,
    eps: float = 1e-12,
    base_seed: int = 0,
):
    """
    Runs all combinations of:
      sim_type ∈ {transcritical, null}
      noise    ∈ {white, env, demo}
    for n replicates each, computes variance + lag-1 autocorr (on window w),
    and plots a 3x2 grid:
      rows = noise type
      cols = [variance, lag1 autocorr]
    Each subplot overlays transcritical vs null distributions.

    Assumes you have:
      import simulation
      and simulation.run_one(sim_type, noise, seed=..., window_points=...)
    """
    import simulation  # expects simulation.py on path

    noises = ["white", "env", "demo"]
    sim_types = ["transcritical", "null"]

    # results[noise][sim_type] = {"var": [...], "ac1": [...]}
    results = {noise: {st: {"var": [], "ac1": []} for st in sim_types} for noise in noises}

    for noise in noises:
        for st in sim_types:
            for j in range(n):
                seed = base_seed + hash((noise, st)) % 10_000_000 + j
                (_t, _S, _I), w, _idx, _params = simulation.run_one(
                    st, noise, seed=seed, window_points=window_points
                )

                x = np.asarray(w, dtype=float)

                # Optional robust log transform (common for EWS features on incidence)
                if log_transform:
                    # Avoid log(0) but do NOT clamp the whole trajectory to a big eps
                    x = np.log(np.maximum(x, eps))

                # variance + lag-1 autocorrelation
                v = float(np.var(x, ddof=1)) if x.size >= 2 else np.nan
                a1 = _lag1_autocorr(x)

                results[noise][st]["var"].append(v)
                results[noise][st]["ac1"].append(a1)

    # --- Plot ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    col_titles = ["Variance", "Lag-1 autocorrelation"]

    for r, noise in enumerate(noises):
        # collect arrays
        tc_var = np.array(results[noise]["transcritical"]["var"], dtype=float)
        nu_var = np.array(results[noise]["null"]["var"], dtype=float)
        tc_ac1 = np.array(results[noise]["transcritical"]["ac1"], dtype=float)
        nu_ac1 = np.array(results[noise]["null"]["ac1"], dtype=float)

        # drop NaNs (can happen if series is constant / too short)
        tc_var = tc_var[~np.isnan(tc_var)]
        nu_var = nu_var[~np.isnan(nu_var)]
        tc_ac1 = tc_ac1[~np.isnan(tc_ac1)]
        nu_ac1 = nu_ac1[~np.isnan(nu_ac1)]

        axv = axes[r, 0]
        axa = axes[r, 1]

        # Overlay histograms
        bins_v = 30
        bins_a = 30

        axv.hist(tc_var, bins=bins_v, density=True, alpha=0.5, label="transcritical")
        axv.hist(nu_var, bins=bins_v, density=True, alpha=0.5, label="null")
        axv.set_ylabel(f"{noise}\nDensity")

        axa.hist(tc_ac1, bins=bins_a, density=True, alpha=0.5, label="transcritical")
        axa.hist(nu_ac1, bins=bins_a, density=True, alpha=0.5, label="null")

        if r == 0:
            axv.set_title(col_titles[0])
            axa.set_title(col_titles[1])

        # Helpful reference line for autocorr
        axa.axvline(0.0, linestyle="--", linewidth=1)

        # Show legend once
        if r == 0:
            axv.legend(fontsize=9)
            axa.legend(fontsize=9)

    axes[-1, 0].set_xlabel("Value")
    axes[-1, 1].set_xlabel("Value")

    # Optional: annotate what transform you used
    subtitle = f"Features computed on window w (n={n}, window_points={window_points})"
    subtitle += " | log-transform" if log_transform else " | raw"
    fig.suptitle(subtitle, y=1.02, fontsize=12)

    plt.tight_layout()
    plt.show()

    return results

    import numpy as np

def _lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation; returns 0.0 if undefined (constant / too short)."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:]  - x[1:].mean()
    denom = np.std(x0) * np.std(x1)
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.mean(x0 * x1) / denom)

def _skew_kurt(x: np.ndarray) -> tuple[float, float]:
    """
    Population-style skewness and kurtosis (not excess).
    Returns (0.0, 0.0) if undefined (constant / too short).
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0, 0.0
    mu = float(np.mean(x))
    xc = x - mu
    s = float(np.std(xc))
    if s == 0 or not np.isfinite(s):
        return 0.0, 0.0
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))
    skew = m3 / (s**3)
    kurt = m4 / (s**4)
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 0.0
    return skew, kurt

import numpy as np
import pandas as pd

def _lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:]  - x[1:].mean()
    denom = np.std(x0) * np.std(x1)
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.mean(x0 * x1) / denom)

def _skew_kurt(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0, 0.0
    mu = float(np.mean(x))
    xc = x - mu
    s = float(np.std(xc))
    if s == 0 or not np.isfinite(s):
        return 0.0, 0.0
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))
    skew = m3 / (s**3)
    kurt = m4 / (s**4)
    return float(skew), float(kurt)

def build_ews_dataframe(noise: str, n: int, window_points: int, base_seed: int = 0):
    """
    Returns a pandas DataFrame with columns:
        ['variance', 'ac1', 'cv', 'skewness', 'kurtosis', 'label', 'noise']

    label: 1 = transcritical, 0 = null
    """

    import simulation  # assumes simulation.py is importable

    n_each = n // 2
    rows = []

    def compute_features(window):
        # log1p is numerically stable and avoids log(0) issues
        x = np.log1p(np.asarray(window, dtype=float))

        var = float(np.var(x, ddof=1)) if x.size >= 2 else 0.0
        ac1 = _lag1_autocorr(x)

        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0
        cv = (std / abs(mean)) if mean != 0 else 0.0

        skew, kurt = _skew_kurt(x)

        return var, ac1, cv, skew, kurt

    # -------------------------
    # Transcritical
    # -------------------------
    for i in range(n_each):
        seed = base_seed + 10_000 + i
        (_t, _S, _I), w, _idx, _p = simulation.run_one(
            "transcritical", noise, seed=seed, window_points=window_points
        )

        var, ac1, cv, skew, kurt = compute_features(w)

        rows.append({
            "variance": var,
            "ac1": ac1,
            "cv": cv,
            "skewness": skew,
            "kurtosis": kurt,
            "label": 1,
            "noise": noise
        })

    # -------------------------
    # Null
    # -------------------------
    for i in range(n_each):
        seed = base_seed + 20_000 + i
        (_t, _S, _I), w, _idx, _p = simulation.run_one(
            "null", noise, seed=seed, window_points=window_points
        )

        var, ac1, cv, skew, kurt = compute_features(w)

        rows.append({
            "variance": var,
            "ac1": ac1,
            "cv": cv,
            "skewness": skew,
            "kurtosis": kurt,
            "label": 0,
            "noise": noise
        })

    df = pd.DataFrame(rows)

    return df