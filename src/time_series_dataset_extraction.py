# generate_window_peak_datasets.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Assumes your existing simulator is saved as simulation.py
# and contains run_one(sim_type, noise, seed=0, window_points=500)
from simulation import run_one


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
WINDOW_POINTS = 500
N_PER_NOISE = 1000               # total per noise type
TRAIN_SIZE = 0.80
RANDOM_STATE = 42

NOISE_TYPES = ["white", "env", "demo"]
SIM_TYPE = "transcritical"       # only transcritical cases


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def build_one_example(sim_type, noise, seed, window_points=500):
    """
    Runs one simulation and returns a dict containing:
      - windowed time series
      - peak magnitude
      - peak time
      - optional metadata
    """
    (t_after, S_after, I_after), window, idx, params = run_one(
        sim_type=sim_type,
        noise=noise,
        seed=seed,
        window_points=window_points
    )

    # Peak over the FULL post-burn-in infected trajectory
    peak_idx = int(np.argmax(I_after))
    peak_magnitude = float(I_after[peak_idx])
    peak_time = float(t_after[peak_idx])

    row = {
        "peak_magnitude": peak_magnitude,
        "peak_time": peak_time,
        "noise_type": noise,
        "sim_type": sim_type,
        "seed": seed,
        "transition_index": -1 if idx is None else int(idx),
        "beta0": float(params["beta0"]),
        "beta1": float(params["beta1"]),
        "sigma1": float(params["sigma1"]),
        "sigma2": float(params["sigma2"]),
    }

    # Add windowed time series columns
    for j, val in enumerate(window):
        row[f"ts_{j:04d}"] = float(val)

    return row


def generate_dataset_for_noise(noise, n_total=1000, window_points=500):
    """
    Generates a dataset of only transcritical cases for one noise type.
    """
    rows = []

    for i in range(n_total):
        rows.append(
            build_one_example(
                SIM_TYPE,
                noise,
                seed=100000 + i,
                window_points=window_points
            )
        )

    df = pd.DataFrame(rows)
    return df


def save_train_test_csvs(df, prefix, train_size=0.8, random_state=42):
    """
    80/20 split, then save train/test CSVs.
    """
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=True
    )

    train_path = f"{prefix}_train.csv"
    test_path = f"{prefix}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved {train_path}: {train_df.shape}")
    print(f"Saved {test_path}:  {test_df.shape}")
    print("-" * 50)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    for noise in NOISE_TYPES:
        print(f"\nGenerating dataset for noise type: {noise}")
        df = generate_dataset_for_noise(
            noise=noise,
            n_total=N_PER_NOISE,
            window_points=WINDOW_POINTS
        )

        save_train_test_csvs(
            df=df,
            prefix=noise,
            train_size=TRAIN_SIZE,
            random_state=RANDOM_STATE
        )

    print("\nDone.")


if __name__ == "__main__":
    main()