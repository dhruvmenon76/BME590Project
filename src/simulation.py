import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Global simulation settings
# ----------------------------
DT = 0.1
T_MAX = 1500.0
BURN_IN = 100.0

# ----------------------------
# Fixed model parameters
# ----------------------------
LAMBDA = 100.0   # recruitment into S
ALPHA  = 1.0     # recovery
MU     = 1.0     # natural death

S0 = LAMBDA/MU
I0 = 5.0

# Numerical safety to keep S,I positive
EPS = 1e-12

def beta_c(lambda_=LAMBDA, alpha=ALPHA, mu=MU):
    # β_c is the transmission threshold corresponding to R0 = 1
    return mu * (alpha + mu) / lambda_

BETA_C = beta_c()

def beta_t(t, beta0, beta1):
    # β(t): transmission rate at time t
    return beta0 + beta1 * t

def R0_t(t, beta0, beta1, lambda_=LAMBDA, alpha=ALPHA, mu=MU):
    # R0(t): reproduction number at time t
    return beta_t(t, beta0, beta1) * lambda_ / (mu * (alpha + mu))

def transition_time(beta0, beta1):
    # time t* when β(t*) = β_c (equivalently R0(t*) = 1)
    if beta1 <= 0:
        return None
    t_star = (BETA_C - beta0) / beta1
    if 0.0 <= t_star <= T_MAX:
        return float(t_star)
    return None
def tri(rng, low, mode, high):
    return float(rng.triangular(low, mode, high))

def sample_params(rng, sim_type):
    # β0 ~ Tri(0, βc/4, βc/2)
    beta0 = tri(rng, 0.0, BETA_C/4.0, BETA_C/2.0)

    # base slope that would reach βc at T_MAX
    base = (BETA_C - beta0) / T_MAX

    if sim_type == "null":
        # ensure β0 + β1*T_MAX <= βc
        beta1 = tri(rng, 0.0, 0.5*max(0.0, base), max(0.0, base)) if base > 0 else 0.0
    elif sim_type == "transcritical":
        # ensure β0 + β1*T_MAX > βc
        low = max(0.0, base)
        high = max(low, (2.0*BETA_C - beta0) / T_MAX)
        beta1 = tri(rng, low, 0.5*high, high)
    else:
        raise ValueError("sim_type must be 'null' or 'transcritical'")

    # noise intensities for white/env noise
    sigma1 = tri(rng, 0.0, 0.5, 1.0)
    sigma2 = tri(rng, 0.0, 0.5, 1.0)

    return beta0, beta1, sigma1, sigma2
def drift_SI(S, I, t, beta0, beta1):
    bt = beta_t(t, beta0, beta1)
    dS = LAMBDA - bt*S*I - MU*S
    dI = bt*S*I - (ALPHA + MU)*I
    return dS, dI
#added ETA term to demo to avoid collpase of I
def simulate_white(rng, beta0, beta1, sigma1, sigma2):
    n = int(T_MAX/DT) + 1
    t = np.linspace(0.0, T_MAX, n)
    S = np.zeros(n); I = np.zeros(n)
    S[0], I[0] = S0, I0

    for k in range(n-1):
        dS, dI = drift_SI(S[k], I[k], t[k], beta0, beta1)
        dW1 = rng.normal(0.0, np.sqrt(DT))
        dW2 = rng.normal(0.0, np.sqrt(DT))
        S[k+1] = max(EPS, S[k] + dS*DT + sigma1*dW1)
        I[k+1] = max(EPS, I[k] + dI*DT + sigma2*dW2)

    return t, S, I


def simulate_env(rng, beta0, beta1, sigma1, sigma2):
    n = int(T_MAX/DT) + 1
    t = np.linspace(0.0, T_MAX, n)
    S = np.zeros(n); I = np.zeros(n)
    S[0], I[0] = S0, I0

    for k in range(n-1):
        dS, dI = drift_SI(S[k], I[k], t[k], beta0, beta1)
        dW1 = rng.normal(0.0, np.sqrt(DT))
        dW2 = rng.normal(0.0, np.sqrt(DT))
        S[k+1] = max(EPS, S[k] + dS*DT + sigma1*S[k]*dW1)
        I[k+1] = max(EPS, I[k] + dI*DT + sigma2*I[k]*dW2)

    return t, S, I


def simulate_demo(rng, beta0, beta1, ETA=1e-6):
    """
    Demographic diffusion-style noise.
    Coefficients depend on rates; two independent Wiener terms are used.

    ETA: tiny "importation" term to prevent I(t) from pinning at EPS.
         Increase (1e-5, 1e-4) if I still collapses; decrease if it dominates dynamics.
    """
    n = int(T_MAX/DT) + 1
    t = np.linspace(0.0, T_MAX, n)
    S = np.zeros(n); I = np.zeros(n)
    S[0], I[0] = S0, I0

    for k in range(n-1):
        bt = beta_t(t[k], beta0, beta1)

        # deterministic drift
        dS, dI = drift_SI(S[k], I[k], t[k], beta0, beta1)

        # ---- NEW: importation term ----
        dI = dI + ETA
        # ------------------------------

        # diffusion coefficients based on rates
        a = LAMBDA + bt*S[k]*I[k] + MU*S[k]
        b = -bt*S[k]*I[k]
        c = bt*S[k]*I[k] + (ALPHA + MU)*I[k]
        d = a*c - b*b
        e = a + c + 2.0*d

        if (e <= 1e-14) or (not np.isfinite(e)):
            g11 = g12 = g21 = g22 = 0.0
        else:
            g11 = (a + d) / e
            g12 = b / e
            g21 = b / e
            g22 = (c + d) / e

        dW1 = rng.normal(0.0, np.sqrt(DT))
        dW2 = rng.normal(0.0, np.sqrt(DT))

        S_next = S[k] + dS*DT + g11*dW1 + g12*dW2
        I_next = I[k] + dI*DT + g21*dW1 + g22*dW2

        S[k+1] = max(EPS, S_next)
        I[k+1] = max(EPS, I_next)

    return t, S, I
def simulate_deterministic(beta0, beta1):
    """
    Deterministic SIR (actually SI here, since R is implicit) using the same Euler scheme
    and drift as the stochastic simulators, but with NO noise terms.
    """
    n = int(T_MAX / DT) + 1
    t = np.linspace(0.0, T_MAX, n)
    S = np.zeros(n); I = np.zeros(n)
    S[0], I[0] = S0, I0

    for k in range(n - 1):
        dS, dI = drift_SI(S[k], I[k], t[k], beta0, beta1)
        S[k+1] = max(EPS, S[k] + dS * DT)
        I[k+1] = max(EPS, I[k] + dI * DT)

    return t, S, I
def apply_burn_in(t, S, I):
    keep = t >= BURN_IN
    return t[keep] - BURN_IN, S[keep], I[keep]

def extract_window(I, t_after, beta0, beta1, window_points=500):
    t_star = transition_time(beta0, beta1)
    if t_star is None:
        # null: last window
        return I[-window_points:].copy(), None

    t_star_after = t_star - BURN_IN
    if t_star_after <= t_after[0]:
        # crossed during burn-in; take first window
        return I[:window_points].copy(), 0

    idx = np.searchsorted(t_after, t_star_after, side="right") - 1
    idx = int(np.clip(idx, 0, len(I)-1))
    start = max(0, idx - window_points + 1)
    w = I[start:idx+1].copy()

    if len(w) < window_points:
        w = np.concatenate([np.full(window_points-len(w), w[0]), w])

    return w, idx
def run_one(sim_type, noise, seed=0, window_points=500):
    rng = np.random.default_rng(seed)
    beta0, beta1, sigma1, sigma2 = sample_params(rng, sim_type)

    if noise == "white":
        t, S, I = simulate_white(rng, beta0, beta1, sigma1, sigma2)
    elif noise == "env":
        t, S, I = simulate_env(rng, beta0, beta1, sigma1, sigma2)
    elif noise == "demo":
        t, S, I = simulate_demo(rng, beta0, beta1)
    elif noise in ["none", "deterministic", "det"]:   # <- NEW
        t, S, I = simulate_deterministic(beta0, beta1)
    else:
        raise ValueError("noise must be 'white', 'env', 'demo', or 'none'")

    t2, S2, I2 = apply_burn_in(t, S, I)
    w, idx = extract_window(I2, t2, beta0, beta1, window_points=window_points)

    t2, S2, I2 = apply_burn_in(t, S, I)

    #w = I2.copy()     # use full curve
    #idx = None        # no transition index needed



    params = dict(beta0=beta0, beta1=beta1, sigma1=sigma1, sigma2=sigma2)
    return (t2, S2, I2), w, idx, params

def demo_plot(window_points=500, upper_q=99.5):
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=False)
    noises = ["white", "env", "demo"]
    EPS_PLOT = 1e-8

    for r, noise in enumerate(noises):
        # -------------------------
        # transcritical
        # -------------------------
        (t, S, I), w, idx, p = run_one("transcritical", noise, seed=10+r, window_points=window_points)
        ax = axes[r, 0]

        if noise in ["env", "demo"]:
            I_plot = np.maximum(I, EPS_PLOT)
            ax.plot(t, I_plot, label="I(t)")
            ax.set_yscale("log")

            y_lo = np.percentile(I_plot, 1)      # or 0.5
            y_hi = np.percentile(I_plot, upper_q)  # e.g. 99.5

            y_lo = max(y_lo, EPS_PLOT)
            y_hi = max(y_hi, y_lo * 10)          # ensure at least 1 decade visible

            ax.set_ylim(y_lo, y_hi)

            # (optional) show true max in title for debugging
            ax.set_title(f"{noise} — transcritical (max={np.max(I_plot):.2e})")
        else:
            ax.plot(t, I, label="I(t)")
            ax.set_title(f"{noise} — transcritical")

        if idx is not None:
            ax.axvline(t[idx], linestyle="--", label="threshold (R0=1)")
            ax.axvspan(t[max(0, idx-window_points+1)], t[idx], alpha=0.2, label="window")

        ax.set_ylabel("Infected I")
        ax.legend(fontsize=8)

        # -------------------------
        # null
        # -------------------------
        (t, S, I), w, idx, p = run_one("null", noise, seed=20+r, window_points=window_points)
        ax = axes[r, 1]

        if noise in ["env", "demo"]:
            I_plot = np.maximum(I, EPS_PLOT)
            ax.plot(t, I_plot, label="I(t)")
            ax.set_yscale("log")

            y_lo = np.percentile(I_plot, 1)      # or 0.5
            y_hi = np.percentile(I_plot, upper_q)  # e.g. 99.5

            y_lo = max(y_lo, EPS_PLOT)
            y_hi = max(y_hi, y_lo * 10)          # ensure at least 1 decade visible

            ax.set_ylim(y_lo, y_hi)

            ax.set_title(f"{noise} — null (max={np.max(I_plot):.2e})")
        else:
            ax.plot(t, I, label="I(t)")
            ax.set_title(f"{noise} — null")

        ax.axvspan(t[-window_points], t[-1], alpha=0.2, label="window")
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("time (after burn-in)")
    axes[-1, 1].set_xlabel("time (after burn-in)")
    plt.tight_layout()
    plt.show()

