import os
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================
PRED_LOAD_CSV = "predicted_RT_hlx.csv"
OUTPUT_DIR = "results_final"
DEBUG = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plant
chiller_caps = np.array([400.0, 400.0, 400.0])
MAX_ACTIVE = 2

# Physical limits
MIN_PLR = 0.4
MAX_PLR_CHANGE = 0.1  # per 15 min

# Time step + anti short cycling
TIME_STEP_MIN = 15
MIN_ON_TIME_MIN = 60
MIN_OFF_TIME_MIN = 60

MIN_ON_STEPS = MIN_ON_TIME_MIN // TIME_STEP_MIN
MIN_OFF_STEPS = MIN_OFF_TIME_MIN // TIME_STEP_MIN

# Short drop duration (60 min)
SHORT_DROP_STEPS = 60 // TIME_STEP_MIN  # 4 steps

# GA parameters
POP_SIZE = 120
N_GA_ITERS = 250
MUTATION_RATE = 0.12
CROSSOVER_RATE = 0.6
np.random.seed(42)

# =========================================================
# POWER MODEL
# =========================================================
plr_points = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
kw_points  = np.array([75.81,75.81,77.49,94.81,116.8,141.2,167.6,195.3,226.1])

def chiller_power(plr, cap):
    if plr <= 0:
        return 0.0
    kw = np.interp(plr, plr_points, kw_points)
    return kw * (cap / 400.0)

def total_power(plrs, caps):
    p = np.array([chiller_power(plrs[i], caps[i]) for i in range(len(caps))])
    return p.sum(), p

# =========================================================
# STAGING WITH TIME LOCK
# =========================================================
def decide_active_chillers(demand, prev_n_active, on_steps_2nd, off_steps_2nd):
    if demand <= 400:
        desired = 1 if demand >= MIN_PLR * 400 else 0
    else:
        desired = 2

    # Enforce ON lock
    if prev_n_active == 2 and on_steps_2nd < MIN_ON_STEPS:
        return 2

    # Enforce OFF lock
    if prev_n_active == 1 and desired == 2 and off_steps_2nd < MIN_OFF_STEPS:
        return 1

    return desired

# =========================================================
# RAMP LIMIT
# =========================================================
def ramp_limit(plr, prev):
    out = plr.copy()
    for i in range(len(plr)):
        if prev[i] == 0 and plr[i] > 0:
            out[i] = max(plr[i], MIN_PLR)
        elif prev[i] > 0 and plr[i] > 0:
            delta = plr[i] - prev[i]
            if delta > MAX_PLR_CHANGE:
                out[i] = prev[i] + MAX_PLR_CHANGE
            elif delta < -MAX_PLR_CHANGE:
                out[i] = prev[i] - MAX_PLR_CHANGE
    for i in range(len(out)):
        if out[i] > 0:
            out[i] = max(out[i], MIN_PLR)
    return np.clip(out, 0.0, 1.0)

# =========================================================
# GA OPTIMIZATION
# =========================================================
def ga_optimize_fixed_n(demand, caps, n_active, prev_plr):
    n = len(caps)
    if n_active == 0 or demand <= 0:
        return np.zeros(n), 0.0

    def repair(x):
        x = x.copy()
        for i in range(n):
            if i < n_active:
                x[i] = np.clip(x[i], MIN_PLR, 1.0)
            else:
                x[i] = 0.0
        return ramp_limit(x, prev_plr)

    def fitness(x):
        x = repair(x)
        supplied = (caps * x).sum()
        tot_pwr, _ = total_power(x, caps)
        penalty = 1e6 * abs(supplied - demand)
        penalty += 20 * np.sum(x)
        return -(tot_pwr + penalty)

    # Initial population
    pop = []
    base = demand / (n_active * caps[0])
    for _ in range(POP_SIZE):
        x = np.zeros(n)
        for i in range(n_active):
            x[i] = np.clip(base + np.random.normal(0, 0.05), MIN_PLR, 1.0)
        pop.append(repair(x))

    # GA loop
    for _ in range(N_GA_ITERS):
        fits = np.array([fitness(x) for x in pop])
        probs = np.exp((fits - fits.max()) / (fits.std() + 1e-9))
        probs /= probs.sum()

        new_pop = []
        elite = np.argsort(fits)[-2:]
        new_pop.extend([pop[i].copy() for i in elite])

        while len(new_pop) < POP_SIZE:
            p1, p2 = np.random.choice(len(pop), 2, replace=False, p=probs)
            a, b = pop[p1], pop[p2]

            if np.random.rand() < CROSSOVER_RATE:
                cx = np.random.randint(1, n)
                child = np.concatenate([a[:cx], b[cx:]])
            else:
                child = a.copy()

            for i in range(n_active):
                if np.random.rand() < MUTATION_RATE:
                    child[i] += np.random.normal(0, 0.05)

            new_pop.append(repair(child))

        pop = new_pop

    best = repair(pop[np.argmax([fitness(x) for x in pop])])
    supplied = (caps * best).sum()
    if supplied > 0:
        best *= demand / supplied
        best = repair(best)

    tot_pwr, _ = total_power(best, caps)
    return best, tot_pwr

# =========================================================
# MAIN
# =========================================================
def main():
    df = pd.read_csv(PRED_LOAD_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    load_col = [c for c in df.columns if "rt" in c.lower() or "load" in c.lower()][0]

    demand_series = df[load_col].values
    timestamps = df["timestamp"]

    schedules, powers, kwrt = [], [], []

    prev_plr = np.zeros(len(chiller_caps))
    prev_n_active = 0
    on_steps_2nd = 0
    off_steps_2nd = MIN_OFF_STEPS

    drop_start_index = None

    for i, d in enumerate(demand_series):
        keep_second_chiller_on = False

        # -----------------------------
        # Detect short drop from previously running 2 chillers
        # -----------------------------
        if prev_n_active == 2 and 150 <= d < 400:
            if drop_start_index is None:
                drop_start_index = i  # start of drop
            drop_duration = i - drop_start_index + 1
            if drop_duration <= SHORT_DROP_STEPS:
                keep_second_chiller_on = True
        else:
            drop_start_index = None

        # -----------------------------
        # Decide active chillers normally
        # -----------------------------
        n_active = decide_active_chillers(d, prev_n_active, on_steps_2nd, off_steps_2nd)

        # Force second chiller ON if needed
        if keep_second_chiller_on:
            n_active = max(n_active, 2)

        # Optimize PLR with GA
        best_plr, tot_pwr = ga_optimize_fixed_n(d, chiller_caps, n_active, prev_plr)

        # Update on/off steps
        if n_active == 2:
            on_steps_2nd += 1
            off_steps_2nd = 0
        else:
            off_steps_2nd += 1
            on_steps_2nd = 0

        # Save results
        schedules.append(best_plr)
        powers.append(tot_pwr)
        kwrt.append(tot_pwr / d if d > 0 else 0.0)

        if DEBUG:
            print(
                f"[{timestamps[i]}] Load={d:.1f} RT | "
                f"Active={n_active} | PLR={best_plr} | "
                f"kW={tot_pwr:.1f} | kW/RT={kwrt[-1]:.3f} | "
                f"{'2nd ON KEPT' if keep_second_chiller_on else 'UPDATED'}"
            )

        prev_plr = best_plr.copy()
        prev_n_active = n_active

    # =========================================================
    out = pd.DataFrame(
        schedules,
        columns=[f"plr_chiller_{i+1}" for i in range(len(chiller_caps))]
    )
    out["timestamp"] = timestamps
    out["demand_RT"] = demand_series
    out["total_power_kW"] = powers
    out["kW_per_RT"] = kwrt

    out.to_csv(
        os.path.join(OUTPUT_DIR, "chiller_schedule_optimized_hlx.csv"),
        index=False
    )

    print("✅ FINAL chiller schedule saved (REALISTIC & STABLE).")

# =========================================================
if __name__ == "__main__":
    main()
