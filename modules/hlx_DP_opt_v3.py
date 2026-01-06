
# dp_optimizer.py
from __future__ import annotations

import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Tuple, Any
from datetime import datetime, timedelta

from deap import base, creator, tools
import warnings

warnings.filterwarnings("ignore")


# ------------------------------
# Configuration dataclass
# ------------------------------
@dataclass
class DPOptConfig:
    # Models (provide loaded models externally; e.g., scikit-learn GradientBoostingRegressor)
    cooling_load_model: Any                   # cl_mdl: predicts Cooling Load (RT)
    power_model: Any                          # p_mdl: predicts kWh
    # Features used by each model
    cl_features: Sequence[str] = ("temperature", "humidity", "weekday", "holiday", "year", "month", "day", "hour", "minute")
    p_features: Sequence[str] = ("DP", "Cooling_Load", "kW_RT", "Year", "Month", "Day", "Hour", "Minute")

    # Timestamp column (in input df)
    timestamp_col: str = "timestamp"

    # Office hours (local time)
    office_hour_start: int = 7
    office_hour_end: int = 21

    # DP bounds and GA parameters
    dp_low: float = 0.10
    dp_high: float = 0.25
    n_pop: int = 70
    n_gen: int = 30
    cx_prob: float = 0.5
    mut_prob: float = 0.2
    mut_sigma: float = 0.3
    # Optional short logic thresholds (kept as in your script)
    dp_rule_threshold_rt: float = 380.0
    dp_rule_low_min: float = 0.075
    dp_rule_low_max: float = 0.125
    dp_rule_high_min: float = 0.15
    dp_rule_high_max: float = 0.25

    # Random seed (for reproducibility)
    seed: Optional[int] = 42

    # Whether to compute holiday/weekday flags (if not present)
    compute_calendar_flags: bool = True


# ------------------------------
# Utilities
# ------------------------------
def _ensure_calendar_flags(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Ensure weekday (0/1) and holiday (0/1) are present; compute if missing."""
    out = df.copy()
    if "weekday" not in out.columns:
        out["weekday"] = pd.to_datetime(out[ts_col], errors="coerce").dt.dayofweek > 4
        out["weekday"] = out["weekday"].astype(int)
    if "holiday" not in out.columns:
        # Without external holiday library, default to 0 (non-holiday).
        out["holiday"] = 0
    # Expand datetime parts commonly used
    dt = pd.to_datetime(out[ts_col], errors="coerce")
    for name, accessor in [
        ("year", dt.dt.year),
        ("month", dt.dt.month),
        ("day", dt.dt.day),
        ("hour", dt.dt.hour),
        ("minute", dt.dt.minute),
    ]:
        if name not in out.columns:
            out[name] = accessor
    return out


def _prep_next_power_features(
    last_row: pd.Series,
    next_dt: datetime,
    predicted_rt: float,
    kW_RT_guess: float,
) -> np.ndarray:
    """
    Build the feature vector for the power model (kWh) for the next timestamp.
    Order must match config.p_features.
    """
    Year = next_dt.year
    Month = next_dt.month
    Day = next_dt.day
    Hour = next_dt.hour
    Minute = next_dt.minute
    # DP will be set by GA; here we keep 0 as placeholder in the vector
    return np.array([0.0, predicted_rt, kW_RT_guess, Year, Month, Day, Hour, Minute], dtype=float)


def _predict_cooling_load_series(X_cl: pd.DataFrame, mdl: Any) -> np.ndarray:
    """Predict cooling load (RT) for the provided feature frame."""
    pred = mdl.predict(X_cl)
    # Ensure 1D float array
    if isinstance(pred, (list, tuple)):
        pred = np.array(pred)
    return np.ravel(pred).astype(float)


def _forecast_power_for_dp(
    p_mdl: Any,
    p_feature_vec_template: np.ndarray,
    dp_value: float,
) -> float:
    """
    Insert DP into the first element, then predict next-step kWh with the power model.
    """
    x = p_feature_vec_template.copy()
    x[0] = float(dp_value)  # Set DP
    y = p_mdl.predict(x.reshape(1, -1))
    return float(np.ravel(y)[0])


def _dp_feasibility(dp: float, predicted_rt: float, cfg: DPOptConfig) -> bool:
    """
    Business-rule feasibility on DP based on predicted cooling load thresholds.
    """
    if predicted_rt <= cfg.dp_rule_threshold_rt:
        return (cfg.dp_rule_low_min <= dp <= cfg.dp_rule_low_max)
    else:
        return (cfg.dp_rule_high_min <= dp <= cfg.dp_rule_high_max)


# ------------------------------
# Main optimization function
# ------------------------------
def run_dp_optimization(df_weather: pd.DataFrame, config: DPOptConfig) -> pd.DataFrame:
    """
    Optimize Differential Pressure (DP) per 15-min step using weather-driven load prediction and kWh model.

    Parameters
    ----------
    df_weather : pd.DataFrame
        Must include at least:
        - config.timestamp_col
        - columns used in config.cl_features (temperature, humidity, weekday, holiday, year, month, day, hour, minute)
          If some calendar flags are missing and config.compute_calendar_flags=True, they will be computed.
        Data should be ordered by timestamp. It can be raw (not necessarily 15-min); you may resample externally.
    config : DPOptConfig
        Contains models and GA parameters.

    Returns
    -------
    pd.DataFrame
        Columns:
        - timestamp
        - Pred_Cl (predicted cooling load RT for each step, from cooling_load_model)
        - optimized_kwh (predicted kWh after optimizing DP via power_model)
        - DP (optimized Differential Pressure)
        - performance_% (delta vs mean prediction heuristic, kept from your original)
    """
    # Seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    # Ensure timestamp is datetime and sorted
    if config.timestamp_col not in df_weather.columns:
        raise KeyError(f"Timestamp column '{config.timestamp_col}' not found in df_weather.")
    df = df_weather.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors="coerce")
    df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    # Ensure calendar flags and datetime parts if needed
    if config.compute_calendar_flags:
        df = _ensure_calendar_flags(df, config.timestamp_col)

    # Validate CL feature presence
    missing = [c for c in config.cl_features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing cooling-load feature columns: {missing}")

    # Build cooling load feature frame
    X_CL = df[list(config.cl_features)].copy()

    # Precompute predicted cooling load for *each* step (vector)
    Pred_Cl_series = _predict_cooling_load_series(X_CL, config.cooling_load_model)

    # DEAP setup (guard against re-creation on multiple imports/runs)
    # Try to create unique classes once; if already created, skip
    try:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    except Exception:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    except Exception:
        pass

    toolbox = base.Toolbox()
    # # DP generator inside allowed bounds
    # def _set_vals():
    #     return [random.uniform(config.dp_low, config.dp_high)]
    # toolbox.register("set_vals", _set_vals)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.set_vals, n=1)

    # Each gene should be a FLOAT, not a list
    toolbox.register("attr_dp", random.uniform, config.dp_low, config.dp_high)

    # Individual: list of one float [DP]
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_dp, n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluate: given an individual's DP, forecast power using current next-step template
    def _evaluate(individual):
        # individual is a list like [DP]
        print(f" individual: {individual[0]}  (should be float or number)")
        dp_val = float(individual[0])
        # Feasibility check (soft): if infeasible, penalize heavily by returning big kWh
        if not _dp_feasibility(dp_val, current_pred_rt, config):
            return (1e9,)  # minimizing
        kwh = _forecast_power_for_dp(config.power_model, current_p_vec, dp_val)
        return (kwh,)

    toolbox.register("evaluate", _evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.25)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=config.mut_sigma, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Iterate over steps (15-min assumed by your workflow; uses the dataframe timestamps)
    results = []
    for i in range(len(df)):
        curr_dt = df[config.timestamp_col].iloc[i]

        # Build "next timestamp" (15-min ahead) for the power prediction features
        next_dt = curr_dt + timedelta(minutes=15)

        # kW/RT guess: keep your original heuristic (random uniform in [0.5, 0.65])
        kW_RT_guess = random.uniform(0.5, 0.65)

        # Obtain predicted RT for current step (you used last element of cl_mdl.predict on history;
        # here we use the per-step vector directly)
        current_pred_rt = Pred_Cl_series[i]

        # Prepare the next-step power features template (DP will be varied by GA)
        last_row = df.iloc[i]
        current_p_vec = _prep_next_power_features(
            last_row=last_row,
            next_dt=next_dt,
            predicted_rt=current_pred_rt,
            kW_RT_guess=kW_RT_guess,
        )

        # Office hours logic
        if not (config.office_hour_start <= next_dt.hour < config.office_hour_end):
            optimized_kwh = 0.0
            dp_opt = 0.0
            perf_pct = 0.0
            office = False
        else:
            # Genetic Algorithm loop
            pop = toolbox.population(n=config.n_pop)

            # Evaluate initial population
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Evolve
            for _ in range(config.n_gen):
                offspring = tools.selTournament(pop, len(pop), tournsize=3)
                offspring = list(map(toolbox.clone, offspring))

                # Crossover
                for c1, c2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < config.cx_prob:
                        toolbox.mate(c1, c2)


if __name__ == "__main__":
    import argparse
    import pickle as p
    import sys
    import os
    from configparser import ConfigParser

    parser = argparse.ArgumentParser(
        description="Run DP optimization using parameters from a config.ini file."
    )
    parser.add_argument(
        "--config",
        default="config.ini",
        help="Path to INI configuration file (default: config.ini)"
    )
    args = parser.parse_args()

    # 1) Read config.ini via ConfigParser
    config = ConfigParser()
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        config.read(args.config)
    except Exception as e:
        print(f"❌ Failed to read config file '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    if not config.has_section("dp_optimization"):
        print("❌ Missing [dp_optimization] section in config.ini", file=sys.stderr)
        sys.exit(1)

    section = "dp_optimization"

    # Helper getters with fallbacks
    def cfg_str(key, default=None, required=False):
        if required and not config.has_option(section, key):
            print(f"❌ Missing required key '{key}' in [dp_optimization]", file=sys.stderr)
            sys.exit(1)
        return config.get(section, key, fallback=default)

    def cfg_int(key, default=None):
        return config.getint(section, key, fallback=default)

    def cfg_float(key, default=None):
        return config.getfloat(section, key, fallback=default)

    def cfg_bool(key, default=False):
        # Accepts true/false, yes/no, 1/0
        return config.getboolean(section, key, fallback=default)

    # 2) Required paths
    csv_path           = cfg_str("csv_path", required=True)
    cooling_model_path = cfg_str("cooling_load_model_path", required=True)
    power_model_path   = cfg_str("power_model_path", required=True)

    # 3) Optional settings
    timestamp_col     = cfg_str("timestamp_col", "timestamp")
    office_start      = cfg_int("office_hour_start", 7)
    office_end        = cfg_int("office_hour_end", 21)
    dp_low            = cfg_float("dp_low", 0.10)
    dp_high           = cfg_float("dp_high", 0.25)
    n_pop             = cfg_int("n_pop", 70)
    n_gen             = cfg_int("n_gen", 30)
    cx_prob           = cfg_float("cx_prob", 0.5)
    mut_prob          = cfg_float("mut_prob", 0.2)
    mut_sigma         = cfg_float("mut_sigma", 0.3)
    seed              = cfg_int("seed", 42)
    compute_calendar  = cfg_bool("compute_calendar_flags", True)

    # DP rule thresholds
    dp_rule_threshold_rt = cfg_float("dp_rule_threshold_rt", 380.0)
    dp_rule_low_min      = cfg_float("dp_rule_low_min", 0.075)
    dp_rule_low_max      = cfg_float("dp_rule_low_max", 0.125)
    dp_rule_high_min     = cfg_float("dp_rule_high_min", 0.15)
    dp_rule_high_max     = cfg_float("dp_rule_high_max", 0.25)

    # Output
    save_csv_path = cfg_str("save_csv_path", "DP_Optimization_HLX_Result.csv")
    no_save       = cfg_bool("no_save", False)

    # Optional feature overrides (comma-separated)
    cl_features_str = cfg_str("cl_features", "")
    p_features_str  = cfg_str("p_features", "")

    # 4) Load input CSV
    try:
        df_weather = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Failed to read CSV '{csv_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if timestamp_col not in df_weather.columns:
        print(f"❌ Timestamp column '{timestamp_col}' not found in CSV.", file=sys.stderr)
        sys.exit(1)
    df_weather[timestamp_col] = pd.to_datetime(df_weather[timestamp_col], errors="coerce")
    df_weather = df_weather.sort_values(timestamp_col).reset_index(drop=True)

    # 5) Load models
    try:
        with open(cooling_model_path, "rb") as f:
            cl_mdl = p.load(f)
    except Exception as e:
        print(f"❌ Failed to load cooling-load model from '{cooling_model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(power_model_path, "rb") as f:
            p_mdl = p.load(f)
    except Exception as e:
        print(f"❌ Failed to load power model from '{power_model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 6) Build DPOptConfig
    cfg_kwargs = dict(
        cooling_load_model=cl_mdl,
        power_model=p_mdl,
        timestamp_col=timestamp_col,
        office_hour_start=office_start,
        office_hour_end=office_end,
        dp_low=dp_low,
        dp_high=dp_high,
        n_pop=n_pop,
        n_gen=n_gen,
        cx_prob=cx_prob,
        mut_prob=mut_prob,
        mut_sigma=mut_sigma,
        seed=seed,
        compute_calendar_flags=compute_calendar,
        dp_rule_threshold_rt=dp_rule_threshold_rt,
        dp_rule_low_min=dp_rule_low_min,
        dp_rule_low_max=dp_rule_low_max,
        dp_rule_high_min=dp_rule_high_min,
        dp_rule_high_max=dp_rule_high_max,
    )

    # Feature overrides if present
    if cl_features_str:
        cfg_kwargs["cl_features"] = tuple(s.strip() for s in cl_features_str.split(",") if s.strip())
    if p_features_str:
        cfg_kwargs["p_features"] = tuple(s.strip() for s in p_features_str.split(",") if s.strip())

    cfg = DPOptConfig(**cfg_kwargs)

    # 7) Run optimization
    results_df = run_dp_optimization(df_weather, cfg)

    # 8) Print and optionally save
    print("✅ Optimization complete. Preview:")
    print(results_df.head(10).to_string(index=False))

    if not no_save:
        try:
            results_df.to_csv(save_csv_path, index=False)
            print(f"💾 Results saved to: {save_csv_path}")
        except Exception as e:
            print(f"⚠️ Could not save results to '{save_csv_path}': {e}", file=sys.stderr)
