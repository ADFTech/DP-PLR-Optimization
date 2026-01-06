
#!/usr/bin/env python3
import os
import sys
import argparse
import re
import configparser
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from train_kwh import train_kwh
from train_cooling_load import train_cooling_load

# --- Config helpers ----------------------------------------------------------

def load_db_config(path: str = "config.ini") -> Dict[str, Any]:
    cfg = configparser.ConfigParser()
    read_ok = cfg.read(path)
    if not read_ok:
        raise FileNotFoundError(f"Could not read config file at {path}")
    if "database" not in cfg:
        raise KeyError("Missing [database] section in config file.")
    db = cfg["database"]
    if "database" not in db:
        raise KeyError("Missing 'database' key in [database] section.")

    return {
        "engine": db.get("engine", "postgresql"),
        "database": db["database"],
        "host": db.get("host", "localhost"),
        "port": db.getint("port", 5432),
        "username": db.get("username", None),
        "password": db.get("password", None),
        "ssl_mode": db.get("ssl_mode", "none"),
        "ssl_ca_cert_path": db.get("ssl_ca_cert_path", None),
        "ssl_cert_path": db.get("ssl_cert_path", None),
        "ssl_key_path": db.get("ssl_key_path", None),
    }


def load_training_config(path: str = "config.ini") -> Dict[str, Any]:
    """
    Read [training] section from config.ini. All keys are optional.
    Keys:
      - view (str)
      - use_exact_kwh (bool-like: true/false/1/0/yes/no)
      - kwh_model_path (str)
      - cl_model_path (str)
    """
    cfg = configparser.ConfigParser()
    read_ok = cfg.read(path)
    if not read_ok:
        # No file: return empty config; DSN resolution will still work via env/CLI.
        return {}
    if "training" not in cfg:
        return {}
    tr = cfg["training"]

    def get_bool(key: str, default: Optional[bool] = None) -> Optional[bool]:
        if key not in tr:
            return default
        val = tr.get(key, "").strip().lower()
        return val in ("1", "true", "yes", "on")

    return {
        "view": tr.get("view", None),
        "use_exact_kwh": get_bool("use_exact_kwh", None),
        "kwh_model_path": tr.get("kwh_model_path", None),
        "cl_model_path": tr.get("cl_model_path", None),
    }


def build_connection_url(db_cfg: Dict[str, Any]) -> str:
    engine = db_cfg["engine"].lower()
    database = db_cfg["database"]
    host = db_cfg.get("host")
    port = db_cfg.get("port")
    username = db_cfg.get("username")
    password = db_cfg.get("password")

    auth = ""
    if username:
        auth = username
        if password:
            auth += f":{password}"
        auth += "@"

    if engine in ("postgresql", "postgres"):
        driver = "postgresql+psycopg2"
        base = f"{driver}://{auth}{host}:{port}/{database}"
        params = []
        ssl_mode = (db_cfg.get("ssl_mode") or "none").lower()
        if ssl_mode and ssl_mode != "none":
            params.append(f"sslmode={ssl_mode}")
        if db_cfg.get("ssl_ca_cert_path"):
            params.append(f"sslrootcert={db_cfg['ssl_ca_cert_path']}")
        if db_cfg.get("ssl_cert_path"):
            params.append(f"sslcert={db_cfg['ssl_cert_path']}")
        if db_cfg.get("ssl_key_path"):
            params.append(f"sslkey={db_cfg['ssl_key_path']}")
        return base + ("?" + "&".join(params) if params else "")
    elif engine in ("mysql", "mariadb"):
        driver = "mysql+pymysql"
        base = f"{driver}://{auth}{host}:{port}/{database}"
        params = []
        if db_cfg.get("ssl_ca_cert_path"):
            params.append(f"ssl_ca={db_cfg['ssl_ca_cert_path']}")
        if db_cfg.get("ssl_cert_path"):
            params.append(f"ssl_cert={db_cfg['ssl_cert_path']}")
        if db_cfg.get("ssl_key_path"):
            params.append(f"ssl_key={db_cfg['ssl_key_path']}")
        return base + ("?" + "&".join(params) if params else "")
    elif engine == "sqlite":
        if database == ":memory:":
            return "sqlite:///:memory:"
        return f"sqlite:///{os.path.abspath(database)}"
    else:
        raise ValueError(f"Unsupported engine: {engine}")


# --- Data fetch --------------------------------------------------------------

SCHEMA_VIEW_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")

def validate_view_name(view_name: str) -> None:
    if not view_name or not SCHEMA_VIEW_RE.match(view_name):
        raise ValueError(
            f"Invalid view/table identifier '{view_name}'. "
            "Use alphanumerics/underscores, optionally 'schema.view'."
        )

def fetch_dataset(dsn: str, view_name: str) -> pd.DataFrame:
    validate_view_name(view_name)
    print(view_name)
    engine = create_engine(dsn, pool_pre_ping=True)
    sql = f"""
        SELECT
          timestamp_15m,
          temp,
          humidity,
          "HLX_L18_AHU_Header_Low_Zone_Differential_Pressure",
          "HLX_B1_Chiller_BTU_METER_Cooling_Capacity",
          "HLX_B1_Chiller_DPM_MSB_IN_1_kW",
          "HLX_B1_Chiller_DPM_MSB_IN_2_kW",
          kwh_15m,
          kwh_15m_exact
        FROM {view_name}
        ORDER BY timestamp_15m;
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, parse_dates=['timestamp_15m'])
    return df


# --- CLI / Merge options -----------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run both trainings using one dataset fetch.')
    parser.add_argument('--config', default='config.ini', help='Path to config.ini with [database] and [training].')
    parser.add_argument('--dsn', default=None, help='SQLAlchemy DSN for HLX DB (overrides config/env).')
    parser.add_argument('--view', default=None, help='View name (boundary/exactsum/meterdelta variant, or schema.view).')
    parser.add_argument('--use-exact-kwh', action='store_true', help='Train KWH using kwh_15m_exact if available.')
    parser.add_argument('--kwh-model-path', default=None, help='Path to save the KWH model.')
    parser.add_argument('--cl-model-path', default=None, help='Path to save the Cooling Load model.')
    return parser.parse_args()


def coalesce(*vals):
    """Return the first non-empty value (None or empty string considered empty)."""
    for v in vals:
        if isinstance(v, str):
            if v is not None and v.strip() != "":
                return v
        else:
            if v is not None:
                return v
    return None


def resolve_dsn(args: argparse.Namespace, training_cfg: Dict[str, Any]) -> str:
    # 1) CLI
    if args.dsn:
        return args.dsn
    # 2) ENV
    env_dsn = os.getenv('HLX_DSN')
    if env_dsn:
        return env_dsn
    # 3) Config.ini [database]
    db_cfg = load_db_config(args.config)
    return build_connection_url(db_cfg)


def resolve_options(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge all options with precedence:
      CLI > Env > Config > Defaults
    """
    training_cfg = load_training_config(args.config)

    # Resolve DSN
    dsn = resolve_dsn(args, training_cfg)

    # Resolve view
    view = coalesce(
        args.view,
        os.getenv('HLX_VIEW'),
        training_cfg.get("view"),
    )
    if not view:
        # Enforce that a view is provided via some source
        raise ValueError("View name not provided. Use --view, HLX_VIEW env var, or [training].view in config.ini.")

    # Resolve boolean for use_exact_kwh
    # CLI flag: True when present; otherwise None
    cli_use_exact = args.use_exact_kwh if args.use_exact_kwh else None
    env_use_exact_raw = os.getenv("HLX_USE_EXACT_KWH")
    env_use_exact = None
    if env_use_exact_raw is not None:
        env_use_exact = env_use_exact_raw.strip().lower() in ("1", "true", "yes", "on")

    use_exact_kwh = coalesce(
        cli_use_exact,
        env_use_exact,
        training_cfg.get("use_exact_kwh"),
        False,  # default
    )

    # Resolve model paths
    kwh_model_path = coalesce(
        args.kwh_model_path,
        os.getenv("HLX_KWH_MODEL_PATH"),
        training_cfg.get("kwh_model_path"),
        "models/hlx_gbt_kwh_model.pkl",
    )
    cl_model_path = coalesce(
        args.cl_model_path,
        os.getenv("HLX_CL_MODEL_PATH"),
        training_cfg.get("cl_model_path"),
        "models/hlx_gbt_cooling_load_model.pkl",
    )

    return {
        "dsn": dsn,
        "view": view,
        "use_exact_kwh": bool(use_exact_kwh),
        "kwh_model_path": kwh_model_path,
        "cl_model_path": cl_model_path,
        "config_path": args.config,
        "dsn_source": "--dsn" if args.dsn else ("HLX_DSN env" if os.getenv("HLX_DSN") else args.config),
        "view_source": "--view" if args.view else ("HLX_VIEW env" if os.getenv("HLX_VIEW") else args.config),
        "use_exact_source": "--use-exact-kwh" if args.use_exact_kwh else (
            "HLX_USE_EXACT_KWH env" if os.getenv("HLX_USE_EXACT_KWH") else args.config),
        "kwh_path_source": "--kwh-model-path" if args.kwh_model_path else (
            "HLX_KWH_MODEL_PATH env" if os.getenv("HLX_KWH_MODEL_PATH") else args.config),
        "cl_path_source": "--cl-model-path" if args.cl_model_path else (
            "HLX_CL_MODEL_PATH env" if os.getenv("HLX_CL_MODEL_PATH") else args.config),
    }


def main():
    args = parse_args()

    try:
        opts = resolve_options(args)
    except Exception as e:
        print(f'ERROR: Option resolution failed. {e}', file=sys.stderr)
        sys.exit(1)

    print(f'[INFO] Using DSN source: {opts["dsn_source"]}')
    print(f'[INFO] Using view source: {opts["view_source"]}')
    print(f'[INFO] Using use_exact_kwh source: {opts["use_exact_source"]}')
    print(f'[INFO] KWH path source: {opts["kwh_path_source"]}')
    print(f'[INFO] CL path source: {opts["cl_path_source"]}')

    print(f'[INFO] Pulling dataset from {opts["view"]} ...')
    try:
        df = fetch_dataset(opts["dsn"], opts["view"])
    except Exception as e:
        print(f'ERROR: Failed to fetch dataset from {opts["view"]}. {e}', file=sys.stderr)
        sys.exit(1)
    print('[INFO] Rows fetched: {n}'.format(n=len(df)))
    timestamp_col = "timestamp_15m"
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['year'] = df[timestamp_col].dt.year
    df['month'] = df[timestamp_col].dt.month
    df['day'] = df[timestamp_col].dt.day
    df['hour'] = df[timestamp_col].dt.hour
    df['minute'] = df[timestamp_col].dt.minute
    df['weekday'] = df[timestamp_col].dt.weekday
    df['holiday'] = df['weekday'].isin([5, 6]).astype(int)  # weekend = holiday

    print(df.columns)
    print(df)
    # df = df[np.isfinite(df['HLX_B1_Chiller_BTU_METER_Cooling_Capacity']) & (df['HLX_B1_Chiller_BTU_METER_Cooling_Capacity'] != 0)]
    print('[INFO] Training Cooling Load model ...')
    cl_metrics = train_cooling_load(df, model_path=opts["cl_model_path"])
    print('[RESULT] Cooling Load: rows={rows} R2={r2:.3f} RMSE={rmse:.3f}'.format(
        rows=cl_metrics['rows'], r2=cl_metrics['r2'], rmse=cl_metrics['rmse']))
    print('[INFO] Saved → {p}'.format(p=cl_metrics['model_path']))

    print('[INFO] Training KWH model ...')
    kwh_metrics = train_kwh(df, model_path=opts["kwh_model_path"], use_exact_kwh=opts["use_exact_kwh"])

    print("[DEBUG] kwh_metrics keys:", list(kwh_metrics.keys()))
    print("[DEBUG] kwh_metrics:", kwh_metrics)

    print('[RESULT] KWH ({t}): rows={rows} R2={r2:.3f} RMSE={rmse:.3f}'.format(
        t=kwh_metrics['target'], rows=kwh_metrics['rows'], r2=kwh_metrics['r2'], rmse=kwh_metrics['rmse']))
    print('[INFO] Saved → {p}'.format(p=kwh_metrics['model_path']))

if __name__ == '__main__':
    main()
