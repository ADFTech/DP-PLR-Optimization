
#!/usr/bin/env python3
"""
Load weather rows from Postgres into a pandas DataFrame using config.ini.

- Columns: datetime (timestamp without time zone), temp (float), humidity (float)
- Filter: rows whose time is >= NOW, using either the naive datetime column
          or the integer EpochDateTime column (whichever matches).
- Prints the DataFrame (for testing).
"""

import time
from datetime import datetime
import configparser

import pandas as pd
from sqlalchemy import create_engine, text

from modules.orm.db_wrapper import DB_wrapper  # local module from your repo

# ---- Configuration helpers --------------------------------------------------

def load_db_config(path: str = "config.ini") -> dict:
    cfg = configparser.ConfigParser()
    read_ok = cfg.read(path)
    if not read_ok:
        raise FileNotFoundError(f"Could not read config file at {path}")

    db = cfg["database-forecast"]
    # Minimal keys (engine, database, host, port) + optional username/password/SSL
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

def make_engine_with_wrapper(cfg: dict):
    """
    Use your project's DB_wrapper to produce the SQLAlchemy URL,
    then create the engine.
    """
    
    wrapper = DB_wrapper(
        engine=cfg["engine"],
        database=cfg["database"],
        host=cfg["host"],
        port=cfg["port"],
        username=cfg["username"],
        password=cfg["password"],
        ssl_mode=cfg["ssl_mode"],
        ssl_ca_cert_path=cfg["ssl_ca_cert_path"],
        ssl_cert_path=cfg["ssl_cert_path"],
        ssl_key_path=cfg["ssl_key_path"],
    )
    # wrapper.url -> "postgresql+psycopg2://user:pass@host:port/db?sslmode=..."
    # (built per your repo's logic)  [2](https://adftechnologiesmy-my.sharepoint.com/personal/davidglover_adftech_com_my/Documents/Microsoft%20Copilot%20Chat%20Files/db_wrapper.py)
    return create_engine(wrapper.url)

# ---- Data loading -----------------------------------------------------------

def read_weather_since_now(engine, epoch_col: str = "EpochDateTime") -> pd.DataFrame:
    """
    Read datetime, temp, humidity from historical_weather for rows at or after 'now'.
    """
    # Compute "now" values:
    now_dt = datetime.now()                 # naive UTC datetime

    # Build SQL with a case-sensitive epoch column (quoted):
    # We use OR so either column can satisfy the "now or later" requirement.
    # If you *only* want to depend on epoch or datetime, replace OR with the one you want.
    sql = text(f"""
        SELECT "DateTime", "Temperature", "RelativeHumidity"
        FROM forecasted_weather
        WHERE ("DateTime" >= :now_dt)
        ORDER BY "DateTime"
    """)

    df = pd.read_sql(
        sql,
        con=engine,
        params={"now_dt": now_dt},
    )

    return df

# ---- Main -------------------------------------------------------------------

def main():
    cfg = load_db_config("config.ini")

    # Choose one of the two engine creators:
    engine = make_engine_with_wrapper(cfg)   # uses your repo's DB_wrapper  [2](https://adftechnologiesmy-my.sharepoint.com/personal/davidglover_adftech_com_my/Documents/Microsoft%20Copilot%20Chat%20Files/db_wrapper.py)
    # engine = make_engine_direct(cfg)      # simpler, no wrapper import

    df = read_weather_since_now(engine, epoch_col="EpochDateTime")

    # Print while testing
    print("Rows at or after now():", len(df))
    if not df.empty:
        print(df.head(13).to_string(index=False))
        # Also print min/max datetime to eyeball the range:
        print("\nMin datetime:", df["DateTime"].min())
        print("Max datetime:", df["DateTime"].max())

if __name__ == "__main__":
    main()
