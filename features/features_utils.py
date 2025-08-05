import pandas as pd
import numpy as np


def apply_saff(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    🧠 Impute missing values using Smart Attenuated Forward Fill (SAFF).
    - df: must contain 'ticker', 'date', and the columns in value_cols
    - value_cols: list of column names to impute
    """
    print("🧠 Applying SAFF to impute missing fundamentals with decay...")
    # Compute per-ticker medians for fallback
    median_vals = df.groupby("ticker")[value_cols].transform("median")

    # Mark nulls
    for col in value_cols:
        df[f"_isnull_{col}"] = df[col].isna().astype(int)

    # Ensure ordering
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Forward fill with exponential decay
    for col in value_cols:
        last_valid = df.groupby("ticker")[col].ffill()
        gap = df.groupby("ticker")[f"_isnull_{col}"].cumsum()
        weight = np.exp(-0.05 * gap)  # decay constant = 0.05
        df[col] = last_valid * weight + median_vals[col] * (1 - weight)

    # Cleanup helper columns
    drop_cols = [c for c in df.columns if c.startswith("_isnull_")]
    df = df.drop(columns=drop_cols)
    return df


def expand_and_saff(
    sparse_df: pd.DataFrame,
    con,
    base_table: str,
    ticker_col: str,
    date_col: str,
    value_cols: list[str]
) -> pd.DataFrame:
    """
    1) Build full ticker×date index from base_table
    2) Left-merge sparse_df onto it
    3) Apply SAFF via apply_saff

    Returns a dense DataFrame with one row per ticker×date and imputed values.
    """
    # 1) get all distinct dates from the base table
    dates_df = con.execute(
        f"SELECT DISTINCT {date_col} FROM {base_table} ORDER BY {date_col}"
    ).fetchdf()
    dates = dates_df[date_col]

    # 2) prepare full ticker×date MultiIndex
    tickers = sparse_df[ticker_col].unique()
    full_idx = pd.MultiIndex.from_product(
        [tickers, dates], names=[ticker_col, date_col] # type: ignore
    )

    # 3) reindex sparse data onto full grid
    df_full = (
        sparse_df
        .set_index([ticker_col, date_col])
        .reindex(full_idx)
        .reset_index()
    )

    # 4) apply SAFF
    df_imputed = apply_saff(df_full, value_cols)
    return df_imputed

def winsorize(df, cols, lower_q=0.01, upper_q=0.99, by_date=False, date_col="date"):
    """
    Caps each column in `cols` at the [lower_q, upper_q] quantiles.
    If by_date=True: compute quantiles separately for each date.
    """
    if by_date:
        def _cap(group):
            lo = group[cols].quantile(lower_q)
            hi = group[cols].quantile(upper_q)
            return group[cols].clip(lo, hi, axis=1)
        capped = df.groupby(date_col, group_keys=False).apply(_cap)
        df = df.assign(**{c: capped[c] for c in cols})
    else:
        lo = df[cols].quantile(lower_q)
        hi = df[cols].quantile(upper_q)
        df[cols] = df[cols].clip(lo, hi, axis=1)
    return df
