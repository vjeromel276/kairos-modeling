#!/usr/bin/env python3
"""
Fills in actual forward returns (1d, 5d, 21d) in live_predictions,
and logs scoring metrics into live_scores for each model/horizon/date group.
"""

import duckdb
from datetime import datetime, timedelta
import numpy as np

con = duckdb.connect("data/kairos.duckdb")
now = datetime.utcnow()

# Ensure score table exists
con.execute("""
CREATE TABLE IF NOT EXISTS live_scores (
    model TEXT,
    prediction_date DATE,
    horizon TEXT,
    n_predictions INT,
    n_actuals INT,
    directional_accuracy DOUBLE,
    sharpe DOUBLE,
    eval_ts TIMESTAMP
)
""")

# Horizon config
horizons = {
    "ret_1d_f_actual": 1,
    "ret_5d_f_actual": 7,
    "ret_21d_f_actual": 30
}

filled = 0
for col, delta_days in horizons.items():
    future_offset = f"DATE_ADD('day', {delta_days}, date)"
    base_return_expr = f"""
        log(f.closeadj / t.closeadj)
        AS {col}
    """

    # Fill in actuals where they are null and future is available
    con.execute(f"""
        UPDATE live_predictions
        SET {col} = (
            SELECT log(f.closeadj / t.closeadj)
            FROM sep_base t
            JOIN sep_base f ON f.ticker = t.ticker
            WHERE t.ticker = live_predictions.ticker
              AND t.date = live_predictions.date
              AND f.date = DATE_ADD('day', {delta_days}, live_predictions.date)
        )
        WHERE {col} IS NULL
          AND DATE_ADD('day', {delta_days}, date) <= CURRENT_DATE
    """)
    print(f"✅ Filled available {col} actuals.")

# Score each model+horizon group
for horizon in horizons.keys():
    result = con.execute(f"""
        SELECT
            model,
            date AS prediction_date,
            COUNT(*) AS n_predictions,
            COUNT({horizon}) AS n_actuals,
            AVG(SIGN({horizon}) = SIGN(CASE
                WHEN '{horizon}' = 'ret_1d_f_actual' THEN ret_1d_f_pred
                WHEN '{horizon}' = 'ret_5d_f_actual' THEN ret_5d_f_pred
                WHEN '{horizon}' = 'ret_21d_f_actual' THEN ret_21d_f_pred
            END)) AS directional_accuracy,
            AVG(CASE
                WHEN '{horizon}' = 'ret_1d_f_actual' THEN ret_1d_f_pred * {horizon}
                WHEN '{horizon}' = 'ret_5d_f_actual' THEN ret_5d_f_pred * {horizon}
                WHEN '{horizon}' = 'ret_21d_f_actual' THEN ret_21d_f_pred * {horizon}
            END) / STDDEV_SAMP({horizon}) AS sharpe
        FROM live_predictions
        WHERE {horizon} IS NOT NULL
        GROUP BY model, date
    """).fetchall()

    for row in result:
        model, pred_date, n_pred, n_actual, acc, sharpe = row
        con.execute("""
            INSERT INTO live_scores (
                model, prediction_date, horizon, n_predictions, n_actuals,
                directional_accuracy, sharpe, eval_ts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model, pred_date, horizon.replace("_actual", ""), n_pred, n_actual, acc, sharpe, now))

        print(f"📊 Scored {model} on {pred_date} [{horizon}] → acc={acc:.2%}, Sharpe={sharpe:.2f}")

print("✅ Evaluation complete.")
