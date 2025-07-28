#!/usr/bin/env python3
"""
Evaluate live_predictions using trading-calendar-aware forward returns.
Fills in ret_{1,5,21}d_f_actual and logs metrics into live_scores.
"""

import duckdb
from datetime import datetime

con = duckdb.connect("data/kairos.duckdb")
now = datetime.utcnow()

# Ensure live_scores table exists
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

# Horizon mapping
horizons = {
    "ret_1d_f_actual": 1,
    "ret_5d_f_actual": 5,
    "ret_21d_f_actual": 21
}

# Fill actuals using trading calendar with scalar-safe subquery
for col, offset in horizons.items():
    con.execute(f"""
        WITH future_dates AS (
            SELECT
                lp.ticker,
                CAST(lp.date AS DATE) AS prediction_date,
                tc_future.trading_date AS future_date
            FROM live_predictions lp
            JOIN trading_calendar tc_now
              ON CAST(lp.date AS DATE) = tc_now.trading_date
            JOIN (
                SELECT
                    trading_date,
                    ROW_NUMBER() OVER (ORDER BY trading_date) AS rn
                FROM trading_calendar
            ) tc_indexed
              ON tc_now.trading_date = tc_indexed.trading_date
            JOIN (
                SELECT
                    trading_date,
                    ROW_NUMBER() OVER (ORDER BY trading_date) AS rn
                FROM trading_calendar
            ) tc_future
              ON tc_future.rn = tc_indexed.rn + {offset}
        )
        UPDATE live_predictions
        SET {col} = (
            SELECT log(f.closeadj / t.closeadj)
            FROM sep_base t
            JOIN sep_base f ON f.ticker = t.ticker
            JOIN future_dates fd ON fd.ticker = t.ticker
            WHERE t.ticker = live_predictions.ticker
              AND CAST(t.date AS DATE) = fd.prediction_date
              AND CAST(f.date AS DATE) = fd.future_date
            LIMIT 1
        )
        WHERE {col} IS NULL
    """)
    print(f"✅ Filled trading-day-aligned {col}.")

# Score directional accuracy + Sharpe
for horizon in horizons.keys():
    result = con.execute(f"""
        SELECT
            model,
            CAST(date AS DATE) AS prediction_date,
            COUNT(*) AS n_predictions,
            COUNT({horizon}) AS n_actuals,
            AVG(CAST(SIGN({horizon}) = SIGN(CASE
                WHEN '{horizon}' = 'ret_1d_f_actual' THEN ret_1d_f_pred
                WHEN '{horizon}' = 'ret_5d_f_actual' THEN ret_5d_f_pred
                WHEN '{horizon}' = 'ret_21d_f_actual' THEN ret_21d_f_pred
            END) AS DOUBLE)) AS directional_accuracy,
            AVG(CASE
                WHEN '{horizon}' = 'ret_1d_f_actual' THEN ret_1d_f_pred * {horizon}
                WHEN '{horizon}' = 'ret_5d_f_actual' THEN ret_5d_f_pred * {horizon}
                WHEN '{horizon}' = 'ret_21d_f_actual' THEN ret_21d_f_pred * {horizon}
            END) / STDDEV_SAMP({horizon}) AS sharpe
        FROM live_predictions
        WHERE {horizon} IS NOT NULL
        GROUP BY model, CAST(date AS DATE)
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
