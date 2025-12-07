import datetime
import duckdb
import pandas as pd

def log_backtest_result(
    db_path,
    strategy_name,
    run_command,
    start_date,
    end_date,
    port_total,
    port_ann,
    port_vol,
    port_sharpe,
    port_dd,
    bench_total=None,
    bench_ann=None,
    bench_vol=None,
    bench_sharpe=None,
    bench_dd=None,
    active_ann=None,
    active_vol=None,
    active_sharpe=None,
):
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS backtest_run_log (
            timestamp TIMESTAMP,
            strategy_name TEXT,
            run_command TEXT,
            start_date DATE,
            end_date DATE,
            port_total DOUBLE,
            port_ann DOUBLE,
            port_vol DOUBLE,
            port_sharpe DOUBLE,
            port_max_dd DOUBLE,
            bench_total DOUBLE,
            bench_ann DOUBLE,
            bench_vol DOUBLE,
            bench_sharpe DOUBLE,
            bench_max_dd DOUBLE,
            active_ann DOUBLE,
            active_vol DOUBLE,
            active_sharpe DOUBLE
        )
    """)

    con.execute("""
        INSERT INTO backtest_run_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now(datetime.timezone.utc),
        strategy_name,
        run_command,
        start_date,
        end_date,
        port_total,
        port_ann,
        port_vol,
        port_sharpe,
        port_dd,
        bench_total,
        bench_ann,
        bench_vol,
        bench_sharpe,
        bench_dd,
        active_ann,
        active_vol,
        active_sharpe,
    ))

    con.close()
