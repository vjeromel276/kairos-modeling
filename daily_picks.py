import duckdb

con = duckdb.connect('data/kairos.duckdb', read_only=True)
result = con.execute("""
    SELECT 
        ml.ticker,
        t.sector,
        ROUND(ml.alpha_ml_v2_tuned_clf, 4) as alpha_score,
        ROUND(s.close, 2) as price
    FROM feat_alpha_ml_xgb_v2_tuned ml
    JOIN tickers t ON ml.ticker = t.ticker
    JOIN sep_base_academic s ON ml.ticker = s.ticker AND ml.date = s.date
    JOIN feat_adv adv ON ml.ticker = adv.ticker AND ml.date = adv.date
    WHERE ml.date = '2026-01-02'
      AND t."table" = 'SEP'
      AND adv.adv_20 > 50000000
      AND s.close > 10
    ORDER BY ml.alpha_ml_v2_tuned_clf DESC
    LIMIT 3
""").fetchdf()

print("🚀 TOP 3 PICKS FOR NEXT WEEK")
print("="*40)
for i, row in result.iterrows():
    print(f"{i+1}. ${row['ticker']} ({row['sector']}) - ${row['price']}")
print("="*40)
print("ML model  |  Not financial advice")
con.close()