[python]
📦 Kairos Full Pipeline (Leak-Free)

🔁 DAILY PIPELINE
✅ Step 0: Daily Download (Optional)

a) python scripts/daily_download.py --date 2025-07-15   

b) python scripts/merge_daily_download_duck.py --update-golden '/mnt/ssd_quant/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos.duckdb'  

c) python filter/filter_common_duck.py
    --db data/kairos.duckdb \
   --min-adv 2000000 \
   --min-days 252 \     
   --start-date 1998-01-01 \
    --min-price 2.00 \
    --bucket midcap_and_up \
    --output-dir feature_sets


🟨 Source Tables in DuckDB

All scripts assume data/kairos.duckdb is preloaded with:
DuckDB Table	Description
sep_base	Raw OHLCV with adjusted close
mid_cap_2025_07_15	Filtered ticker universe (used for 3-feature pipeline)
feat_matrix	Full 75+ feature set


If you're pulling fresh pricing or fundamentals, you'd update sep_base or feat_matrix via:

python scripts/refresh_duckdb.py --pull-metrics

But for the modeling pipeline, we start at...
🧱 MODELING PIPELINE

✅ Step 1: Generate 1998–2022 training shards (for model training)

3-feature version:

python scripts/generate_mh_dataset_shards.py \
  --window 126 \
  --n_jobs 8 \
  --cutoff 2022-12-31

Full-feature version:

python scripts/generate_mh_dataset_shards.py \
  --window 126 \
  --n_jobs 8 \
  --cutoff 2022-12-31 \
  --full

Output:
mh_126_<TICKER>_X.parquet, y.parquet, meta.parquet → in scripts/shards/



✅ Step 2: Train the model on pre-2023 shards

python scripts/train_model_shards_cutoff.py \
  --window 126 \
  --model lgbm \
  --cutoff 2022-12-31 \
  --out-dir scripts/shards \
  --save-every 10

Output:
mh_lgbm_126_cutoff_final.pkl

✅ Step 3: Generate 2023+ prediction shards (for inference)

python scripts/generate_mh_dataset_shards.py \
  --window 126 \
  --n_jobs 8 \
  --cutoff 2025-12-31 \
  --full


Output:
mh_126_<TICKER>_pred.parquet (2023+ rows only)

✅ Step 4: Predict on 2023+ shards using trained model

python scripts/predict_and_rank_cutoff.py \
  --window 126 \
  --model mh_lgbm_126_cutoff_final.pkl \
  --shard-dir scripts/shards \
  --cutoff 2023-01-01


✅ Step 5: Simulate Top-K strategy on 2023+ predictions

python scripts/simulate_strategy_shards.py \
  --window 126 \
  --shard-dir scripts/shards \
  --topk 50

Output:
strategy_sim_126_top50.csv + printed metrics
