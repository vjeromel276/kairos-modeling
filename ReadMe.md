[python]
1) python scripts/daily_download.py --date 2025-07-15   

2) python scripts/merge_daily_download_duck.py --update-golden '/mnt/ssd_quant/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos.duckdb'  

3) python filter/filter_common_duck.py
    --db data/kairos.duckdb \
   --min-adv 2000000 \
   --min-days 252 \     
   --start-date 1998-01-01 \
    --min-price 2.00 \
    --bucket midcap_and_up \
    --output-dir feature_sets

4) python scripts/generate_mh_dataset.py --window 252 --n_jobs 12 

5) python scripts/train_mh_model.py  


6) python scripts/simulate_mh_strategy.py \
     --db data/kairos.duckdb \
     --window 252 \
     --model mh_lgbm.pkl \
     --top-k 50 \
     --hold 5

7) python scripts/walk_forward_mh.py \ ### to much memory used 
  --db data/kairos.duckdb \
  --window 252 \
  --initial-train-year 2015 \
  --n-estimators 500 \
  --learning-rate 0.05 \
  --num-leaves 31
