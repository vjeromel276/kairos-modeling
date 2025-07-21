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

4) python scripts/generate_mh_dataset.py --window 126 --n_jobs 12 #=> puts shard groups in scripts/shards

5) ❯ python scripts/train_model_shards.py --window 126 --model lgbm --save-every 20 --out-dir '/home/vjerome2/SSD_Quant/media/vjerome2/Extreme Pro/kairos_phase4/' 

5) python scripts/predict_and_rank_shards.py --window 126 --model '/home/vjerome2/SSD_Quant/media/vjerome2/Extreme Pro/kairos_phase4/mh_lgbm_126.pkl' --shard-dir '/home/vjerome2/SSD_Quant/media/vjerome2/Extreme Pro/kairos_phase4/scripts/shards'

6) python scripts/simulate_strategy_shards.py \
  --window 126 \
  --shard-dir /home/vjerome2/SSD_Quant/media/vjerome2/Extreme\ Pro/kairos_phase4/data/shards \
  --topk 50


#### Need to re-run  python scripts/predict_and_rank_shards.py --window 126 --model '/home/vjerome2/SSD_Quant/media/vjerome2/Extreme Pro/kairos_phase4/mh_lgbm_126.pkl' --shard-dir '/home/vjerome2/SSD_Quant/media/vjerome2/Extreme Pro/kairos_phase4/scripts/shards' to get the proper pred files for scoring and strategy