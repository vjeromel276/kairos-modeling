# Kairos Quant Pipeline Makefile

# 🔁 Run full pipeline
all: predict

# 🥇 Download latest SEP data
download:
	python ingest/download_sep_daily.py

# 🧠 Feature engineering
features:
	python features/price_action_features.py
	python features/trend_features.py
	python features/volume_volatility_features.py
	python features/statistical_features.py
	python features/price_shape_features.py
	python features/fundamental_features.py
	python features/quality_features.py
	python features/ownership_features.py

# 🎯 Build multi-horizon targets
targets:
	python features/build_targets.py --year 1999

# 🧱 Rebuild complete feature matrix
matrix:
	python features/build_feat_matrix_complete.py \
	  --year 1999 \
	  --universe midcap_long_history_universe \
	  --full

# 🔮 Run live predictions with latest model
predict:
	python models/predict_live.py \
	  --model-file models/output/xgb_ret_1d_f_ret_5d_f_ret_21d_f_1999_top15_scaled.pkl \
	  --config models/config/xgb_top15_scaled.yaml \
	  --year 1999