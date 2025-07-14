# scripts/models/train_from_feature_matrix.py
'''
python scripts/models/train_from_feature_matrix.py \
  --matrix scripts/feature_matrices/2025-07-13_full_feature_matrix.parquet \
  --target ret_1d_f \
  --model lgbm \
  --output models/lgbm_2025-07-13_from_matrix.joblib \
  --db data/kairos.duckdb

'''
import argparse
import pandas as pd
import numpy as np
import joblib
# import duckdb
from pathlib import Path
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, GlobalMaxPooling1D  # type: ignore
from scripts.register_model_features import register_features

def load_matrix(matrix_path, target_col):
    df = pd.read_parquet(matrix_path)
    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(np.float32).values
    feature_cols = [c for c in df.columns if c not in ["ticker", "date", target_col]]
    X = df[feature_cols].astype(np.float32).values
    return X, y, feature_cols

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    directional = np.mean((y_pred > 0) == (y_test > 0))
    return mse, r2, directional

def build_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, help="Path to full feature matrix (.parquet)")
    parser.add_argument("--target", required=True, help="Target column to predict (e.g. ret_1d_f)")
    parser.add_argument("--model", required=True, choices=["ridge", "lgbm", "mlp"])
    parser.add_argument("--output", required=True, help="Path to save trained model")
    parser.add_argument("--db", required=True, help="Path to DuckDB for registering feature list")
    args = parser.parse_args()

    X, y, features = load_matrix(args.matrix, args.target)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.model == "ridge":
        imputer = SimpleImputer(strategy="mean")
        # Check and sanitize values
        num_infs = np.isinf(X_train).sum()
        num_nans = np.isnan(X_train).sum()

        if num_infs > 0 or num_nans > 0:
            print(f"⚠️ Found {num_infs} infs and {num_nans} NaNs — replacing with np.nan and imputing...")
            X_train = np.where(np.isinf(X_train), np.nan, X_train)

        # Impute after cleanup
        X_train = imputer.fit_transform(X_train)

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        joblib.dump((model, imputer), args.output)
        X_test = imputer.transform(X_test)
    elif args.model == "lgbm":
        model = LGBMRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-1)
        model.fit(X_train, y_train)
        joblib.dump(model, args.output)
    elif args.model == "mlp":
        model = build_mlp(X.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.1, verbose=1)
        model.save(args.output)

    mse, r2, directional = evaluate(model, X_test, y_test)
    model_name = Path(args.output).stem
    register_features(model_name, features, args.db)

    print(f"📉 MSE: {mse:.6f}")
    print(f"📊 R²: {r2:.4f}")
    print(f"📈 Directional Accuracy: {directional:.2%}")
    print(f"✅ Saved model to {args.output}")

if __name__ == "__main__":
    main()