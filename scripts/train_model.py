# scripts/train_model.py

import argparse
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Dropout  # type: ignore

from scripts.register_model_features import register_features

FEATURE_NAMES = [
    f"log_return_{i:03d}" for i in range(252)
] + [
    f"volume_z_{i:03d}" for i in range(252)
] + [
    f"price_norm_{i:03d}" for i in range(252)
]

def load_data(X_path, y_path):
    print("📦 Loading data...")
    X = np.load(X_path)["arr_0"].astype(np.float32)
    y = np.load(y_path)["arr_0"].astype(np.float32)
    return X, y

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    directional = np.mean((y_pred > 0) == (y_test > 0))
    return mse, r2, directional

def train_ridge(X, y):
    X_flat = X.reshape(len(X), -1)
    nan_total = np.isnan(X_flat).sum()
    print(f"🔍 Found {nan_total:,} NaNs across {X_flat.shape[1]} features")

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_flat)

    model = Ridge(alpha=1.0)
    model.fit(X_imputed, y)
    return model, imputer

def train_lgbm(X, y):
    print("🚀 Training LightGBM model...")
    X_flat = X.reshape(len(X), -1)
    model = LGBMRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-1)
    model.fit(X_flat, y)
    return model

def build_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_cnn(X, y):
    print("🚀 Training 1D-CNN model...")
    model = build_cnn_model((252, 3))
    model.fit(X, y, epochs=10, batch_size=256, validation_split=0.1, verbose=1)
    return model

def build_mlp_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_mlp(X, y):
    print("🚀 Training MLP model...")
    X_flat = X.reshape(len(X), -1)
    model = build_mlp_model(X_flat.shape[1])
    model.fit(X_flat, y, epochs=10, batch_size=256, validation_split=0.1, verbose=1)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X-path", required=True)
    parser.add_argument("--y-path", required=True)
    parser.add_argument("--model", choices=["ridge", "lgbm", "cnn", "mlp"], required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--db", default="data/kairos.duckdb", help="Path to DuckDB registry")
    args = parser.parse_args()

    X, y = load_data(args.X_path, args.y_path)
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.model == "ridge":
        model, imputer = train_ridge(X_train, y_train)
        joblib.dump((model, imputer), args.output_path)
        X_eval = imputer.transform(X_test.reshape(len(X_test), -1))
        mse, r2, directional = evaluate(model, X_eval, y_test)

    elif args.model == "lgbm":
        model = train_lgbm(X_train, y_train)
        joblib.dump(model, args.output_path)
        mse, r2, directional = evaluate(model, X_test.reshape(len(X_test), -1), y_test)

    elif args.model == "cnn":
        model = train_cnn(X_train, y_train)
        model.save(args.output_path)
        mse, r2, directional = evaluate(model, X_test, y_test)

    elif args.model == "mlp":
        model = train_mlp(X_train, y_train)
        model.save(args.output_path)
        mse, r2, directional = evaluate(model, X_test.reshape(len(X_test), -1), y_test)

    model_name = Path(args.output_path).stem
    register_features(model_name, FEATURE_NAMES, db_path=args.db)

    print("📈 Evaluation Results:")
    print(f"📉 MSE: {mse:.6f}")
    print(f"📊 R²: {r2:.4f}")
    print(f"📈 Directional Accuracy: {directional:.2%}")

if __name__ == "__main__":
    main()
