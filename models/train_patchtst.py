import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import joblib
from scripts.register_model_features import register_features

def patchify(X, patch_size):
    n, t, f = X.shape
    assert t % patch_size == 0, "Sequence length must be divisible by patch_size"
    num_patches = t // patch_size
    return X.reshape(n, num_patches, patch_size * f)

def build_patchtst(input_shape, d_model=64, n_heads=4, ff_dim=128, dropout=0.1):
    inputs = layers.Input(shape=input_shape)

    x = layers.LayerNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([x, inputs])  # Residual 1

    y = layers.LayerNormalization()(x)
    y = layers.Dense(ff_dim, activation="relu")(y)
    y = layers.Dense(x.shape[-1])(y)  # 💡 Project to match x
    x = layers.Add()([x, y])  # Residual 2

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
    print("✅ PatchTST model built successfully")
    print(f"📐 Input shape: {input_shape}, Output shape: {outputs.shape}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", default="data/model_ready/X_frames_scaled.npz")
    parser.add_argument("--y", default="data/model_ready/y_targets.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--db", default="data/kairos.duckdb")
    parser.add_argument("--patch-size", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    print("📦 Loading data...")
    X = np.load(args.X)["arr_0"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.load(args.y)["arr_0"]
    if np.isnan(y).any():
        raise ValueError("❌ y contains NaNs — check preprocessing pipeline.")
    n = len(X)
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"✅ Loaded shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

    print("🧩 Creating patches...")
    X_train_patch = patchify(X_train, args.patch_size)
    X_test_patch = patchify(X_test, args.patch_size)

    print(f"📐 Patch shape: {X_train_patch.shape[1:]}")

    model = build_patchtst(X_train_patch.shape[1:])
    model.fit(X_train_patch, y_train,
              validation_data=(X_test_patch, y_test),
              epochs=args.epochs,
              batch_size=args.batch_size,
              verbose=1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

    print("💾 Saving model...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)

    print("📊 Evaluating...")
    y_pred = model.predict(X_test_patch).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    directional = np.mean((y_pred > 0) == (y_test > 0))

    print(f"📉 MSE: {mse:.6f}")
    print(f"📊 R²: {r2:.4f}")
    print(f"📈 Directional Accuracy: {directional:.2%}")

    model_name = Path(args.output).stem
    features = ["log_return", "volume_z", "price_norm"]
    register_features(model_name, features, args.db)

if __name__ == "__main__":
    main()
