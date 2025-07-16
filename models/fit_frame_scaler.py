import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def main():
    X_path = "data/model_ready/X_frames.npz"
    scaler_path = "data/model_ready/frame_scaler.joblib"
    X_scaled_path = "data/model_ready/X_frames_scaled.npz"

    print("📦 Loading X_frames...")
    X = np.load(X_path)["arr_0"]
    n_samples, n_timesteps, n_features = X.shape
    print(f"✅ Loaded shape: {X.shape}")

    # Flatten for scaler
    X_flat = X.reshape(n_samples, -1)

    print("🧪 Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)

    print("💾 Saving scaler...")
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # Reshape and save scaled X
    X_scaled = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
    np.savez_compressed(X_scaled_path, X_scaled)
    print(f"✅ Saved scaled X to {X_scaled_path}")
    print(f"📐 New shape: {X_scaled.shape}")

if __name__ == "__main__":
    main()
