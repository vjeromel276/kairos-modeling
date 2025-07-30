#!/usr/bin/env python3
"""
Kairos CLI: Command-line interface to run the complete prediction pipeline.
"""

import typer
from pathlib import Path
import subprocess

app = typer.Typer(help="Kairos Quant Pipeline CLI")

DATA_DIR = Path("data")
MODELS_DIR = Path("models/output")

# ----------------------------
# 🛠️ Build
# ----------------------------
@app.command()
def build_universe(year: int):
    """Filter midcap universe by start year."""
    subprocess.run(["python", "filter/filter_universe_by_start_date.py", "--year", str(year)], check=True)

@app.command()
def build_features(year: int, full: bool = False):
    """Build feature matrix for a given year."""
    cmd = ["python", "features/build_feat_matrix_complete.py", "--year", str(year)]
    if full:
        cmd.append("--full")
    subprocess.run(cmd, check=True)

@app.command()
def generate_targets(year: int):
    """Generate forward return targets."""
    subprocess.run(["python", "scripts/generate_targets.py", "--year", str(year)], check=True)

@app.command()
def refresh_feature_tables():
    """Rebuild all feat_* tables from sep_base."""
    subprocess.run(["python", "features/refresh_feature_tables.py"], check=True)

# ----------------------------
# 🤖 Train & Predict
# ----------------------------
@app.command()
def train(config: str, year: int):
    """Train model using config YAML."""
    subprocess.run(["python", "models/train_model.py", "--config", config, "--year", str(year)], check=True)

@app.command()
def predict_live(model_file: str, config: str, year: int):
    """Run live predictions using the latest frame."""
    subprocess.run([
        "python", "scripts/predict_live.py",
        "--model-file", model_file,
        "--config", config,
        "--year", str(year)
    ], check=True)

@app.command()
def simulate(pred_file: str, tag: str):
    """Run strategy simulation and log to DuckDB."""
    subprocess.run([
        "python", "scripts/simulate_strategy.py",
        "--pred-file", pred_file,
        "--tag", tag
    ], check=True)

@app.command()
def evaluate_live():
    """Evaluate live predictions with available actuals."""
    subprocess.run(["python", "scripts/evaluate_live_predictions.py"], check=True)

# ----------------------------
# 🎯 Ensemble
# ----------------------------
@app.command()
def build_ensemble():
    """Aggregate live model predictions into ensemble_predictions table."""
    subprocess.run(["python", "scripts/build_ensemble_predictions.py"], check=True)

# ----------------------------
# 🧠 PatchTST
# ----------------------------
@app.command()
def train_patchtst():
    """Train PatchTST on limited set for dev run."""
    subprocess.run(["python", "models/train_patchtst.py"], check=True)

# ----------------------------
# 🧪 Misc
# ----------------------------
@app.command()
def version():
    """Show Kairos CLI version."""
    print("Kairos CLI v1.0")


if __name__ == "__main__":
    app()
