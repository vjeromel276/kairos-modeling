# Kairos Unified Pipeline Container

## Overview

This container runs the complete Kairos data pipeline (Phases 1-5):

| Phase | Name | Scripts |
|-------|------|---------|
| 1 | Universe & Base | create_option_b_universe, create_academic_base |
| 2 | Technical Features | 9 price/volume feature scripts |
| 3 | Fundamental Factors | value, quality, momentum, insider, institutional |
| 4 | Composites (Base) | long_v2, academic, v31, v32b |
| 5 | Regime & Final | regime_detector, v33_regime, v7, v8, feature_matrix |

## Build Instructions

### Step 1: Create the build context

From your kairos_phase4 project root:

```bash
mkdir -p docker/pipeline

# Copy these files into docker/pipeline/:
# - Dockerfile
# - requirements.txt  
# - run_pipeline.py

# Copy the entire scripts directory
cp -r scripts docker/pipeline/
```

### Step 2: Build the Docker image

```bash
cd docker/pipeline
docker build -t kairos-pipeline:v1 .
```

### Step 3: Verify the build

```bash
docker run kairos-pipeline:v1 run_pipeline.py --list
```

This shows all scripts that will run in order.

## Usage

### Run Full Pipeline (Phases 1-5)

```bash
docker run \
  --user $(id -u):$(id -g) \
  -v /path/to/kairos_phase4/data:/data \
  -v /path/to/kairos_phase4/scripts/sep_dataset/feature_sets:/features \
  kairos-pipeline:v1 \
  run_pipeline.py \
  --db /data/kairos.duckdb \
  --universe /features/option_b_universe.csv \
  --date 2025-12-26
```

### Run Specific Phase Only

```bash
# Run only Phase 2 (Technical Features)
docker run \
  --user $(id -u):$(id -g) \
  -v /path/to/data:/data \
  kairos-pipeline:v1 \
  run_pipeline.py \
  --phase 2 \
  --db /data/kairos.duckdb
```

### Run Phase Range

```bash
# Run Phases 3-5 only (skip universe/base rebuild)
docker run \
  --user $(id -u):$(id -g) \
  -v /path/to/data:/data \
  -v /path/to/features:/features \
  kairos-pipeline:v1 \
  run_pipeline.py \
  --start-phase 3 \
  --end-phase 5 \
  --db /data/kairos.duckdb \
  --universe /features/option_b_universe.csv \
  --date 2025-12-26
```

### Dry Run (Preview)

```bash
docker run kairos-pipeline:v1 run_pipeline.py \
  --dry-run \
  --db /data/kairos.duckdb \
  --universe /features/option_b_universe.csv \
  --date 2025-12-26
```

### Run Individual Script

```bash
# Run any script directly
docker run \
  --user $(id -u):$(id -g) \
  -v /path/to/data:/data \
  kairos-pipeline:v1 \
  scripts/features/build_alpha_composite_v8.py \
  --db /data/kairos.duckdb
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `data/` | `/data` | DuckDB database |
| `scripts/sep_dataset/feature_sets/` | `/features` | Universe CSV |

## Pipeline Output

After running, verify results:

```bash
python3 -c "
import duckdb
con = duckdb.connect('data/kairos.duckdb', read_only=True)
print('sep_base_academic rows:', con.execute('SELECT COUNT(*) FROM sep_base_academic').fetchone()[0])
print('feat_matrix rows:', con.execute('SELECT COUNT(*) FROM feat_matrix').fetchone()[0])
print('Latest date:', con.execute('SELECT MAX(date) FROM feat_matrix').fetchone()[0])
con.close()
"
```

## Troubleshooting

### Script not found

If a script is missing, the pipeline will warn and continue. Check that all scripts were copied to `docker/pipeline/scripts/`.

### Missing dependencies

Some scripts may require tables to exist. Run phases in order (1→5) for a clean build.

### Memory issues

DuckDB operations are memory-intensive. Ensure your system has sufficient RAM (recommend 16GB+).
