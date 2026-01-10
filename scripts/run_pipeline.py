#!/usr/bin/env python3
"""
run_pipeline.py - Master Pipeline Orchestrator for Kairos

Executes all pipeline phases in order:
  Phase 1: Universe & Base (after download/merge already done)
  Phase 2: Technical Features
  Phase 3: Fundamental Factors
  Phase 4: Composites (including regime detection)
  Phase 5: Feature Matrix Assembly

Usage:
    python scripts/run_pipeline.py --db data/kairos.duckdb --universe scripts/sep_dataset/feature_sets/option_b_universe.csv --date 2025-12-26
    python scripts/run_pipeline.py --phase 2 --db data/kairos.duckdb  # Run only Phase 2
    python scripts/run_pipeline.py --list  # Show all scripts in order
"""

import argparse
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# PIPELINE DEFINITION
# =============================================================================

PIPELINE = {
    1: {
        "name": "Universe & Base",
        "scripts": [
            (
                "scripts/create_option_b_universe.py",
                [
                    "--db",
                    "{db}",
                    "--min-adv",
                    "500000",
                    "--min-price",
                    "2.0",
                    "--universe-csv",
                    "{universe}",
                ],
            ),
            (
                "scripts/create_academic_base.py",
                ["--db", "{db}", "--universe", "{universe}"],
            ),
            # After each phase, add:
            # ("scripts/fix_all_date_types.py", ["--db", "{db}"]),
        ],
    },
    2: {
        "name": "Technical Features",
        "scripts": [
            ("scripts/features/price_action_features.py", ["--db", "{db}"]),
            ("scripts/features/trend_features.py", ["--db", "{db}"]),
            ("scripts/features/statistical_features.py", ["--db", "{db}"]),
            ("scripts/features/volume_volatility_features.py", ["--db", "{db}"]),
            ("scripts/features/price_shape_features.py", ["--db", "{db}"]),
            ("scripts/features/adv_features.py", ["--db", "{db}"]),
            ("scripts/features/vol_sizing_features.py", ["--db", "{db}"]),
            ("scripts/features/beta_features.py", ["--db", "{db}"]),
            ("scripts/features/generate_targets.py", ["--db", "{db}"]),
            # After each phase, add:
            # ("scripts/fix_all_date_types.py", ["--db", "{db}"]),
        ],
    },
    3: {
        "name": "Fundamental Factors",
        "scripts": [
            ("scripts/features/build_value_factors_v2.py", ["--db", "{db}"]),
            ("scripts/features/build_quality_factors_v2.py", ["--db", "{db}"]),
            ("scripts/features/build_momentum_factors_v2.py", ["--db", "{db}"]),
            ("scripts/features/build_insider_factors.py", ["--db", "{db}"]),
            ("scripts/features/institutional_factor_academic.py", ["--db", "{db}"]),
            ("scripts/features/rebuild_feat_fundamental.py", ["--db", "{db}"]),
            # After each phase, add:
            # ("scripts/fix_all_date_types.py", ["--db", "{db}"]),
        ],
    },
    4: {
        "name": "Composites (Base)",
        "scripts": [
            # Long composite
            ("scripts/features/build_composite_long_v2.py", ["--db", "{db}"]),
            # Academic composite factors (creates feat_composite_academic)
            ("scripts/features/build_academic_composite_factors.py", ["--db", "{db}"]),
            # Intermediate composites (v31, v32b are dependencies for v33)
            ("scripts/features/build_composite_v31.py", ["--db", "{db}"]),
            ("scripts/features/smooth_alpha_v31.py", ["--db", "{db}"]),
            ("scripts/features/build_composite_v32b.py", ["--db", "{db}"]),
            # After each phase, add:
            # ("scripts/fix_all_date_types.py", ["--db", "{db}"]),
        ],
    },
    5: {
        "name": "Regime & Final Composites & Matrix",
        "scripts": [
            # Regime detection (creates regime_history_academic)
            ("scripts/regime/regime_detector_academic.py", ["--db", "{db}"]),
            # Regime-aware composite (depends on regime_history_academic + v32b)
            ("scripts/features/build_composite_v33_regime.py", ["--db", "{db}"]),
            # Blended composites (depend on v33_regime, quality, value)
            ("scripts/features/build_alpha_composite_v7.py", ["--db", "{db}"]),
            ("scripts/features/build_alpha_composite_v8.py", ["--db", "{db}"]),
        ],
    },
    6: {
        "name": "ML Predictions",
        "scripts": [
            ("scripts/ml/generate_ml_predictions_v2.py", ["--db", "{db}"]),
            ("scripts/ml/generate_ml_predictions_v2_tuned.py", ["--db", "{db}"]),
        ],
    },
    7: {
        "name": "Feature Matrix Assembly",
        "scripts": [
        # Feature matrix assembly (last step)
            (
                "scripts/build_feature_matrix_v2.py",
                ["--db", "{db}", "--date", "{date}", "--universe", "{universe}"],
            ),
            # After each phase, add:
            # ("scripts/fix_all_date_types.py", ["--db", "{db}"]),
        ],
    },
}


def list_pipeline():
    """Print the complete pipeline structure."""
    print("\n" + "=" * 70)
    print("KAIROS PIPELINE STRUCTURE")
    print("=" * 70)

    for phase_num, phase in PIPELINE.items():
        print(f"\nPhase {phase_num}: {phase['name']}")
        print("-" * 50)
        for i, (script, _) in enumerate(phase["scripts"], 1):
            print(f"  {phase_num}.{i} {script}")

    print("\n" + "=" * 70)


def run_script(script_path: str, args: list, dry_run: bool = False) -> bool:
    """Run a single Python script with arguments."""
    cmd = ["python", script_path] + args
    cmd_str = " ".join(cmd)

    if dry_run:
        logger.info(f"[DRY RUN] Would run: {cmd_str}")
        return True

    logger.info(f"Running: {cmd_str}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Let output flow to console
            text=True,
            check=True,
        )
        logger.info(f"✓ Completed: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {script_path} (exit code {e.returncode})")
        return False
    except FileNotFoundError:
        logger.warning(f"⚠ Script not found, skipping: {script_path}")
        return True  # Continue pipeline even if optional script missing


def run_phase(
    phase_num: int, db: str, universe: str, date: str, dry_run: bool = False
) -> bool:
    """Run all scripts in a single phase."""
    if phase_num not in PIPELINE:
        logger.error(f"Invalid phase number: {phase_num}")
        return False

    phase = PIPELINE[phase_num]
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE {phase_num}: {phase['name']}")
    logger.info(f"{'='*60}")

    success_count = 0
    fail_count = 0

    for script, arg_template in phase["scripts"]:
        # Substitute placeholders
        args = [arg.format(db=db, universe=universe, date=date) for arg in arg_template]

        if run_script(script, args, dry_run):
            success_count += 1
        else:
            fail_count += 1
            # Continue to next script even if one fails

    logger.info(
        f"\nPhase {phase_num} complete: {success_count} succeeded, {fail_count} failed"
    )
    return fail_count == 0


def run_full_pipeline(
    db: str,
    universe: str,
    date: str,
    start_phase: int = 1,
    end_phase: int = 7,
    dry_run: bool = False,
) -> bool:
    """Run the complete pipeline or a range of phases."""

    logger.info("\n" + "=" * 70)
    logger.info("KAIROS WEEKLY PIPELINE")
    logger.info(f"Database: {db}")
    logger.info(f"Universe: {universe}")
    logger.info(f"Date: {date}")
    logger.info(f"Phases: {start_phase} to {end_phase}")
    logger.info("=" * 70)

    start_time = datetime.now()
    all_success = True

    for phase_num in range(start_phase, end_phase + 1):
        if not run_phase(phase_num, db, universe, date, dry_run):
            all_success = False
            # Continue to next phase even if current has failures

    elapsed = datetime.now() - start_time

    logger.info("\n" + "=" * 70)
    if all_success:
        logger.info(f"✓ PIPELINE COMPLETE (elapsed: {elapsed})")
    else:
        logger.info(f"⚠ PIPELINE COMPLETE WITH ERRORS (elapsed: {elapsed})")
    logger.info("=" * 70 + "\n")

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Kairos Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_pipeline.py --db /data/kairos.duckdb --universe scripts/sep_dataset/feature_sets/option_b_universe.csv --date 2025-12-26
  
  # Run only Phase 2 (Technical Features)
  python scripts/run_pipeline.py --phase 2 --db /data/kairos.duckdb
  
  # Run Phases 3-5
  python scripts/run_pipeline.py --start-phase 3 --end-phase 5 --db /data/kairos.duckdb --universe scripts/sep_dataset/feature_sets/option_b_universe.csv --date 2025-12-26
  
  # Dry run (show what would execute)
  python scripts/run_pipeline.py --dry-run --db /data/kairos.duckdb --universe scripts/sep_dataset/feature_sets/option_b_universe.csv --date 2025-12-26
  
  # List all scripts
  python scripts/run_pipeline.py --list
""",
    )

    parser.add_argument("--db", type=str, help="Path to DuckDB database")
    parser.add_argument("--universe", type=str, help="Path to universe CSV file")
    parser.add_argument("--date", type=str, help="Date for feature matrix (YYYY-MM-DD)")

    parser.add_argument("--phase", type=int, help="Run only this phase (1-7)")
    parser.add_argument(
        "--start-phase", type=int, default=1, help="Start from this phase"
    )
    parser.add_argument("--end-phase", type=int, default=7, help="End at this phase")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run without executing"
    )
    parser.add_argument("--list", action="store_true", help="List all pipeline scripts")

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_pipeline()
        return 0

    # Validate required args
    if not args.db:
        parser.error("--db is required")

    # Set defaults for optional args
    universe = args.universe or "scripts/sep_dataset/feature_sets/option_b_universe.csv"
    date = args.date or datetime.now().strftime("%Y-%m-%d")

    # Handle single phase vs range
    if args.phase:
        start_phase = end_phase = args.phase
    else:
        start_phase = args.start_phase
        end_phase = args.end_phase

    # Run pipeline
    success = run_full_pipeline(
        db=args.db,
        universe=universe,
        date=date,
        start_phase=start_phase,
        end_phase=end_phase,
        dry_run=args.dry_run,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
