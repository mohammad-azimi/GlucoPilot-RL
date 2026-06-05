"""Lock the validated DQN checkpoint before opening the final suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy a chosen validation checkpoint to a locked model file.")
    parser.add_argument(
        "--source",
        default=r"models\dqn_discrete_shield_selection_51k\dqn_discrete_shield_selection_51k_step-010240.zip",
        help="Source checkpoint selected from validation analysis.",
    )
    parser.add_argument(
        "--locked-name",
        default="dqn_discrete_shield_locked_010240",
        help="Output model stem under models/. The .zip suffix is added automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = ROOT / args.source
    if not source.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {source}\n"
            "Run Phase 8 DQN training first, or pass --source to the checkpoint you want to lock."
        )

    locked_model = ROOT / "models" / f"{args.locked_name}.zip"
    locked_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, locked_model)

    report_dir = ROOT / "outputs" / "final_locked_model"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "locked_model_report.txt"
    report_path.write_text(
        "\n".join(
            [
                "Locked shielded discrete DQN model",
                f"source_checkpoint: {source}",
                f"locked_model: {locked_model}",
                "validation_basis: step 10240",
                "validation_mean_tir_pct: 79.7917",
                "validation_tir_delta_pct_points: +4.5313",
                "validation_mean_risk: 3.7104",
                "validation_mean_risk_delta: -1.2952",
                "validation_max_below_pct: 5.0000",
                "validation_max_very_low_pct: 0.0000",
                "final_held_out_suite_used_before_lock: no",
                "research_simulation_only: not for real medical decisions",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("DQN checkpoint locked for final evaluation.")
    print(f"Source checkpoint: {source}")
    print(f"Locked model: {locked_model}")
    print(f"Lock report saved to: {report_path}")
    print("Next, run evaluate_locked_dqn_final.py exactly once for the final held-out comparison.")


if __name__ == "__main__":
    main()
