from __future__ import annotations
from pathlib import Path
from datetime import datetime
from .utils import ARTIFACTS_DIR


def generate_model_card(metrics: dict, params: dict, notes: str = "") -> Path:
    report_dir = ARTIFACTS_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "model_card.md"
    lines = []
    lines.append(f"# Model Card\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
    lines.append("## Overview\n")
    lines.append("Tabular binary classifier trained on synthetic data (sklearn make_classification).\n\n")
    lines.append("## Training configuration\n")
    for k, v in params.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## Metrics (test)\n")
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## Data\n- StandardScaler\n- 20 features default\n\n")
    lines.append("## Limitations\n- Synthetic dataset; may not reflect real-world distributions.\n\n")
    lines.append("## Ethical considerations\n- Ensure fairness and bias testing before production use.\n\n")
    if notes:
        lines.append("## Notes\n" + notes + "\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path