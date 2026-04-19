"""
Shared runtime summary logging for inference scripts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_SUMMARY_PATH = BASE_DIR / "outputs" / "runtime_summary.csv"
FIELDNAMES = (
    "method_name",
    "total_simulator_calls",
    "wall_clock_seconds",
    "posterior_sample_size",
    "acceptance_rate",
    "ess",
)


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(key): _to_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(subvalue) for subvalue in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _serialize_optional(value: Any) -> str:
    jsonable_value = _to_jsonable(value)
    if jsonable_value is None:
        return ""
    if isinstance(jsonable_value, (dict, list)):
        return json.dumps(jsonable_value, sort_keys=True)
    return str(jsonable_value)


def write_runtime_summary(*,
                          method_name: str,
                          total_simulator_calls: int,
                          wall_clock_seconds: float,
                          posterior_sample_size: int,
                          acceptance_rate: float | None = None,
                          ess: Any = None,
                          output_path: Path = RUNTIME_SUMMARY_PATH) -> None:
    """
    Upsert one method-specific runtime row into the shared CSV summary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "method_name": method_name,
        "total_simulator_calls": str(int(total_simulator_calls)),
        "wall_clock_seconds": f"{float(wall_clock_seconds):.6f}",
        "posterior_sample_size": str(int(posterior_sample_size)),
        "acceptance_rate": _serialize_optional(acceptance_rate),
        "ess": _serialize_optional(ess),
    }

    existing_rows: list[dict[str, str]] = []
    if output_path.exists():
        try:
            with output_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    existing_rows.append({field: row.get(field, "") for field in FIELDNAMES})
        except PermissionError:
            print(
                f"[runtime_summary] Skipping update for {method_name}: "
                f"could not read locked file {output_path}."
            )
            return

    updated = False
    for row in existing_rows:
        if row["method_name"] == method_name:
            row.update(new_row)
            updated = True
            break

    if not updated:
        existing_rows.append(new_row)

    existing_rows.sort(key=lambda row: row["method_name"])

    try:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(existing_rows)
    except PermissionError:
        print(
            f"[runtime_summary] Skipping update for {method_name}: "
            f"could not write locked file {output_path}."
        )
