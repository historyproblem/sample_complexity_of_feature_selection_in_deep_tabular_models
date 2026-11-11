"""Extract literal result tables without executing the source notebook."""

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
TABLES = {
    "non_iter": ("mobilenetv2", "one_pass", 7),
    "iter_00": ("mobilenetv2", "iterative", 10),
    "df_07": ("resnet50", "threshold_0.7", 10),
    "df_01": ("resnet50", "threshold_0.1", 10),
    "df_low_pruning": ("resnet50", "budget_0.05", 10),
}


def extract(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    records = []
    found = set()
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in TABLES:
                continue
            name = target.id
            call = node.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "pd" and call.func.attr == "DataFrame"
                    and len(call.args) == 1 and not call.keywords):
                raise ValueError(f"Expected literal pd.DataFrame for {name}")
            data = ast.literal_eval(call.args[0])
            model, series, count = TABLES[name]
            if name in found or any(len(v) != count for v in data.values()):
                raise ValueError(f"Unexpected/duplicate rows for {name}")
            found.add(name)
            for row_index, (run, params, accuracy) in enumerate(zip(
                data["Run"], data["model.trainable_parameters"], data["test_accuracy"]
            )):
                cycle = re.search(r"_cycle_(\d+)_recovery$", run)
                threshold = re.search(r"thr([0-9.]+)", run)
                cap = re.search(r"max_param([0-9.]+)", run)
                if not isinstance(params, int) or params <= 0 or not 0 <= accuracy <= 1:
                    raise ValueError(f"Invalid measurement in {run}")
                records.append({
                    "model": model, "series": series, "run": run,
                    "parameters": params, "test_accuracy": accuracy,
                    "source_cycle": cycle.group(1) if cycle else "",
                    "stage": "recovery" if cycle else "final",
                    "threshold": threshold.group(1) if threshold else "",
                    "parameter_cap": cap.group(1) if cap else ("0.05" if series == "budget_0.05" else ""),
                    "source_cell": cell_index, "source_row": row_index,
                })
    if found != set(TABLES):
        raise ValueError(f"Missing tables: {set(TABLES) - found}")
    for series in ("iterative", "threshold_0.7", "threshold_0.1", "budget_0.05"):
        rows = [r for r in records if r["series"] == series]
        assert sorted(int(r["source_cycle"]) for r in rows if r["source_cycle"] != "") == list(range(9))
        assert sum(r["stage"] == "final" for r in rows) == 1
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    rows = extract(args.notebook)
    destination = PAPER / "data"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "graphics_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "source": args.notebook.name,
        "sha256": hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
        "tables": {k: {"model": v[0], "series": v[1], "rows": v[2]} for k, v in TABLES.items()},
        "notes": [
            "Literal cell dictionaries only; notebook code and outputs are not executed.",
            "Run ID values are zero placeholders and are not treated as experiment identifiers.",
            "Parameters are exact supplied integers; accuracy retains the supplied precision.",
            "Source cycle indices are zero-based. Final is the tenth recorded recovery model.",
            "Other notebook series do not replace the manuscript's comparator measurements.",
        ],
    }
    (destination / "graphics_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(rows)} rows ({len(rows) - 10} requested additions, 10 existing MobileNetV2 trajectory points).")


if __name__ == "__main__":
    main()
