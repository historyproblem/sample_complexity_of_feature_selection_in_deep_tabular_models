"""Audit beta-run archives and build descriptive channel-pruning figures.

Local: python paper/scripts/build_beta_ablation.py
Server: .venv/bin/python paper/scripts/build_beta_ablation.py
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
COLORS = {0.0: "#0072B2", 0.5: "#E69F00", 1.0: "#009E73", 2.0: "#CC79A7"}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {k: v for name, child in value.items()
                for k, v in flatten(child, f"{prefix}.{name}" if prefix else name).items()}
    if isinstance(value, list):
        return {k: v for i, child in enumerate(value)
                for k, v in flatten(child, f"{prefix}.{i}").items()}
    return {prefix: value}


def entropy(prob):
    p = np.asarray(prob, dtype=float)
    h = np.zeros_like(p)
    mid = (p > 0) & (p < 1)
    h[mid] = -p[mid] * np.log(p[mid]) - (1 - p[mid]) * np.log1p(-p[mid])
    return h


def group_order(name):
    index = int(re.search(r"features\.(\d+)\.", name).group(1))
    return index, 0 if "mid_gumbel" in name else 1


def group_label(name):
    index, kind = group_order(name)
    return f"{index}:" + ("hidden" if kind == 0 else "output")


def load_run(path):
    with zipfile.ZipFile(path) as archive:
        def read(suffix, stage=None):
            matches = [n for n in archive.namelist()
                       if n.endswith(suffix) and (stage is None or stage in n)]
            if len(matches) != 1:
                raise ValueError(f"Expected one {stage}/{suffix}: {matches}")
            return archive.read(matches[0])
        cfg = yaml.safe_load(read("config_resolved.yaml", "/cycle_0/"))
        recovery_cfg = yaml.safe_load(read("config_resolved.yaml", "/final/"))
        cyclic = json.loads(read("cyclic_summary.json"))
        summary = json.loads(read("summary.json", "/final/"))
        search = json.loads(read("summary.json", "/cycle_0/"))
        history = list(csv.DictReader(io.StringIO(read("history.csv", "/cycle_0/").decode())))
        gates = [json.loads(line) for line in
                 gzip.decompress(read("gate_history.jsonl.gz", "/cycle_0/")).splitlines()]
    beta = float(cfg["model"]["entropy_regularization_coef"])
    if beta not in COLORS:
        raise ValueError(f"Unexpected beta {beta}")
    epochs = sorted({g["epoch"] for g in gates})
    expected_epochs = cfg["training_arguments"]["num_epochs"]
    assert epochs == list(range(1, expected_epochs + 1))
    assert len(history) == expected_epochs
    assert cyclic["num_cycles_completed"] == 1
    mask = cyclic["final_disabled_channels"]
    assert mask == recovery_cfg["channel_pruning"]["mask"]
    by_epoch = {}
    for g in gates:
        assert g["split"] == "valid"
        p = np.asarray(g["selection_probs"], dtype=float)
        assert len(p) == g["num_channels"] and np.all(np.isfinite(p))
        assert np.all((p >= 0) & (p <= 1))
        groups = by_epoch.setdefault(g["epoch"], {})
        assert g["layer_name"] not in groups
        groups[g["layer_name"]] = p
    widths = {name: len(p) for name, p in by_epoch[epochs[-1]].items()}
    for groups in by_epoch.values():
        assert {name: len(p) for name, p in groups.items()} == widths
    assert set(mask) <= set(widths)
    for name, ids in mask.items():
        assert len(ids) == len(set(ids)) and len(ids) < widths[name]
        assert all(0 <= i < widths[name] for i in ids)
    return dict(beta=beta, cfg=cfg, by_epoch=by_epoch, widths=widths, mask=mask,
                history=history, summary=summary, search=search, path=path,
                final_epoch=epochs[-1])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot(runs, out):
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42})
    names = sorted(runs[0]["widths"], key=group_order)

    def distribution(ax):
        for run in runs:
            p = np.concatenate(list(run["by_epoch"][run["final_epoch"]].values()))
            ax.hist(p, bins=np.linspace(0, 1, 31), density=True, histtype="step",
                    linewidth=1.5, color=COLORS[run["beta"]], label=rf"$\beta={run['beta']:g}$")
        ax.set(xlim=(0, 1), xlabel="Opening score r", ylabel="Density",
               title="Opening scores at the end of search phase")
        ax.legend(frameon=False)

    def dynamics(ax):
        for run in runs:
            epochs = sorted(run["by_epoch"])
            h = [np.mean([entropy(p).mean() for p in run["by_epoch"][e].values()]) for e in epochs]
            ax.plot(epochs, h, color=COLORS[run["beta"]], linewidth=1.6,
                    label=rf"$\beta={run['beta']:g}$")
        ax.axhline(np.log(2), color="0.5", linestyle=":", linewidth=1)
        ax.set(xlim=(1, 30), ylim=(0, 0.73), xlabel="Search epoch",
               ylabel="Mean group entropy (nats)", title="(b) Gate entropy during search")
        ax.legend(frameon=False)

    def retention(ax):
        x = np.arange(len(names))
        for i, run in enumerate(runs):
            kept = [(run["widths"][n] - len(run["mask"].get(n, []))) / run["widths"][n] * 100 for n in names]
            ax.bar(x + (i - 1.5) * 0.19, kept, width=0.18, color=COLORS[run["beta"]],
                   label=rf"$\beta={run['beta']:g}$")
        ax.set_xticks(x, [group_label(n) for n in names], rotation=60, ha="right")
        ax.set(ylim=(0, 105), ylabel="Retained channels (%)",
               xlabel="MobileNetV2 feature index : gate type",
               title="(b) Physical channel retention after budgeted selection")
        ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))

    def mean_scores(ax):
        for run in runs:
            epochs = sorted(run["by_epoch"])
            means = [np.mean([p.mean() for p in run["by_epoch"][e].values()]) for e in epochs]
            ax.plot(epochs, means, color=COLORS[run["beta"]], linewidth=1.6,
                    label=rf"$\beta={run['beta']:g}$")
        ax.set(xlim=(1, 30), ylim=(0, 1), xlabel="Search epoch",
               ylabel="Mean opening score (group-normalized)",
               title="Mean opening score during search")
        ax.legend(frameon=False)

    for name, draw, size in [("opening_scores", distribution, (6, 3.6)),
                              ("entropy_dynamics", dynamics, (6, 3.6)),
                              ("mean_opening_score", mean_scores, (6, 3.6)),
                              ("channel_retention", retention, (12, 4.5))]:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        save(fig, out, name)
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    distribution(ax)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, out, "beta_ablation")
    shutil.copyfile(out / "opening_scores.pdf",
                    ROOT / "paper/manuscript/figures/beta_opening_scores.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=ROOT / "ablation_data")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/ablation_beta")
    args = parser.parse_args()
    runs = sorted([load_run(p) for p in args.archives.glob("*.zip")], key=lambda r: r["beta"])
    assert [r["beta"] for r in runs] == [0, 0.5, 1, 2]
    baseline = runs[0]
    ignored = {"model.entropy_regularization_coef", "run_history.run_name",
               "run_history.root_dir", "mlflow.run_name"}
    f0 = flatten(baseline["cfg"])
    for run in runs:
        f = flatten(run["cfg"])
        changes = {k for k in f0.keys() | f.keys() if f0.get(k) != f.get(k)}
        assert changes <= ignored, changes - ignored
        assert run["widths"] == baseline["widths"]
        for key in ["lambda_coef", "reference_acc"]:
            assert [row[key] for row in run["history"]] == [row[key] for row in baseline["history"]]
    args.output.mkdir(parents=True, exist_ok=True)
    scores, groups, dynamics, summaries, mean_scores = [], [], [], [], []
    for run in runs:
        beta = run["beta"]
        final = run["by_epoch"][run["final_epoch"]]
        for name in sorted(final, key=group_order):
            removed = set(run["mask"].get(name, []))
            for i, prob in enumerate(final[name]):
                scores.append(dict(beta=beta, epoch=run["final_epoch"], group=name,
                                   channel=i, opening_score=prob, removed=int(i in removed)))
            n = len(final[name])
            groups.append(dict(beta=beta, group=name, label=group_label(name), original=n,
                               removed=len(removed), retained=n-len(removed), retained_fraction=1-len(removed)/n))
        for epoch, epoch_groups in sorted(run["by_epoch"].items()):
            row = run["history"][epoch-1]
            mean_scores.append(dict(beta=beta, epoch=epoch,
                group_mean_opening_score=float(np.mean([p.mean() for p in epoch_groups.values()])),
                channel_mean_opening_score=float(np.concatenate(list(epoch_groups.values())).mean())))
            dynamics.append(dict(beta=beta, epoch=epoch,
                group_mean_entropy=float(np.mean([entropy(p).mean() for p in epoch_groups.values()])),
                channel_mean_entropy=float(entropy(np.concatenate(list(epoch_groups.values()))).mean()),
                lambda_coef=row["lambda_coef"], reference_accuracy=row["reference_acc"]))
        summaries.append(dict(beta=beta, source=run["path"].name,
            sha256=hashlib.sha256(run["path"].read_bytes()).hexdigest(),
            groups=len(final), original_channels=sum(run["widths"].values()),
            removed_channels=sum(len(ids) for ids in run["mask"].values()),
            final_group_mean_entropy=dynamics[-1]["group_mean_entropy"],
            recovery_test_accuracy=run["summary"]["test"]["test_accuracy"],
            seed=run["search"]["identity"].get("seed"),
            provenance=run["search"]["provenance"]))
    for name, rows in [("opening_scores", scores), ("channel_retention", groups),
                       ("entropy_dynamics", dynamics), ("mean_opening_score", mean_scores)]:
        write_csv(args.output / f"{name}.csv", rows)
    (args.output / "provenance.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    plot(runs, args.output)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
