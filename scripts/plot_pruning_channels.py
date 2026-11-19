"""Plot the two internal channel widths of the ResNet-50/CIFAR-10 recovery study.

Read a results .tar.gz, branch_state.json, or final deployment.pt. No training,
dataset access, or inference is performed. Mask lists contain REMOVED indices.
The reference topology is this study's base-width-64 ResNet-50, with 3/4/6/3 blocks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


DEPTHS = (3, 4, 6, 3)
BRANCHES = ("inherited", "scratch", "random")


def load_artifact(source, architecture, branch):
    source = Path(source)
    if source.name.endswith((".tar.gz", ".tgz")):
        suffix = f"/{architecture}/{branch}/branch_state.json"
        with tarfile.open(source) as bundle:
            matches = [m for m in bundle if m.isfile() and m.name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected one {suffix} in the archive, found {len(matches)}")
            artifact = json.load(bundle.extractfile(matches[0]))
            member = matches[0].name
    elif source.suffix == ".json":
        artifact = json.loads(source.read_text())
        member = None
    elif source.suffix in {".pt", ".pth"}:
        import torch
        artifact = torch.load(source, map_location="cpu", weights_only=True)
        member = None
    else:
        raise ValueError("Input must be a results .tar.gz, branch_state.json, or deployment.pt")
    if artifact.get("artifact_type") != "physical_ungated" or artifact.get("branch") != branch:
        raise ValueError("Expected a final physical deployment from the requested recovery branch")
    if artifact.get("status") not in {"completed", "infeasible"}:
        raise ValueError("Recovery is not complete")
    ledger = artifact["ledger"]
    if (ledger["search_epochs_consumed"], ledger["global_training_epoch"],
            artifact["final_training_epochs_executed"]) != (60, 150, 90):
        raise ValueError("Expected the completed 60-search + 90-recovery protocol")
    mask = artifact["pruning_mask"]
    digest = hashlib.sha256(json.dumps(mask, sort_keys=True).encode()).hexdigest()
    if digest != artifact["mask_hash"]:
        raise ValueError("Saved mask hash does not match the channel indices")
    parameters = artifact["final_cost"]["physical_total_parameters"]
    if architecture != f"p{parameters}":
        raise ValueError(f"Requested {architecture}, but the artifact has {parameters} parameters")
    return artifact, {"source": str(source.resolve()), "archive_member": member}


def channel_rows(mask):
    """Return all 32 internal groups, including groups absent from the sparse mask."""
    rows = []
    recognized = set()
    for stage, depth in enumerate(DEPTHS, 1):
        original = 64 * 2 ** (stage - 1)
        for block in range(depth):
            for boundary in ("mid1", "mid2"):
                key = f"backbone.layer{stage}.{block}.{boundary}_gumbel_layer"
                recognized.add(key)
                indices = mask.get(key, [])
                if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
                        or len(indices) != len(set(indices))
                        or any(i < 0 or i >= original for i in indices)):
                    raise ValueError(f"Invalid removed-channel indices in {key}")
                preserved = original - len(indices)
                if preserved < 1:
                    raise ValueError(f"All channels removed in {key}")
                rows.append({"block": f"layer{stage}.{block}", "boundary": boundary,
                             "original": original, "preserved": preserved,
                             "removed": len(indices)})
    if set(mask) - recognized:
        raise ValueError("This plot supports internal-only ResNet-50 masks; unexpected groups: "
                         + ", ".join(sorted(set(mask) - recognized)))
    return rows


def verify_checkpoint_widths(artifact, rows):
    """For a .pt input, verify physical Conv shapes and retained original indices."""
    state = artifact.get("model_state_dict")
    if state is None:
        return False
    if tuple(state["backbone.conv1.weight"].shape) != (64, 3, 3, 3):
        raise ValueError("Expected the unchanged base-width-64 CIFAR stem")
    widths = {(r["block"], r["boundary"]): r["preserved"] for r in rows}
    for row in rows:
        block, boundary = row["block"], row["boundary"]
        prefix = "backbone." + block
        w1, w2 = widths[block, "mid1"], widths[block, "mid2"]
        if (tuple(state[prefix + ".conv1.weight"].shape[:1]) != (w1,)
                or tuple(state[prefix + ".conv2.weight"].shape) != (w2, w1, 3, 3)
                or tuple(state[prefix + ".conv3.weight"].shape[:2]) != (row["original"] * 4, w2)):
            raise ValueError(f"Physical Conv shapes disagree with the mask in {block}")
        removed = set(artifact["pruning_mask"].get(prefix + f".{boundary}_gumbel_layer", []))
        expected = [i for i in range(row["original"]) if i not in removed]
        if state[prefix + f".{boundary}_active_indices"].tolist() != expected:
            raise ValueError(f"Physical retained-channel indices disagree in {block}/{boundary}")
    return True


def plot_channels(rows, artifact):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    original_total = sum(r["original"] for r in rows)
    preserved_total = sum(r["preserved"] for r in rows)
    parameters = artifact["final_cost"]["physical_total_parameters"]
    method = {"inherited": "Learned mask / inherited weights", "scratch": "Learned mask / fresh weights",
              "random": "Matched random mask / inherited weights"}[artifact["branch"]]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 7.5), sharey=True)
        green, blue = "#229844", "#3568C0"
        for ax, boundary, title in zip(axes, ("mid1", "mid2"),
                                      ("mid1: conv1 output / conv2 input", "mid2: conv2 output / conv3 input")):
            selected = [r for r in rows if r["boundary"] == boundary]
            y = list(range(len(selected)))
            kept = [r["preserved"] for r in selected]
            removed = [r["removed"] for r in selected]
            ax.barh(y, kept, height=.66, color=green)
            ax.barh(y, removed, left=kept, height=.66, color=blue)
            for i, row in enumerate(selected):
                ax.text(row["original"] + 8, i, f"{row['preserved']} / {row['original']}",
                        va="center", fontsize=8.5, color="#383838")
            ax.set_yticks(y, [r["block"] for r in selected])
            ax.set_xlim(0, 625)
            ax.set_xticks([0, 128, 256, 384, 512])
            ax.set_ylim(-.7, len(selected) - .2)
            ax.set_title(title, fontsize=11, pad=13)
            ax.set_xlabel("Number of channels")
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#E3E7EB", linewidth=.7)
            for divider in (2.5, 6.5, 12.5):
                ax.axhline(divider, color="#DBDFE4", linewidth=.8)
            ax.tick_params(axis="both", length=0, pad=6)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#AFB5BC")
        fig.suptitle("ResNet-50 on CIFAR-10: preserved internal channels", fontsize=15, y=.975)
        fig.text(.5, .923, f"{method}  |  {parameters / 1e6:.3f}M parameters  |  "
                 f"{preserved_total:,} / {original_total:,} internal channels retained "
                 f"({preserved_total / original_total:.1%})", ha="center", fontsize=10)
        fig.legend(handles=[Patch(color=green, label="Preserved"), Patch(color=blue, label="Removed")],
                   loc="upper center", bbox_to_anchor=(.5, .905), ncol=2, frameon=False)
        fig.text(.5, .025, "Total bar length = original width. Labels = preserved / original.\n"
                 "Stem: 64 channels, unchanged. Residual output widths are unchanged.",
                 ha="center", fontsize=9, color="#50565D", linespacing=1.6)
        fig.subplots_adjust(left=.10, right=.98, bottom=.14, top=.82, wspace=.20)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--architecture", default="p12373042")
    parser.add_argument("--branch", choices=BRANCHES, default="inherited")
    parser.add_argument("--output", type=Path, default=Path("figures/pruning_channels"),
                        help="Output prefix for .png, .pdf, .svg and source-data .json")
    args = parser.parse_args(argv)
    try:
        artifact, provenance = load_artifact(args.source, args.architecture, args.branch)
        rows = channel_rows(artifact["pruning_mask"])
        verified = verify_checkpoint_widths(artifact, rows)
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plot_channels(rows, artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for extension in ("png", "pdf", "svg"):
            path = Path(str(args.output) + "." + extension)
            fig.savefig(path, dpi=220, metadata={"Creator": "plot_pruning_channels.py"}
                        if extension == "pdf" else None)
            print(path)
    plt.close(fig)
    report = {**provenance, "architecture": args.architecture, "branch": args.branch,
              "mask_hash": artifact["mask_hash"], "model_state_hash": artifact["model_state_hash"],
              "physical_parameters": artifact["final_cost"]["physical_total_parameters"],
              "physical_checkpoint_widths_verified": verified,
              "total_internal_channels": sum(r["original"] for r in rows),
              "preserved_internal_channels": sum(r["preserved"] for r in rows),
              "stem_channels": 64, "stem_pruned": False, "residual_outputs_pruned": False,
              "channel_counts": rows}
    Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
