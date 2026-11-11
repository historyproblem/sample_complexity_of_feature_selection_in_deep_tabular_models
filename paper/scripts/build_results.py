"""Rebuild manuscript figures and notebook-result tables from checked-in CSVs."""

import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parents[1]
MANUSCRIPT = PAPER / "manuscript"
FIGURES = MANUSCRIPT / "figures"
COLORS = {"iterative": "#0072B2", "one_pass": "#D55E00", "threshold_0.7": "#0072B2",
          "threshold_0.1": "#009E73", "budget_0.05": "#CC79A7",
          "autopruner": "#E69F00", "depgraph": "#6B6B6B", "aig": "#009E73",
          "stochastic_depth": "#7B61A8"}


def read_csv(name):
    with (PAPER / "data" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ordered(rows):
    return sorted(rows, key=lambda r: int(r["source_cycle"]) if r["source_cycle"] else 9)


def round_to(value, digits):
    return format(Decimal(str(value)).quantize(Decimal(10) ** -digits, rounding=ROUND_HALF_UP), f".{digits}f")


def coordinates(rows, notebook=False):
    if notebook:
        return [int(r["parameters"]) / 1e6 for r in rows], [float(r["test_accuracy"]) * 100 for r in rows]
    return [float(r["parameters_m"]) for r in rows], [float(r["test_accuracy_pct"]) for r in rows]


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def axis_style(ax, xlabel="Physical parameters (M)"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test top-1 accuracy (%)")
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def trajectory(ax, rows, series, label):
    x, y = coordinates(ordered(rows), notebook=True)
    ax.plot(x, y, "o-", color=COLORS[series], label=label, markersize=3.5, linewidth=1.1)
    ax.scatter(x[-1], y[-1], marker="s", color=COLORS[series], s=30, zorder=5)


def figures(rows, reference):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    for model, filename, ylim in [
        ("mobilenetv2", "mobilenetv2_tinyimagenet200_tradeoff", (47, 74)),
        ("resnet50", "resnet50_cifar10_selected", (87, 95.3)),
    ]:
        fig, ax = plt.subplots(figsize=(8.3, 4.2))
        fig.subplots_adjust(bottom=0.30)
        existing = [r for r in reference if r["model"] == model]
        added = [r for r in rows if r["model"] == model]
        axis_style(ax, "Parameters (M; physical or active)")
        ax.set_ylim(*ylim)
        for method in ["autopruner", "depgraph", "stochastic_depth", "aig"]:
            selected = sorted((r for r in existing if r["method"] == method), key=lambda r: float(r["parameters_m"]))
            label = {"autopruner": "AutoPruner", "depgraph": "DepGraph (random)",
                     "stochastic_depth": "SD (expected active)", "aig": "AIG (mean active)"}[method]
            ax.scatter(*coordinates(selected), s=20, marker={"autopruner": "^", "depgraph": "v", "stochastic_depth": "o", "aig": "D"}[method],
                       color=COLORS[method], label=label, zorder=3)
        baseline = [r for r in existing if r["method"] == "dense"]
        ax.axhline(float(baseline[0]["test_accuracy_pct"]), color="#333333", linestyle=":", linewidth=0.8)
        ax.scatter(*coordinates(baseline), color="#222222", marker="*", s=65, label="Dense baseline", zorder=6)
        if model == "mobilenetv2":
            trajectory(ax, [r for r in added if r["series"] == "iterative"], "iterative", "Ours: iterative")
            ax.scatter(*coordinates([r for r in added if r["series"] == "one_pass"], True),
                       color=COLORS["one_pass"], marker="X", s=32, label="Ours: one pass", zorder=5)
            ax.set_xlim(0.95, 2.57)
        else:
            ours = [r for r in existing if r["method"] == "ours"]
            for prefix, label, marker, color in [("S1:", "One-shot: S1", "o", "#D55E00"),
                                                ("S2:", "One-shot: S2", "X", "#222222"),
                                                ("Mean", "Selected mean (n=2)", "D", "#D55E00")]:
                subset = [r for r in ours if r["setting"].startswith(prefix)]
                ax.scatter(*coordinates(subset), marker=marker, s=36, edgecolors=color,
                           facecolors="none" if marker != "X" else color, label=label, zorder=6)
            for series, label in [("threshold_0.7", r"Iterative: $\tau=0.7$"),
                                  ("threshold_0.1", r"Iterative: $\tau=0.1$"),
                                  ("budget_0.05", r"Iterative: $\rho=0.05$")]:
                trajectory(ax, [r for r in added if r["series"] == series], series, label)
            ax.set_xlim(1.6, 24.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False,
                  columnspacing=1.5, handletextpad=0.4)
        save(fig, filename)

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.6))
    fig.subplots_adjust(bottom=0.28, wspace=0.22)
    mobile = [r for r in rows if r["model"] == "mobilenetv2"]
    trajectory(axes[0], [r for r in mobile if r["series"] == "iterative"], "iterative", "Iterative")
    axes[0].scatter(*coordinates([r for r in mobile if r["series"] == "one_pass"], True),
                    marker="X", s=28, color=COLORS["one_pass"], label="One pass")
    axes[0].set_title("(a) MobileNetV2 / TinyImageNet-200")
    for series, label in [("threshold_0.7", r"Channels: $\tau=0.7$"), ("threshold_0.1", r"Channels: $\tau=0.1$"),
                          ("budget_0.05", r"Channels: $\rho=0.05$")]:
        trajectory(axes[1], [r for r in rows if r["series"] == series], series, label)
    axes[1].plot([18.974, 16.384, 15.266, 15.266], [93.27, 92.81, 92.96, 92.80],
                 "v--", color="#777777", markersize=4, label="Historical block pruning")
    axes[1].set_title("(b) ResNet-50 / CIFAR-10")
    for ax in axes:
        axis_style(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    save(fig, "additional_pruning_trajectories")

    fig, axes = plt.subplots(1, 3, figsize=(8.3, 2.4), sharey=True)
    studies = [
        ("Weight initialization", ["Inherited\n17.322M", "Random\n17.322M"], [93.77, 92.95]),
        ("Historical selection recipes", ["Adaptive\n12.728M", "Random\n12.984M"], [93.72, 93.36]),
        ("Gate placement", ["Internal\n12.983M", "Output\n20.189M"], [93.23, 93.16]),
    ]
    for ax, (title, labels, values) in zip(axes, studies):
        ax.scatter([0, 1], values, c=["#0072B2", "#777777"], s=35)
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(92.7, 94.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        for x, value in enumerate(values):
            ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 7), textcoords="offset points", ha="center")
    axes[0].set_ylabel("Test top-1 accuracy (%)")
    fig.tight_layout()
    save(fig, "resnet50_cifar10_ablations")


def tables(rows):
    generated = MANUSCRIPT / "generated"
    generated.mkdir(exist_ok=True)
    header = "% Generated by paper/scripts/build_results.py from graphics_results.csv.\n"
    mobile = [r for r in rows if r["model"] == "mobilenetv2"]
    one_pass = [r for r in mobile if r["series"] == "one_pass"]
    iterative = ordered([r for r in mobile if r["series"] == "iterative"])
    dense = Decimal(2480072)
    def measurements(r):
        params, acc = Decimal(r["parameters"]), Decimal(r["test_accuracy"]) * 100
        return " & ".join([round_to(params / 1000000, 3), round_to(params / dense * 100, 1), round_to(acc, 1), round_to(acc - Decimal("71.6"), 1)])
    lines = [header, r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
             r"\caption{MobileNetV2 pruning results. Physical parameter counts, share of the dense model, test accuracy and accuracy change are reported. The dense baseline has 2.480M parameters and 71.6\% accuracy. Left: successive recovery models from one iterative run. Right: seven independent one-pass settings; $\tau$ is the probability threshold and $\rho$ the configured parameter-removal cap, not the achieved parameter share. Training budgets are not matched.}",
             r"\label{tab:mbv2-tin-all}", r"\begin{minipage}[t]{0.47\textwidth}\centering",
             r"\textbf{Iterative pruning}\par\smallskip",
             r"\begin{tabular}{@{}lrrrr@{}}\toprule", r"Cycle & Params (M) & Share (\%) & Test (\%) & $\Delta$ (pp) \\\midrule"]
    for i, row in enumerate(iterative, 1):
        lines.append(f"{i}" + (" (final)" if i == 10 else "") + " & " + measurements(row) + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{minipage}\hfill", r"\begin{minipage}[t]{0.51\textwidth}\centering",
              r"\textbf{One-pass pruning}\par\smallskip", r"\begin{tabular}{@{}rrrrrr@{}}\toprule",
              r"$\tau$ & $\rho$ & Params (M) & Share (\%) & Test (\%) & $\Delta$ (pp) \\\midrule"]
    for row in one_pass:
        lines.append(f"{row['threshold']} & {row['parameter_cap']} & " + measurements(row) + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{minipage}", r"\end{table*}"]
    (generated / "mobilenet_pruning_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [header, r"\begin{table*}[t]\centering\small",
             r"\caption{All 30 iterative ResNet-50/CIFAR-10 channel-pruning models from the supplied notebook. $\tau$ denotes a probability threshold; $\rho$ denotes a parameter-removal budget without a probability threshold. Cycle labels are one-based (source indices 0--8, then final). Accuracies retain the supplied precision. Rows within each series are successive recovery models, not independent repeats.}",
             r"\label{tab:resnet-notebook-trajectories}", r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
             r"& \multicolumn{2}{c}{Threshold $\tau=0.7$} & \multicolumn{2}{c}{Threshold $\tau=0.1$} & \multicolumn{2}{c}{Budget $\rho=0.05$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Cycle & Params (M) & Test (\%) & Params (M) & Test (\%) & Params (M) & Test (\%) \\\midrule"]
    series_rows = [ordered([r for r in rows if r["series"] == series]) for series in ["threshold_0.7", "threshold_0.1", "budget_0.05"]]
    for i, triplet in enumerate(zip(*series_rows), 1):
        cells = [str(i) + (" (final)" if i == 10 else "")]
        for row in triplet:
            cells += [round_to(Decimal(row["parameters"]) / 1000000, 3), round_to(Decimal(row["test_accuracy"]) * 100, 1)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    (generated / "resnet_iterative_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finals = [series[-1] for series in series_rows]
    lines = [header, r"\newcommand{\ResNetIterativeFinalRows}{%"]
    for label, row in zip([r"$\tau=0.7$", r"$\tau=0.1$", r"$\rho=0.05$"], finals):
        params, acc = Decimal(row["parameters"]), Decimal(row["test_accuracy"]) * 100
        cells = [label, round_to(params / 1000000, 3), round_to(params / Decimal(23547338) * 100, 1),
                 round_to(acc, 1), round_to(acc - Decimal("93.93"), 2)]
        lines.append(" & ".join(cells) + r" \\")
    lines.append("}")
    (generated / "resnet_iterative_final_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Generated 4 figures, 17 MobileNetV2 rows and 30 ResNet-50 trajectory rows.")


if __name__ == "__main__":
    notebook_rows = read_csv("graphics_results.csv")
    figures(notebook_rows, read_csv("manuscript_comparators.csv"))
    tables(notebook_rows)
