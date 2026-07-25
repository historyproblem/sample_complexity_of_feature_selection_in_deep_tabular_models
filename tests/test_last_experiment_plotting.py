import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from net_complexity.studies.last_experiment import (
    _apply_automatic_run_labels,
    extract_run_label_values,
    gradient_norm_catalog,
    make_run_label,
    plot_channel_counts,
    plot_gradient_norms,
    plot_metric,
    plot_metric_by_epoch,
    normalize_history_columns,
)


@pytest.fixture
def history_df():
    rows = []
    for run_name, label, offset in (("run_1", "lambda=0.1", 0.0), ("run_2", "lambda=1", 1.0)):
        for epoch in (1, 2):
            rows.append(
                {
                    "run_name": run_name,
                    "run_label": label,
                    "config_lambda_coef": offset + 0.1,
                    "epoch": epoch,
                    "valid_accuracy": epoch + offset,
                    "valid_real_active_channels": 8 - epoch,
                    "valid_real_zero_channels": epoch,
                    "grad_norm_ce_total_mean": epoch + offset,
                    "grad_norm_regularization_total_mean": 2 * epoch + offset,
                    "grad_norm_total_total_mean": 3 * epoch + offset,
                }
            )
    return pd.DataFrame(rows)


def test_plot_metric_is_the_canonical_plotter(history_df):
    ax = plot_metric(history_df, "valid_accuracy", show=False)

    assert len(ax.lines) == 2
    assert ax.get_ylabel() == "valid_accuracy"


def test_interactive_metric_uses_clickable_external_legend(history_df):
    figure = plot_metric(
        history_df,
        "valid_accuracy",
        interactive=True,
        show=False,
    )

    assert len(figure.data) == 2
    assert figure.layout.legend.x > 1
    assert figure.layout.legend.groupclick == "togglegroup"


def test_legacy_plot_metric_by_epoch_delegates_to_canonical_plotter(history_df):
    ax = plot_metric_by_epoch(
        history_df,
        "valid_accuracy",
        average_by_lambda=True,
    )

    assert len(ax.lines) == 2


def test_plot_gradient_norms_builds_standard_component_plots(history_df):
    axes = plot_gradient_norms(history_df, show=False)

    assert set(axes) == {
        "grad_norm_ce_total_mean",
        "grad_norm_regularization_total_mean",
        "grad_norm_total_total_mean",
    }
    assert all(len(ax.lines) == 2 for ax in axes.values())


def test_gradient_norm_catalog_parses_metric_columns(history_df):
    catalog = gradient_norm_catalog(history_df)

    assert set(catalog["component"]) == {"ce", "regularization", "total"}
    assert set(catalog["parameter_group"]) == {"total"}
    assert set(catalog["statistic"]) == {"mean"}


def test_plot_channel_counts_has_count_and_percentage_axes(history_df):
    ax = plot_channel_counts(history_df, show=False)

    assert len(ax.lines) == 4
    assert ax.get_ylabel() == "channels"
    assert ax.child_axes[0].get_ylabel() == "channels, % of total"


def test_interactive_channel_plot_groups_open_and_closed_by_run(history_df):
    figure = plot_channel_counts(history_df, interactive=True, show=False)

    visible_legend_traces = [trace for trace in figure.data if trace.showlegend]
    assert len(visible_legend_traces) == 2
    assert figure.layout.legend.x > 1
    assert figure.layout.legend.groupclick == "togglegroup"


def test_plot_channel_counts_postprocesses_zero_probabilities():
    history = normalize_history_columns(
        pd.DataFrame(
            {
                "epoch": [1, 2],
                "valid_layer_1_zero_prob": [0.2, 0.8],
                "valid_layer_2_zero_prob": [0.9, 0.1],
            }
        )
    )

    ax = plot_channel_counts(history, show=False)

    assert history["open_channels"].tolist() == [1, 1]
    assert history["zero_channels"].tolist() == [1, 1]
    assert len(ax.lines) == 2


def test_plot_channel_counts_postprocesses_aig_gate_probabilities():
    history = normalize_history_columns(
        pd.DataFrame(
            {
                "epoch": [1, 2],
                "valid_g_prob_block_1": [0.2, 0.8],
                "valid_g_prob_block_2": [0.9, 0.1],
            }
        )
    )

    ax = plot_channel_counts(history, show=False)

    assert history["valid_active_blocks"].tolist() == [1, 1]
    assert history["valid_inactive_blocks"].tolist() == [1, 1]
    assert ax.get_ylabel() == "blocks"


def test_aig_block_counts_take_priority_over_channel_aliases():
    history = normalize_history_columns(
        pd.DataFrame(
            {
                "epoch": [1],
                "valid_g_prob_block_1": [0.8],
                "valid_real_active_channels": [10],
                "valid_real_zero_channels": [20],
            }
        )
    )

    ax = plot_channel_counts(history, show=False)

    assert ax.get_ylabel() == "blocks"


def test_config_driven_run_label_uses_aliases_and_compact_scientific_notation(history_df):
    config = {
        "model": {"lambda_coef": 1e-6},
        "training_arguments": {"adaptive_lambda": {"update_every_epochs": 2}},
        "reporting": {
            "run_label_fields": {
                "lambda_init": "model.lambda_coef",
                "step": "training_arguments.adaptive_lambda.update_every_epochs",
            }
        },
    }

    values = extract_run_label_values(config)
    label = make_run_label(history_df, "long-run-name", 1e-6, values)

    assert label == "lambda_init_1e-6_step_2"


def test_automatic_labels_use_only_config_fields_that_vary(history_df):
    summary = pd.DataFrame(
        {
            "run_name": ["run_1", "run_2"],
            "model.lambda_coef": [1e-4, 1.0],
            "model.entropy_regularization": ["disabled", "plus_negative_entropy"],
            "model.entropy_regularization_coef": [0.0, 0.3],
            "model.backbone.resnet_block.gate_regularization": [
                "l1_probability",
                "l1_probability",
            ],
        }
    )

    labeled_summary, labeled_history = _apply_automatic_run_labels(summary, history_df)

    assert labeled_summary["run_label"].tolist() == [
        "lambda_init_1e-4_entropy_disabled_entropy_coef_0",
        "lambda_init_1_entropy_plus_negative_entropy_entropy_coef_0.3",
    ]
    assert set(labeled_history["run_label"]) == set(labeled_summary["run_label"])
