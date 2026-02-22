import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import numpy as np

plot_dir = pathlib.Path(__file__).resolve().parent

# Shared palette
MAT_ORANGE = "#ff7f0e"  # LETHE
MAT_BLUE = "#1f77b4"  # NACC
MAT_SAGE = "#a3c9a8"  # Overall (light sage)

def target_class_balance(lethe, nacc):
    lbl = ["NCI", "MCI", "AD"]
    lethe_cog_status = lethe["cog_status"].value_counts().reindex([0, 1, 2])
    nacc_cog_status = nacc["cog_status"].value_counts().reindex([0, 1, 2])

    all_cog_status = nacc_cog_status + lethe_cog_status

    fig, ax_arr = plt.subplots(1, 3)
    fig.subplots_adjust(top=3)

    # LETHE
    ax_arr[0].bar(lbl, lethe_cog_status, color = MAT_ORANGE)
    ax_arr[0].set_title("Lethe")
    ax_arr[0].set_title("LETHE")
    ax_arr[0].set_ylabel("Count")

    # NACC
    ax_arr[1].bar(lbl, nacc_cog_status, color = MAT_BLUE)
    ax_arr[1].set_title("NACC")

    # OVERALL
    ax_arr[2].bar(lbl, all_cog_status, color = MAT_SAGE)
    ax_arr[2].set_title("Overall")

    plt.tight_layout()
    fig.suptitle("Class balance (target)", fontsize=12)
    fig.savefig(plot_dir / "01_class_balance.png", dpi=300)


def numeric_distributions(lethe, nacc):

    numeric_clinical = [
        "age",
        "educ_years",
        "mmse",
        "cdr_total",
        "ravlt_imm",
        "tmt_b",
        "sbp",
        "bmi",
        "diabetes_t2",
    ]

    fig, ax_arr = plt.subplots(3, 3, figsize=(12, 10))
    ax_arr = ax_arr.flatten()

    for i, col in enumerate(numeric_clinical):

        if col not in lethe.columns or col not in nacc.columns:
            ax_arr[i].set_title(col + " (missing)")
            ax_arr[i].axis("off")
            continue

        lethe_vals = lethe[col].dropna()
        nacc_vals  = nacc[col].dropna()

        bins = 20

        # weights → each participant contributes equally
        lethe_weights = np.ones(len(lethe_vals)) / len(lethe_vals) * 100
        nacc_weights  = np.ones(len(nacc_vals))  / len(nacc_vals)  * 100

        # LETHE (filled)
        ax_arr[i].hist(
            lethe_vals,
            bins=bins,
            weights=lethe_weights,
            alpha=0.6
        )

        # NACC (outline)
        ax_arr[i].hist(
            nacc_vals,
            bins=bins,
            weights=nacc_weights,
            histtype="step",
            linewidth=2
        )

        ax_arr[i].set_title(col)
        ax_arr[i].set_ylabel("Percentage (%)")

    plt.tight_layout()
    plt.show()