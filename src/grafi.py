import matplotlib.pyplot as plt
import pathlib
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
    ax_arr[0].bar(lbl, lethe_cog_status, color = MAT_BLUE)
    ax_arr[0].set_title("Lethe")
    ax_arr[0].set_title("LETHE")
    ax_arr[0].set_ylabel("Count")

    # NACC
    ax_arr[1].bar(lbl, nacc_cog_status, color = MAT_ORANGE)
    ax_arr[1].set_title("NACC")

    # OVERALL
    ax_arr[2].bar(lbl, all_cog_status, color = MAT_SAGE)
    ax_arr[2].set_title("Overall")

    plt.tight_layout()
    fig.suptitle("Class balance (target)", fontsize=12)
    fig.savefig(plot_dir / "01_class_balance.png", dpi=300)

