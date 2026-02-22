import matplotlib.pyplot as plt
import pathlib
import numpy as np

plot_dir = pathlib.Path(__file__).resolve().parent

# Shared palette
MAT_ORANGE = "#ff7f0e"  # LETHE
MAT_BLUE = "#1f77b4"  # NACC
MAT_SAGE = "#a3c9a8"  # Overall (light sage)
TARGET_COL = "cog_status"

# -----------------------------------------------------------------------------
# 1) CLASS BALANCE (target)
# -----------------------------------------------------------------------------

def graf(dfL, dfN):

    label_map = {0: "NCI", 2: "MCI", 3: "AD"}

    counts_L = dfL[TARGET_COL].value_counts().reindex([0, 1, 2], fill_value=0)
    counts_N = dfN[TARGET_COL].value_counts().reindex([0, 1, 2], fill_value=0)
    counts_overall = counts_L + counts_N

    fig_cb, axes_cb = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    x = np.arange(3)
    labels = label_map.values()

    # Overall = sage
    ax = axes_cb[0]
    ax.bar(x, counts_overall.values, color=MAT_SAGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Overall")
    ax.set_ylabel("Count")

    # LETHE = orange
    ax = axes_cb[1]
    ax.bar(x, counts_L.values, color=MAT_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("LETHE")
    ax.set_ylabel("Count")

    # NACC = blue
    ax = axes_cb[2]
    ax.bar(x, counts_N.values, color=MAT_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("NACC")
    ax.set_ylabel("Count")

    fig_cb.suptitle("Class balance (target)", fontsize=16)
    fig_cb.savefig(plot_dir / "01_class_balance.png", dpi=300)
    plt.close(fig_cb)
    print(f"[PLOTS] Saved class balance → {plot_dir / '01_class_balance.png'}")
