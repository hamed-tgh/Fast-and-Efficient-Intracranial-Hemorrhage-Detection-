# import matplotlib.pyplot as plt

# classes = [
#     "Epidural",
#     "Intraparenchymal",
#     "Intraventricular",
#     "Subarachnoid",
#     "Subdural",
#     "Any"
# ]

# percentages = [0.42, 4.80, 3.48, 4.74, 6.27, 14.34]
# #Class Distribution (%) in RSNA Intracranial Hemorrhage Dataset
# #Hemorrhage Class

# plt.figure(figsize=(8, 5))
# plt.bar(classes, percentages)
# plt.xlabel("")
# plt.ylabel("Percentage of Positive Samples (%)")
# plt.title("")
# plt.xticks(rotation=30, ha="right")
# plt.tight_layout()

# plt.savefig("Pictures//class_distribution_percentage.png", dpi=300, bbox_inches="tight")

# plt.show()


# counters = [3145,36118,26205,35675,47166,107933] 

# plt.figure(figsize=(8, 5))
# plt.bar(classes, counters)
# plt.xlabel("")
# plt.ylabel("Number of Positive Samples")
# plt.title("")
# plt.xticks(rotation=30, ha="right")
# plt.tight_layout()

# plt.savefig("Pictures//class_distribution_Numbers.png", dpi=300, bbox_inches="tight")

# plt.show()




import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot():
    
    # ---------- Publication-style defaults ----------
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
    })

    df = pd.read_csv("repvit_rsna_checkpoints_eval.csv")

    x_labels = df["checkpoint"].astype(str).values
    y = df["auroc"].values
    x = np.arange(len(x_labels))  
    fig, ax = plt.subplots(figsize=(9.0, 3.5))  
    ax.plot(x, y, marker="o", markersize=2.8, linewidth=1.0)

    ax.set_title("AUROC Across Checkpoints")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("AUROC")


    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, ha="center")

    best_idx = np.argmax(y)
    ax.scatter(best_idx, y[best_idx], s=30, zorder=3)
    ax.annotate(
        f"Best: {y[best_idx]:.4f}",
        (best_idx, y[best_idx]),
        textcoords="offset points",
        xytext=(5, 6),
        fontsize=9
    )

    fig.tight_layout()

    fig.savefig("auroc_per_checkpoint.pdf", bbox_inches="tight")
    fig.savefig("auroc_per_checkpoint.png", bbox_inches="tight")
    plt.show()





