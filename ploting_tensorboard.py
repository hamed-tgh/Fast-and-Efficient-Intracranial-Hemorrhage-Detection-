from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def tensorboard_to_df(log_dir, tag):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    if len(events) == 0:
        raise ValueError(f"No scalar events found for tag='{tag}' in '{log_dir}'. "
                         f"Check the tag name in TensorBoard Scalars.")
    return pd.DataFrame({"step": [e.step for e in events], "value": [e.value for e in events]})

def ema_smooth(y, alpha=0.2):
    
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]
    return out

def plot_journal_curve(
    df_train, df_val,
    xlabel="Epoch", ylabel="Loss",
    title=None,
    smooth_alpha=0.0,           # 0 = no smoothing, e.g. 0.15..0.3 recommended if noisy
    save_base="training_curve", # outputs: .pdf and .png
    width_in=3.35,              # ~ single column (IEEE) ; use 6.9 for double column
    height_in=2.4,
    dpi=600
):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "figure.dpi": 120,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42,   
        "ps.fonttype": 42
    })

    x_tr, y_tr = df_train["step"].to_numpy(), df_train["value"].to_numpy()
    x_va, y_va = df_val["step"].to_numpy(), df_val["value"].to_numpy()

    if smooth_alpha and smooth_alpha > 0:
        y_tr_s = ema_smooth(y_tr, alpha=smooth_alpha)
        y_va_s = ema_smooth(y_va, alpha=smooth_alpha)
    else:
        y_tr_s, y_va_s = y_tr, y_va

    fig, ax = plt.subplots(figsize=(width_in, height_in))

    ax.plot(x_tr, y_tr_s, label="Train")
    ax.plot(x_va, y_va_s, label="Validation")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, which="major", linewidth=0.4, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.2)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()

    
    fig.savefig(f"{save_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_base}.png", bbox_inches="tight")
    plt.show()



def main():

    log_dir = "runs/RepViT_experiment"
    df_train = tensorboard_to_df(log_dir, "Loss/train_epoch")
    df_val   = tensorboard_to_df(log_dir, "Loss/val_epoch")

    plot_journal_curve(
        df_train, df_val,
        xlabel="Epoch",
        ylabel="BCEWithLogitsLoss",
        title="Training and Validation Loss",
        smooth_alpha=0.15,        
        save_base="loss_curve_journal_singlecol",
        width_in=3.35, height_in=2.4, dpi=600
    )


