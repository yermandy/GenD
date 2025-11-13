import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from scipy.interpolate import interp1d
from sklearn import metrics as M

from src.utils import logger
from src.utils.decorators import TryExcept


@TryExcept("plot_curve")
def plot_curve(
    xs: list[ndarray],
    ys: list[ndarray],
    auc_threshold: float = 0.01,
    class_names: None | dict[int, str] = None,
    ax_plot=None,
    interpolate: int = 200,
    mean: bool = True,
    linestyles=["-", "--", "-.", ":"],
    palette: None | list | dict = None,
):
    # Use only one linestyle if up to 4 classes
    if len(xs) <= 4:
        linestyles = linestyles[:1]

    # Create figure with larger size and better aspect ratio
    plt.figure(figsize=(10, 8), tight_layout=True)

    # Create two subplots - one for the plot, one for the legend
    gs = plt.GridSpec(1, 2, width_ratios=[4, 1])
    ax_plot = plt.subplot(gs[0])
    ax_legend = plt.subplot(gs[1])

    if palette is None:
        palette = sns.husl_palette(len(xs))

    if interpolate != -1:
        x_new = np.linspace(0, 1, interpolate)
        ys = [interp1d(x, y)(x_new) for x, y in zip(xs, ys)]
        xs = [x_new] * len(xs)

    # Plot curves on the main axis
    active_classes = []
    for c, (x, y) in enumerate(zip(xs, ys)):
        auc = M.auc(x, y)
        if auc >= auc_threshold:  # Only plot and include in legend if AUC > threshold
            class_name = f"{c}: {class_names[c]}" if class_names else c
            label = f"{class_name} (AUC: {auc:.2f})"
            linestyle = linestyles[c % len(linestyles)]
            line = ax_plot.plot(x, y, label=label, linewidth=1.5, color=palette[c], linestyle=linestyle)
            active_classes.append((line[0], label))

    if mean and interpolate != -1:
        ys_mean = np.mean(ys, axis=0)
        xs_mean = np.mean(xs, axis=0)

        # Plot mean curve
        auc = M.auc(xs_mean, ys_mean)
        label = f"avg (AUC: {auc:.2f})"
        ax_plot.plot(xs_mean, ys_mean, label="avg", linewidth=1.5, color="black", linestyle="-")
        active_classes.append((ax_plot.lines[-1], label))

    # Set square aspect ratio
    ax_plot.set_aspect("equal")

    # Set limits explicitly to ensure square plot
    ax_plot.set_xlim(-0.02, 1.02)  # Slight padding for better visibility
    ax_plot.set_ylim(-0.02, 1.02)

    # Customize the main plot
    ax_plot.grid(True, linestyle="--", alpha=0.3)

    # Create legend in the second subplot
    ax_legend.axis("off")  # Hide the axis
    if active_classes:
        lines, labels = zip(*active_classes)
        ax_legend.legend(lines, labels, loc="center left", fontsize=10, borderaxespad=0)

    return ax_plot


@TryExcept("plot_roc_curve")
def plot_roc_curve(
    fprs: list[ndarray],
    tprs: list[ndarray],
    ths: list[ndarray],
    title: str = "ROC",
    path: str = "roc_curve.png",
    auc_threshold: float = 0.01,
    class_names: None | dict[int, str] = None,
):
    """
    Plot ROC curve for multiple classes.
    """
    ax_plot = plot_curve(fprs, tprs, auc_threshold, class_names)

    # Add the diagonal line
    ax_plot.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.5)

    ax_plot.set_title(title, fontsize=14)
    ax_plot.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax_plot.set_ylabel("True Positive Rate (TPR)", fontsize=12)

    # Save with high quality
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


@TryExcept("plot_prc_curve")
def plot_prc_curve(
    prcs: list[ndarray],
    recs: list[ndarray],
    ths: list[ndarray],
    title: str = "PRC",
    path: str = "pr_curve.png",
    auc_threshold: float = 0.01,
    class_names: None | dict[int, str] = None,
    show_f1_lines: bool = True,
):
    """
    Plot Precision-Recall curve for multiple classes.
    """
    ax_plot = plot_curve(recs, prcs, auc_threshold, class_names)

    if show_f1_lines:
        f_scores = np.linspace(0.1, 0.9, num=9)  # F1 scores to plot
        for f_score in f_scores:
            r = np.linspace(0.001, 1, 100)  # Recall
            p = f_score * r / (2 * r - f_score)  # Precision for given F1 score
            mask = p > 0
            ax_plot.plot(r[mask], p[mask], color="gray", alpha=0.2, linestyle="--")
            ax_plot.annotate("F1={0:0.1f}".format(f_score), xy=(0.95, p[-1] - 0.02), alpha=0.2)

    # Customize the main plot
    ax_plot.set_title(title, fontsize=14)
    ax_plot.set_xlabel("Recall", fontsize=12)
    ax_plot.set_ylabel("Precision", fontsize=12)

    # Save with high quality
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


@TryExcept("plot_f1_curve")
def plot_f1_curve(
    prcs: list[ndarray],
    recs: list[ndarray],
    ths: list[ndarray],
    title: str = "F1",
    path: str = "f1_curve.png",
    auc_threshold: float = 0.01,
    class_names: None | dict[int, str] = None,
):
    """
    Plot F1 curve for multiple classes
    """
    f1s = []
    for prc, rec in zip(prcs, recs):
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where((prc + rec) == 0, 0, 2 * prc * rec / (prc + rec))
        f1 = f1[:-1]
        f1s.append(f1)

    ax_plot = plot_curve(ths, f1s, auc_threshold, class_names)

    # Customize the main plot
    ax_plot.set_title(title, fontsize=14)
    ax_plot.set_xlabel("Threshold", fontsize=12)
    ax_plot.set_ylabel("F1 Score", fontsize=12)

    # Save with high quality
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


@TryExcept("plot_fpr_fnr_curve")
def plot_fpr_fnr_curve(
    fprs: list[ndarray],  # 2 x ths
    tprs: list[ndarray],  # 2 x ths
    ths: list[ndarray],  # 2 x ths
    title: str = "FPR vs FNR",
    path: str = "fpr_fnr_curve.png",
    auc_threshold: float = 0.01,
    eer: None | float = None,
):
    """
    Plot FPR vs FNR curve and EER for binary classification
    """
    if len(fprs) != 2:
        logger.print_warning_once("FPR vs FNR curve is only plotted for 2 classes")
        return

    # Calculate FNR from TPR
    fpr = fprs[1]
    fnr = 1 - tprs[1]

    xs = [ths[1], ths[1]]
    ys = [fpr, fnr]

    class_names = {0: "FPR", 1: "FNR"}

    ax_plot = plot_curve(xs, ys, auc_threshold, class_names, mean=False, linestyles=["-"])

    if eer is not None:
        ax_plot.axhline(y=eer, color="black", linestyle="--")
        ax_plot.text(0, eer + 0.02, f"EER: {eer:.2f}", color="black", fontsize=10)

    ax_plot.set_title(title, fontsize=14)
    ax_plot.set_xlabel("Threshold", fontsize=12)
    ax_plot.set_ylabel("FPR vs FNR", fontsize=12)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


@TryExcept("plot_confusion_matrix")
def plot_confusion_matrix(
    confusion_matrix: ndarray,
    class_names: None | dict[int, str] = None,
    title: str = "Confusion Matrix",
    path: str = "confusion_matrix.png",
    normalize: bool = False,
):
    """
    Plot confusion matrix
    """
    N = len(confusion_matrix)
    size = max(10, N / 2)
    plt.figure(figsize=(size, size), tight_layout=True)
    fmt = "d"
    if normalize:
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True) * 100
        confusion_matrix[np.isnan(confusion_matrix)] = 0
        fmt = ".2f"

    labels = [f"{k}: {v}" for k, v in class_names.items()] if class_names else None
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"fontsize": 8},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


@TryExcept("plot_features_2d")
def plot_features_2d(
    features_2d: np.ndarray,  # (N, 2)
    set_ids: np.ndarray,  # (N,)
    id2label: dict[int, str],  # dict {id: label}
    output_path: str,
):
    assert isinstance(features_2d, np.ndarray)
    assert isinstance(set_ids, np.ndarray)
    assert isinstance(id2label, dict)

    plt.figure(figsize=(25, 25))

    palette = sns.husl_palette(len(id2label))
    id2color = {id: palette[i] for i, id in enumerate(id2label)}

    for id, label in id2label.items():
        mask = set_ids == id

        if not np.any(mask):
            continue

        xs = features_2d[mask, 0]
        ys = features_2d[mask, 1]

        if "real" in label:
            marker = "."
        else:
            marker = "x"

        plt.scatter(xs, ys, c=[id2color[id]] * len(xs), marker=marker, label=label)

        for x, y, label in zip(xs, ys, set_ids[mask]):
            plt.text(x, y, label, c=id2color[id], fontsize=9)

    plt.legend(loc="best", title="Models")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_path.replace(".png", ".svg"))


@TryExcept("plot_probs_distribution")
def plot_probs_distribution(
    probs: np.ndarray,  # (N, C)
    labels: np.ndarray,  # (N,)
    class_names: dict[int, str],  # dict {id: label}
    output_path: str,
):
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 4 * n_classes))
    palette = sns.husl_palette(n_classes)

    # Find global min and max for x-axis limits
    x_min = probs.min()
    x_max = probs.max()
    x_min, x_max = -0.005, 1.005

    for idx, (class_idx, class_name) in enumerate(class_names.items()):
        ax = axes[idx]

        # Get probabilities for current class
        class_mask = labels == class_idx
        class_probs = probs[class_mask]

        # Plot probability distribution for each possible class prediction
        for pred_idx, pred_name in class_names.items():
            pred_probs = class_probs[:, pred_idx]
            sns.histplot(
                data=pred_probs,
                label=f"ŷ={pred_name}",
                color=palette[pred_idx],
                alpha=0.2,
                bins=100,
                stat="probability",
                kde=True,
                element="step",
                ax=ax,
            )

        ax.set_xlabel("Scores")
        ax.set_ylabel("Probability")
        ax.set_title(f"Histogram p(ŷ|y={class_name})     y – true, ŷ – predicted class", color=palette[class_idx])
        ax.set_xlim(x_min, x_max)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
