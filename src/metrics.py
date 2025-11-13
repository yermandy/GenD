import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import wasserstein_distance
from sklearn import metrics as M


def ovr_roc(labels: np.ndarray, probs: np.ndarray):
    """
    Calculate the One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) and Area Under the ROC Curve (AUROC) for each class.

    Parameters:
    labels (np.ndarray): Array of true class labels. Shape should be (n_samples,).
    probs (np.ndarray): Array of predicted probabilities for each class. Shape should be (n_samples, n_classes).

    Returns:
    tuple: A tuple containing:
        - aurocs (list): List of AUROC values for each class.
        - fprs (list): List of false positive rates for each class.
        - tprs (list): List of true positive rates for each class.
        - ths (list): List of thresholds for each class.
        - ovr_macro_auroc (float): Macro-averaged AUROC for the OvR setting.
    """
    num_classes = probs.shape[1]
    labels_one_hot = np.eye(num_classes)[labels]
    fprs, tprs, ths = [], [], []

    # Why OvR with macro avg: https://chatgpt.com/share/677e448d-5bc0-8006-b9b5-081427b02857
    ovr_macro_auroc = M.roc_auc_score(labels_one_hot, probs, multi_class="ovr", average="macro")

    # Calculate OvR ROC and AUROC for each class
    for i in range(num_classes):
        fpr_class, tpr_class, ths_class = M.roc_curve(labels_one_hot[:, i], probs[:, i])
        ths_class = np.nan_to_num(ths_class, posinf=1.0)  # replace inf with max value
        ths_class = np.concatenate(([1], ths_class, [0]))  # add 0 and 1 thresholds
        fpr_class = np.concatenate(([0], fpr_class, [1]))  # add 0 and 1 fpr
        tpr_class = np.concatenate(([0], tpr_class, [1]))  # add 0 and 1 tpr
        fprs.append(fpr_class)
        tprs.append(tpr_class)
        ths.append(ths_class)

    return fprs, tprs, ths, ovr_macro_auroc


def ovr_prc(labels: np.ndarray, probs: np.ndarray):
    """
    Calculate the One-vs-Rest (OvR) Precision-Recall Curve (PRC) and the mean Average Precision (mAP) for a multi-class classification problem.

    Args:
        labels (np.ndarray): Array of true class labels with shape (n_samples,).
        probs (np.ndarray): Array of predicted probabilities with shape (n_samples, n_classes).

    Returns:
        tuple: A tuple containing:
            - precs (list of np.ndarray): List of precision values for each class.
            - recs (list of np.ndarray): List of recall values for each class.
            - ths (list of np.ndarray): List of threshold values for each class.
            - ovr_macro_ap (float): The mean Average Precision (mAP) score.
    """
    num_classes = probs.shape[1]
    labels_one_hot = np.eye(num_classes)[labels]
    precs, recs, ths = [], [], []

    # The same as mAP (mean Average Precision)
    ovr_macro_ap = M.average_precision_score(labels_one_hot, probs, average="macro")

    # Calculate OvR PRC for each class
    for i in range(num_classes):
        prec_class, rec_class, ths_class = M.precision_recall_curve(labels_one_hot[:, i], probs[:, i])
        ths_class = np.nan_to_num(ths_class, posinf=1.0)  # replace inf with max value
        ths_class = np.concatenate(([1], ths_class, [0]))  # add 0 and 1 thresholds
        prec_class = np.concatenate(([0], prec_class, [1]))  # add 0 and 1 precision
        rec_class = np.concatenate(([1], rec_class, [0]))  # add 0 and 1 recall
        precs.append(prec_class)
        recs.append(rec_class)
        ths.append(ths_class)

    return precs, recs, ths, ovr_macro_ap


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray, return_threshold: bool = False):
    """
    Returns the equal error rate (EER) and the threshold at which EER occurs
    for a binary classifier output.

    Args:
        y_true (np.ndarray): True binary labels.
        y_score (np.ndarray): Target scores, can either be probability estimates of the positive class,
                              confidence values, or non-thresholded measure of decisions.
                              Assumes shape (n_samples, 2) where column 1 is the positive class score.

    Returns:
        tuple: A tuple containing:
            - eer (float): The Equal Error Rate.
            - threshold (float): The threshold at which EER occurs. Returns NaN if EER calculation fails.
    """
    fpr, tpr, thresholds = M.roc_curve(y_true, y_score[:, 1], pos_label=1)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ValueError:
        eer = np.nan

    if return_threshold:
        return eer, float(interp1d(fpr, thresholds)(eer))

    return eer


def calculate_tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_targets: list = [0.01, 0.05]):
    """
    Calculate True Positive Rate (TPR) at specified False Positive Rate (FPR) levels for binary classification.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_score (np.ndarray): Predicted probabilities or scores, shape (n_samples, 2), where column 1 is for positive class.
        fpr_targets (list): List of FPR targets (e.g., [0.01, 0.05] for 1% and 5%).

    Returns:
        list: List of TPR values corresponding to the specified FPR targets. If a target FPR is out of range, NaN is returned for that target.
    """
    fpr, tpr, _ = M.roc_curve(y_true, y_score[:, 1], pos_label=1)

    results = []
    for target in fpr_targets:
        if target < fpr.min() or target > fpr.max():
            results.append(np.nan)
        else:
            results.append(np.interp(target, fpr, tpr))

    return results


def compute_wasserstein1_metrics(probs: np.ndarray, labels: np.ndarray):
    is_real = labels == 0
    is_fake = labels == 1

    if is_real.any() and is_fake.any():
        #! Compute Wasserstein-1 distance for inter-class separability
        # These W1(u, v) reflect how well the model separates the two classes
        # u ~ P(p(y=0|x) | y=0)
        # v ~ P(p(y=0|x) | y=1)
        W1_sep_real = wasserstein_distance(probs[is_real, 0], probs[is_fake, 0])

        # u ~ P(p(y=1|x) | y=0)
        # v ~ P(p(y=1|x) | y=1)
        W1_sep_fake = wasserstein_distance(probs[is_real, 1], probs[is_fake, 1])

        #! Compute Wasserstein-1 distance for intra-sample confidence margin
        # These W1(u, v) reflect how confident the model is about its predictions
        # u ∼ P(p(y=0∣x) ∣ y=0)
        # v ∼ P(p(y=1∣x) ∣ y=0)
        W1_conf_real = wasserstein_distance(probs[is_real, 0], probs[is_real, 1])

        # u ∼ P(p(y=0∣x) ∣ y=1)
        # v ∼ P(p(y=1∣x) ∣ y=1)
        W1_conf_fake = wasserstein_distance(probs[is_fake, 0], probs[is_fake, 1])

        return W1_sep_real, W1_sep_fake, W1_conf_real, W1_conf_fake

    return -1, -1, -1, -1
