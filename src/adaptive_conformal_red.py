"""
Adaptive conformal redistribution (AdaConRed).

Implements the entropy-modulated, margin-aware nonconformity score,
the adaptive prediction-set construction, and the label-free
redistribution rule.

The redistribution rule is identical to that in split_conformal_red.py, so
differences in refined accuracy are attributable to set construction alone.

Requires the .npy softmax outputs written by encoder_classifier.py.

NOTE: class_names below MUST match LABELS in encoder_classifier.py, in the
same order, or every class index will be wrong.
"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Directory containing the .npy files written by encoder_classifier.py
DATA_ROOT = "path_to_dir_of_softmax_outputs"

# Load data
softmax_calib = np.load(os.path.join(DATA_ROOT, "calib_softmax_scores.npy"))
y_calib = np.load(os.path.join(DATA_ROOT, "y_calib.npy"))

# Compute nonconformity scores
probs = softmax_calib

# Entropy scaling parameter
lambda_entropy = 2.0

def aps_score_single(probs_row, y_true, lambda_entropy=2.0):

    sorted_idx = np.argsort(probs_row)[::-1]
    sorted_probs = probs_row[sorted_idx]

    # find rank of true label
    rank = np.where(sorted_idx == y_true)[0][0]

    # probability of true label
    p_true = probs_row[y_true]

    # entropy adaptation
    entropy = -np.sum(probs_row * np.log(probs_row + 1e-12))

    alpha_gap = lambda_entropy * entropy

    theta = 0.0

    for k in range(rank + 1):

        p_k = sorted_probs[k]

        # gap from true label
        gap = p_k - p_true

        # adaptive weight
        weight = 1 + alpha_gap * gap

        theta += weight * p_k

    return theta

# Apply to calibration set
nonconformity_scores = np.array([
    aps_score_single(
        probs[i],
        y_calib[i],
        lambda_entropy=lambda_entropy
    )
    for i in range(len(y_calib))
])

alpha = 0.2    # for 80% marginal coverage

# Load data
softmax_test = np.load(os.path.join(DATA_ROOT, "test_softmax_scores.npy"))
y_test = np.load(os.path.join(DATA_ROOT, "y_test.npy"))
cal_scores = nonconformity_scores

# Must match LABELS in encoder_classifier.py exactly, including order.
OSCC_SETTING = "3class"  # "3class" 

class_names = np.array(
    ["Benign", "OPMD", "OCA"] if OSCC_SETTING == "3class"
    else ["Healthy", "Benign", "OPMD", "OCA"]
)

assert softmax_test.shape[1] == len(class_names), (
    f"Softmax has {softmax_test.shape[1]} columns but {len(class_names)} class "
    "names are defined. class_names must match LABELS in encoder_classifier.py."
)

# Finite-sample conformal quantile
n = len(cal_scores)
q_hat = np.quantile(
    cal_scores,
    np.ceil((n + 1) * (1 - alpha)) / n,
    method="higher"
)

probs_test = softmax_test

# Build conformal prediction sets (not sorted)

def aps_prediction_set(probs_row, q_hat, lambda_entropy=1.0):
    sorted_idx = np.argsort(probs_row)[::-1]
    sorted_probs = probs_row[sorted_idx]

    entropy = -np.sum(probs_row * np.log(probs_row + 1e-12))
    
    alpha_gap = lambda_entropy * entropy

    cum_vals = []
    running = 0.0

    for k in range(len(sorted_probs)):
        p_k = sorted_probs[k]

        # approximate gap (since true label unknown at test time)
        if k == 0:
            gap = 0.0
        else:
            gap = sorted_probs[k-1] - p_k   # local drop

        weight = 1 + alpha_gap * gap

        running += weight * p_k
        cum_vals.append(running)

    cum_vals = np.array(cum_vals)

    # find cutoff
    idx = np.where(cum_vals >= q_hat)[0]
    if len(idx) == 0:
        cutoff = len(probs_row) - 1
    else:
        cutoff = idx[0]

    return sorted_idx[:cutoff+1]


prediction_sets = [
    aps_prediction_set(
        probs_test[i],
        q_hat,
        lambda_entropy=lambda_entropy
    )
    for i in range(len(probs_test))
]

prediction_sets = np.array(prediction_sets, dtype=object)

# Write class names in prediction sets

prediction_sets_labels = [
    [str(class_names[j]) for j in idx_set]
    for idx_set in prediction_sets
]

# Prediction sets with softmax scores

prediction_sets_scores = [
    [
        f"{class_names[j]} ({probs_test[i][j]:.4f})"
        for j in idx_set
    ]
    for i, idx_set in enumerate(prediction_sets)
]

# aps_prediction_set already returns indices in descending-probability order,
# so the "sorted" sets are the same objects. Aliased here to keep the naming
# parallel with split_conformal_red.py.

prediction_sets_sort = prediction_sets

# Write class names in prediction sets

prediction_sets_sort_labels = [
    [str(class_names[j]) for j in idx_set]
    for idx_set in prediction_sets_sort
]

# Sorted prediction sets with softmax scores 

prediction_sets_sort_scores = [
    [
        f"{class_names[j]} ({probs_test[i][j]:.4f})"
        for j in idx_set
    ]
    for i, idx_set in enumerate(prediction_sets_sort)
]

# Calculate coverage and average set size

coverage = np.mean([
    y_test[i] in prediction_sets[i]
    for i in range(len(y_test))
])

avg_set_size = np.mean([len(s) for s in prediction_sets])

print("Empirical coverage:", coverage)
print("Average set size:", avg_set_size)

# STANDARD CLASSIFICATION METRICS

preds = np.argmax(probs_test, axis=1)

# Overall accuracy
overall_acc = np.mean(preds == y_test)
correct_sample = np.sum(preds == y_test)
total_sample = len(y_test)
print("\nOverall Accuracy (argmax):", overall_acc)
print(f"Out of {total_sample} samples, {correct_sample} samples are correctly predicted")

# ------------------ Sensitivity (Recall - Macro) ------------------
sensitivity_macro = recall_score(y_test, preds, average='macro')

# ------------------ F1-score (Macro) ------------------
f1_macro = f1_score(y_test, preds, average='macro')

# ------------------ Specificity (Macro) ------------------
cm = confusion_matrix(y_test, preds)

specificity_list = []
for i in range(len(class_names)):
    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = np.sum(cm) - (TP + FP + FN)
    
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    specificity_list.append(specificity)

specificity_macro = np.mean(specificity_list)

# ------------------ AUC (Macro) ------------------
auc_macro = roc_auc_score(y_test, probs_test, multi_class='ovr', average='macro')

# ------------------ Print ------------------
print("Sensitivity (Macro):", sensitivity_macro)
print("Specificity (Macro):", specificity_macro)
print("F1-score (Macro):", f1_macro)
print("AUC (Macro):", auc_macro)

# ------------------ ROC Curve (Macro) ------------------
# Binarize labels
y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

fpr = dict()
tpr = dict()

for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs_test[:, i])

# Compute macro-average ROC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(len(class_names)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= len(class_names)

roc_auc_macro = auc(all_fpr, mean_tpr)

# Plot
plt.figure()
plt.plot(all_fpr, mean_tpr, label=f"Macro ROC (AUC = {roc_auc_macro:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Macro-Averaged ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6,5))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 14},
    linewidths=1,
    linecolor='black',
    square=True,
    cbar=False
)

ax.set_xlabel("Predicted Label", fontsize=14)
ax.set_ylabel("True Label", fontsize=14)

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()

# Class-wise accuracy
classwise_acc = {}

print("\nClass-wise Accuracy (argmax):")

for c in range(len(class_names)):
    idx = (y_test == c)
    if np.sum(idx) == 0:
        continue

    correct_c = np.sum(preds[idx] == y_test[idx])
    total_c = np.sum(idx)

    acc_c = correct_c / total_c
    classwise_acc[class_names[c]] = acc_c

    print(f"{class_names[c]}: {acc_c:.4f} "
          f"(Out of {total_c}, {correct_c} correctly predicted)")
    
    
# Cross-check
    
preds = np.argmax(probs_test, axis=1)

match = [
    preds[i] == prediction_sets_sort[i][0] if len(prediction_sets_sort[i]) > 0 else False
    for i in range(len(preds))
]

print("Total match:", np.sum(match))
print("Match rate:", np.mean(match))

# 5. Transitional Redistribution

# Get class indices
idx_benign = np.where(class_names == "Benign")[0][0]
idx_opmd   = np.where(class_names == "OPMD")[0][0]
idx_oca    = np.where(class_names == "OCA")[0][0]

# Copy original predictions
preds_adjusted = preds.copy()

# Set counters
corrections = 0
opmd_to_benign = 0
opmd_to_oca = 0

# Store sample indices (IDs)
opmd_to_benign_ids = []
opmd_to_oca_ids = []

# Correct / Wrong tracking
opmd_to_benign_correct = []
opmd_to_benign_wrong   = []

opmd_to_oca_correct = []
opmd_to_oca_wrong   = []

for i in range(len(preds)):

    pred_label = preds[i]

    # --- Label-free gate ---
    # Depends only on the point prediction and the cardinality of the adaptive
    # conformal set. Ground-truth labels are NOT consulted here.
    
    if pred_label != idx_opmd:
        continue
    if len(prediction_sets_sort[i]) < 2:
        continue

    # --- Reduced set ---
    # Gamma'_a(x) = Gamma_a(x) \ {AMB}; the set is already sorted by descending
    # probability, so the first survivor is the argmax.
    
    reduced = [j for j in prediction_sets_sort[i] if j != idx_opmd]
    if not reduced:
        continue

    new_label = int(reduced[0])
    preds_adjusted[i] = new_label
    corrections += 1

    # --- Outcome bookkeeping: y_test is used for REPORTING ONLY, after the
    #     reassignment decision has already been made. ---
    true_label = y_test[i]
    if new_label == idx_benign:
        opmd_to_benign += 1
        opmd_to_benign_ids.append(i)
        if true_label == idx_benign:
            opmd_to_benign_correct.append(i)
        else:
            opmd_to_benign_wrong.append(i)
    elif new_label == idx_oca:
        opmd_to_oca += 1
        opmd_to_oca_ids.append(i)
        if true_label == idx_oca:
            opmd_to_oca_correct.append(i)
        else:
            opmd_to_oca_wrong.append(i)

print("\nTotal reassignments applied:", corrections)
print(f"OPMD -> Benign: {opmd_to_benign}")
print(f"OPMD -> OCA:    {opmd_to_oca}")

# Transitional-class cost: true-OPMD samples moved away.
true_opmd_moved = [
    i for i in range(len(preds))
    if y_test[i] == idx_opmd and preds_adjusted[i] != idx_opmd
]
n_opmd_total = int(np.sum(y_test == idx_opmd))
to_benign = sum(1 for i in true_opmd_moved if preds_adjusted[i] == idx_benign)
to_oca = sum(1 for i in true_opmd_moved if preds_adjusted[i] == idx_oca)
print(
    f"\nTrue-OPMD samples reassigned: {len(true_opmd_moved)} of {n_opmd_total} "
    f"({to_benign} to Benign, {to_oca} to OCA)"
)

print("\n===== DETAILED CORRECTION ANALYSIS =====")

print("\n(i) OPMD -> Benign (Correct, true = Benign):")
print(opmd_to_benign_correct)

print("\n(ii) OPMD -> Benign (Wrong, true != Benign):")
print(opmd_to_benign_wrong)

print("\n(iii) OPMD -> OCA (Correct, true = OCA):")
print(opmd_to_oca_correct)

print("\n(iv) OPMD -> OCA (Wrong, true != OCA):")
print(opmd_to_oca_wrong)

print("\nCounts:")
print(f"Correct OPMD -> Benign: {len(opmd_to_benign_correct)}")
print(f"Wrong   OPMD -> Benign: {len(opmd_to_benign_wrong)}")

print(f"Correct OPMD -> OCA: {len(opmd_to_oca_correct)}")
print(f"Wrong   OPMD -> OCA: {len(opmd_to_oca_wrong)}")

# Compute new accuracy after redistribution

adjusted_correct = np.sum(preds_adjusted == y_test)
total_samples = len(y_test)

adjusted_acc = adjusted_correct / total_samples

print("\nAdjusted Accuracy (after OPMD redistribution):", adjusted_acc)
print(f"Out of {total_samples} samples, {adjusted_correct} samples are correctly predicted")
print("Total corrections applied:", corrections)

preds_new = preds_adjusted

# ------------------ Sensitivity (Macro) ------------------
sensitivity_macro_new = recall_score(y_test, preds_new, average='macro')

# ------------------ F1-score (Macro) ------------------
f1_macro_new = f1_score(y_test, preds_new, average='macro')

# ------------------ Specificity (Macro) ------------------
cm_new = confusion_matrix(y_test, preds_new)

specificity_list_new = []
for i in range(len(class_names)):
    TP = cm_new[i, i]
    FN = np.sum(cm_new[i, :]) - TP
    FP = np.sum(cm_new[:, i]) - TP
    TN = np.sum(cm_new) - (TP + FP + FN)
    
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    specificity_list_new.append(specificity)

specificity_macro_new = np.mean(specificity_list_new)

# ------------------ AUC (Macro) ------------------

# Redistribution reassigns discrete labels without modifying probs_test, so
# AUC is invariant by construction. Recomputed here purely as a
# cross check that the probability array was not mutated in place.

auc_macro_new = roc_auc_score(y_test, probs_test, multi_class="ovr", average="macro")
assert np.isclose(auc_macro_new, auc_macro), "AUC changed - probs_test was mutated."

# ------------------ Print ------------------
print("\n=== AFTER OPMD REDISTRIBUTION ===")
print("Sensitivity (Macro):", sensitivity_macro_new)
print("Specificity (Macro):", specificity_macro_new)
print("F1-score (Macro):", f1_macro_new)
print("AUC (Macro):", auc_macro_new)

# Confusion matrix
cm2 = confusion_matrix(y_test, preds_adjusted)
plt.figure(figsize=(6,5))
ax = sns.heatmap(
    cm2,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 14},
    linewidths=1,
    linecolor='black',
    square=True,
    cbar=False
)

ax.set_xlabel("Predicted Label", fontsize=14)
ax.set_ylabel("True Label", fontsize=14)

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()

# Class-wise accuracy after redistribution

classwise_adjusted_acc = {}

print("\nClass-wise Accuracy (after redistribution):")

for c in range(len(class_names)):
    idx = (y_test == c)
    if np.sum(idx) == 0:
        continue

    correct_c = np.sum(preds_adjusted[idx] == y_test[idx])
    total_c = np.sum(idx)

    acc_c = correct_c / total_c
    classwise_adjusted_acc[class_names[c]] = acc_c

    print(f"{class_names[c]}: {acc_c:.4f} "
          f"(Out of {total_c}, {correct_c} correctly predicted)")


