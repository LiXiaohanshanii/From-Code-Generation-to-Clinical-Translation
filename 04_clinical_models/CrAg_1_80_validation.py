"""Temporal validation of the final safeguarded CrAg >=1:80 Random Forest model.

The fitted pipeline and operating threshold are derived exclusively from the
2024 development dataset. The 2025 temporal-validation cohort is used only for
out-of-sample evaluation. Two probability cutoffs are reported:
    - 0.50: conventional reference cutoff;
    - development-derived operating cutoff (expected to be 0.47 in the reported analysis).
AUROC, average precision, Brier score, log loss, ROC/PR curves, and calibration
are probability-based and therefore do not depend on the classification cutoff.
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("models/CrAg80_final_model.pkl")
META_PATH = Path("models/CrAg80_model_metadata.pkl")
THRESHOLD_PATH = Path("models/CrAg80_operating_threshold.json")
EXTERNAL_FILE = Path("external_test.csv")
RESULT_DIR = Path("results_temporal_validation")
FIGURE_DIR = RESULT_DIR / "figures"
RESULT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# Plot style retained from the original CrAg prediction code.
plt.style.use("seaborn-v0_8-white")
sns.despine(top=False, right=False, left=False, bottom=False)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (9, 8)
plt.rcParams["axes.unicode_minus"] = False

model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)
with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold_meta = json.load(f)

feature_cols = meta["feature_cols"]
TARGET_TITER = int(meta["target_titer_threshold"])
OPERATING_THRESHOLD = float(threshold_meta["operating_probability_threshold"])
REFERENCE_THRESHOLD = 0.50

# -----------------------------------------------------------------------------
# Temporal-validation data and probability predictions
# -----------------------------------------------------------------------------
df_ext = pd.read_csv(EXTERNAL_FILE, encoding="utf-8")
required = set(feature_cols + ["CSF-T"])
missing = sorted(required - set(df_ext.columns))
if missing:
    raise ValueError(f"Missing required columns in temporal-validation data: {missing}")

X_ext = df_ext[feature_cols].copy()
y_titer = pd.to_numeric(df_ext["CSF-T"], errors="raise")
y_true = (y_titer >= TARGET_TITER).astype(int).to_numpy()
y_prob = model.predict_proba(X_ext)[:, 1]


def classification_metrics(cutoff: float) -> dict:
    y_pred = (y_prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "probability_threshold": cutoff,
        "threshold_role": "conventional reference" if np.isclose(cutoff, 0.50) else "2024 development-derived operating threshold",
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "accuracy": (tp + tn) / len(y_true),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Youden_index": sensitivity + specificity - 1,
    }


# Threshold-dependent metrics at both prespecified cutoffs in one run.
threshold_metrics = pd.DataFrame(
    [classification_metrics(REFERENCE_THRESHOLD), classification_metrics(OPERATING_THRESHOLD)]
)
threshold_metrics.to_csv(RESULT_DIR / "CrAg80_temporal_threshold_metrics.csv", index=False)

# Threshold-independent probability metrics.
probability_metrics = pd.DataFrame(
    [
        {
            "N": len(y_true),
            "events": int(y_true.sum()),
            "event_rate": float(y_true.mean()),
            "AUROC": roc_auc_score(y_true, y_prob),
            "average_precision": average_precision_score(y_true, y_prob),
            "Brier_score": brier_score_loss(y_true, y_prob),
            "log_loss": log_loss(y_true, y_prob),
        }
    ]
)
probability_metrics.to_csv(RESULT_DIR / "CrAg80_temporal_probability_metrics.csv", index=False)

# -----------------------------------------------------------------------------
# Standard calibration intercept and slope
# Logistic recalibration: logit(Y) = intercept + slope * logit(predicted risk)
# -----------------------------------------------------------------------------
eps = 1e-6
p_clip = np.clip(y_prob, eps, 1 - eps)
logit_p = np.log(p_clip / (1 - p_clip))
X_cal = sm.add_constant(logit_p)
cal_model = sm.GLM(y_true, X_cal, family=sm.families.Binomial()).fit()
calibration_intercept = float(cal_model.params[0])
calibration_slope = float(cal_model.params[1])

pd.DataFrame(
    [{"calibration_intercept": calibration_intercept, "calibration_slope": calibration_slope}]
).to_csv(RESULT_DIR / "CrAg80_temporal_calibration_parameters.csv", index=False)

# -----------------------------------------------------------------------------
# Predictions: keep both classification cutoffs explicitly
# -----------------------------------------------------------------------------
predictions = pd.DataFrame(
    {
        "true_CSF_T": y_titer.to_numpy(),
        "true_label_ge80": y_true,
        "predicted_probability": y_prob,
        "predicted_label_cutoff_0.50": (y_prob >= REFERENCE_THRESHOLD).astype(int),
        "predicted_label_operating_cutoff": (y_prob >= OPERATING_THRESHOLD).astype(int),
    }
)
predictions.to_csv(RESULT_DIR / "CrAg80_temporal_predictions.csv", index=False)

# -----------------------------------------------------------------------------
# ROC curve
# -----------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_true, y_prob)
fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(fpr, tpr, color="black", label=f"AUROC = {roc_auc_score(y_true, y_prob):.4f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve (Threshold ≥ {TARGET_TITER}) - Temporal validation")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "External_ROC_thr80.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# Precision-recall curve
# Empirical PR curve plotted directly without precision-envelope interpolation.
# AP itself remains the standard average_precision_score and is unaffected.
# -----------------------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_true, y_prob)
fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(recall, precision, color="black", label=f"AP = {average_precision_score(y_true, y_prob):.4f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"PR Curve (Threshold ≥ {TARGET_TITER}) - Temporal validation")
ax.legend(loc="best")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0, 1.01, 0.2))
fig.tight_layout()
fig.savefig(FIGURE_DIR / "External_PR_thr80.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# Calibration plot
# Binned points are for visualization only. The red dashed curve is the fitted
# logistic recalibration relationship based on individual predicted probabilities.
# -----------------------------------------------------------------------------
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(
    mean_pred,
    frac_pos,
    linestyle="-",
    marker="o",
    color="black",
    label="Model Calibration",
)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")

# Preserve the original red dashed descriptive linear fit through the binned
# calibration points. Its slope is NOT the formal calibration slope.
mask = ~np.isnan(mean_pred) & ~np.isnan(frac_pos)
if np.sum(mask) > 1:
    reg = LinearRegression()
    reg.fit(mean_pred[mask].reshape(-1, 1), frac_pos[mask])
    trend_slope = float(reg.coef_[0])
    trend_intercept = float(reg.intercept_)
    trend_r2 = float(reg.score(mean_pred[mask].reshape(-1, 1), frac_pos[mask]))
    x_line = np.linspace(0, 1, 100)
    y_line = reg.predict(x_line.reshape(-1, 1))
    ax.plot(
        x_line, y_line, "r--", linewidth=2,
        label=(f"Descriptive linear fit: y={trend_slope:.2f}x"
               f"{trend_intercept:+.2f}, R²={trend_r2:.3f}")
    )

brier = brier_score_loss(y_true, y_prob)
logloss = log_loss(y_true, y_prob)
textstr = f"Brier Score: {brier:.4f}\nLog Loss: {logloss:.4f}"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(
    0.05,
    0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=props,
)

ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives", fontsize=12)
ax.set_title(f"Calibration Curve (Threshold ≥ {TARGET_TITER}) - Temporal validation", fontsize=12)
ax.legend(loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "External_Cal_thr80.png", dpi=300)
plt.close(fig)

print("Temporal validation complete.")
print(probability_metrics.to_string(index=False))
print(threshold_metrics.to_string(index=False))
print(f"Calibration intercept: {calibration_intercept:.4f}")
print(f"Formal logistic calibration intercept: {calibration_intercept:.4f}")
print(f"Formal logistic calibration slope: {calibration_slope:.4f}")
if "trend_slope" in globals():
    print(f"Descriptive binned linear-fit slope: {trend_slope:.4f}")
    print(f"Descriptive binned linear-fit R^2: {trend_r2:.4f}")
print("\nFigures saved to:")
for _f in ["External_ROC_thr80.png", "External_PR_thr80.png", "External_Cal_thr80.png"]:
    print("  -", (FIGURE_DIR / _f).resolve())
