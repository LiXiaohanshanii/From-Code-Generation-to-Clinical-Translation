"""Development and final fitting of the safeguarded CrAg Random Forest model.

Primary target
--------------
CSF cryptococcal antigen (CrAg) titer >= 1:80.

Design
------
1. The 2024 CrAg development dataset is split 4:1 (stratified; random_state=42)
   into an internal training subset and an internal validation subset.
2. Preprocessing and SMOTE are contained within an imblearn Pipeline to avoid
   preprocessing/resampling leakage.
3. Random-forest hyperparameters are NOT selected in this script. They are the
   fixed optimum obtained previously from the prespecified expanded grid search
   using mean cross-validated F1 score in the 2024 development data.
4. The internal validation subset is used to evaluate the selected model and to
   determine the reagent-saving operating probability threshold.
5. To reproduce the prespecified LLM-generated 4:1 workflow, the model fitted
   on the 80% internal training subset is retained unchanged and saved for
   temporal validation in the 2025 cohort; no full-development refit is performed.

No 2025 temporal-validation observations are used for model fitting, tuning, or
operating-threshold selection.
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Plot style: retained from the original CrAg modeling code
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_TITER = 80
INPUT_FILE = Path("CrAg_train.csv")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results_development")
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

NUM_COLS = ["CL", "GLU", "Protein", "RBC", "AGE"]
OHE_COLS = ["Color", "Transparency", "SEX", "DEPT", "DIAGNOSIS"]
ORD_COLS = ["SER-T", "Ink staining"]
FEATURE_COLS = NUM_COLS + OHE_COLS + ORD_COLS

# Fixed optimum from the previously completed expanded CV grid search.
# These parameters are not re-selected using the hold-out subset in this script.
SELECTED_PARAMS_FROM_EXPANDED_CV = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
}

# Protocol-based reagent utility assumptions used in the manuscript.
TP_SAVING = 5.0
FP_COST = 2.0
FN_COST = 1.0
THRESHOLD_GRID = np.arange(0.01, 1.00, 0.01)


def make_pipeline() -> ImbPipeline:
    """Create an unfitted leakage-safe preprocessing/SMOTE/RF pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_COLS),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                OHE_COLS,
            ),
            (
                "ord",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ORD_COLS,
            ),
        ],
        remainder="drop",
    )

    pipe = ImbPipeline(
        steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "clf",
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            ),
        ]
    )
    pipe.set_params(
        **{f"clf__{key}": value for key, value in SELECTED_PARAMS_FROM_EXPANDED_CV.items()}
    )
    return pipe


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, cutoff: float) -> dict:
    """Threshold-dependent classification metrics."""
    y_pred = (y_prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    accuracy = (tp + tn) / len(y_true)
    return {
        "probability_threshold": cutoff,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "accuracy": accuracy,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Youden_index": sensitivity + specificity - 1,
    }


def reagent_saving_curve(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Calculate mean protocol-based reagent saving over candidate cutoffs."""
    rows = []
    n = len(y_true)
    for cutoff in THRESHOLD_GRID:
        y_pred = (y_prob >= cutoff).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        saving = (TP_SAVING * tp - FP_COST * fp - FN_COST * fn) / n
        rows.append({"probability_threshold": cutoff, "mean_net_reagent_saving": saving})
    return pd.DataFrame(rows)



def plot_development_figures(model, X_val, y_val, y_prob, operating_threshold, best_saving, saving_curve):
    """Generate manuscript-style development figures using the original visual style."""
    y_arr = np.asarray(y_val, dtype=int)

    # ROC
    auc = roc_auc_score(y_arr, y_prob)
    fpr, tpr, _ = roc_curve(y_arr, y_prob)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(fpr, tpr, color="black", linestyle="-", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - Threshold ≥ {TARGET_TITER}-Development")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "ROC_thr80.png", dpi=300)
    plt.close(fig)

    # Empirical precision-recall curve plotted directly without
    # precision-envelope interpolation.
    # AP itself is calculated independently using average_precision_score().
    ap = average_precision_score(y_arr, y_prob)
    precision, recall, _ = precision_recall_curve(y_arr, y_prob)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(recall, precision, color="black", linestyle="-", label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - Threshold ≥ {TARGET_TITER}-Development")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "PR_thr80.png", dpi=300)
    plt.close(fig)

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_arr, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(mean_pred, frac_pos, linestyle="-", marker="o", color="black", label="Model Calibration")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")

    # Original descriptive trend line fitted to the binned calibration points.
    # IMPORTANT: this slope is a graphical/descriptive quantity and is NOT reported
    # as the formal calibration slope.
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

    brier = brier_score_loss(y_arr, y_prob)
    ll = log_loss(y_arr, y_prob)
    textstr = f"Brier Score: {brier:.4f}\nLog Loss: {ll:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(f"Calibration Curve - Threshold ≥ {TARGET_TITER}", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "Cal_thr80.png", dpi=300)
    plt.close(fig)

    # Reagent consumption-benefit curve
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(saving_curve["probability_threshold"], saving_curve["mean_net_reagent_saving"], color="black", linestyle="-", label="Model")
    ax.plot(operating_threshold, best_saving, marker="o", markersize=8, color="red",
            label=f"Best threshold = {operating_threshold:.2f}, Net saving = {best_saving:.4f}")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1)
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Average Net Reagent Saving per Case", fontsize=12)
    ax.set_title(f"Reagent Consumption-Benefit Curve - Threshold ≥ {TARGET_TITER}", fontsize=12)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.1), fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "RBC_thr80.png", dpi=300)
    plt.close(fig)

    # Feature importance from the internal-development model
    clf = model.named_steps["clf"]
    feature_names = model.named_steps["prep"].get_feature_names_out()
    importances = clf.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
    names, vals = zip(*feat_imp)
    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color="steelblue", edgecolor="k", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance (Top {len(names)}) - Threshold ≥ {TARGET_TITER}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "FI_thr80.png", dpi=300)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Data and target
# -----------------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE, encoding="utf-8")
required = set(FEATURE_COLS + ["CSF-T"])
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df[FEATURE_COLS].copy()
y = (pd.to_numeric(df["CSF-T"], errors="raise") >= TARGET_TITER).astype(int)

X_train, X_internal_val, y_train, y_internal_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

# -----------------------------------------------------------------------------
# Internal development-stage evaluation
# -----------------------------------------------------------------------------
development_model = make_pipeline()
development_model.fit(X_train, y_train)
y_prob_internal = development_model.predict_proba(X_internal_val)[:, 1]

probability_metrics = pd.DataFrame(
    [
        {
            "AUROC": roc_auc_score(y_internal_val, y_prob_internal),
            "average_precision": average_precision_score(y_internal_val, y_prob_internal),
            "Brier_score": brier_score_loss(y_internal_val, y_prob_internal),
            "log_loss": log_loss(y_internal_val, y_prob_internal),
            "n_internal_validation": len(y_internal_val),
            "event_rate": float(y_internal_val.mean()),
        }
    ]
)
probability_metrics.to_csv(RESULT_DIR / "CrAg80_internal_probability_metrics.csv", index=False)

# Conventional 0.50 cutoff is retained as a reference classification threshold.
pd.DataFrame(
    [classification_metrics(y_internal_val.to_numpy(), y_prob_internal, 0.50)]
).to_csv(RESULT_DIR / "CrAg80_internal_metrics_cutoff_0.50.csv", index=False)

# Select workflow operating threshold exclusively from 2024 internal validation data.
saving_curve = reagent_saving_curve(y_internal_val.to_numpy(), y_prob_internal)
saving_curve.to_csv(RESULT_DIR / "CrAg80_reagent_saving_curve.csv", index=False)
best_row = saving_curve.loc[saving_curve["mean_net_reagent_saving"].idxmax()]
OPERATING_THRESHOLD = float(best_row["probability_threshold"])
BEST_SAVING = float(best_row["mean_net_reagent_saving"])

plot_development_figures(
    development_model, X_internal_val, y_internal_val, y_prob_internal,
    OPERATING_THRESHOLD, BEST_SAVING, saving_curve
)

pd.DataFrame(
    [classification_metrics(y_internal_val.to_numpy(), y_prob_internal, OPERATING_THRESHOLD)]
).to_csv(RESULT_DIR / "CrAg80_internal_metrics_operating_cutoff.csv", index=False)

threshold_metadata = {
    "target_definition": "CSF-T >= 1:80",
    "target_titer_threshold": TARGET_TITER,
    "operating_probability_threshold": OPERATING_THRESHOLD,
    "operating_threshold_selection_dataset": "2024 internal validation subset",
    "selection_rule": "maximum prespecified mean net reagent saving",
    "mean_net_reagent_saving_at_selected_threshold": BEST_SAVING,
    "reference_probability_threshold": 0.50,
}
with open(MODEL_DIR / "CrAg80_operating_threshold.json", "w", encoding="utf-8") as f:
    json.dump(threshold_metadata, f, indent=2)

# -----------------------------------------------------------------------------
# Reproduce the original 4:1 workflow: retain the 80% training-subset model
# -----------------------------------------------------------------------------
joblib.dump(development_model, MODEL_DIR / "CrAg80_final_model.pkl")

model_metadata = {
    "feature_cols": FEATURE_COLS,
    "num_cols": NUM_COLS,
    "ohe_cols": OHE_COLS,
    "ord_cols": ORD_COLS,
    "target_titer_threshold": TARGET_TITER,
    "random_state": RANDOM_STATE,
    "selected_params_source": "previously completed expanded grid search using mean cross-validated F1",
    "selected_params": SELECTED_PARAMS_FROM_EXPANDED_CV,
    "final_fit_dataset": "80% internal training subset from the prespecified 4:1 split of the 2024 CrAg development dataset",
}
joblib.dump(model_metadata, MODEL_DIR / "CrAg80_model_metadata.pkl")

print(f"Target event rate (2024): {y.mean():.2%}")
print(f"Selected operating probability threshold: {OPERATING_THRESHOLD:.2f}")
print(f"Mean net reagent saving at selected threshold: {BEST_SAVING:.4f}")
print("Model fitted on the prespecified 80% 2024 training subset and saved unchanged for temporal validation.")
print("\nFigures saved to:", RESULT_DIR.resolve())
for _f in ["ROC_thr80.png", "PR_thr80.png", "Cal_thr80.png", "RBC_thr80.png", "FI_thr80.png"]:
    print("  -", (RESULT_DIR / _f).resolve())
