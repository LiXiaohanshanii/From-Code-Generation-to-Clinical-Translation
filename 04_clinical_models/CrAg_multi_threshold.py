import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# Global plotting settings
# ============================================================

plt.style.use('seaborn-v0_8-white')

# Retain all four plot borders.
sns.despine(
    top=False,
    right=False,
    left=False,
    bottom=False
)

plt.rcParams['axes.grid'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (9, 8)
plt.rcParams['axes.unicode_minus'] = False


# Seven line styles correspond to the seven prespecified
# CrAg titer targets and remain distinguishable in grayscale.
LINE_STYLES = [
    '-',
    '--',
    '-.',
    ':',
    (0, (3, 1, 1, 1)),
    (0, (5, 2)),
    (0, (1, 1))
]

MARKERS = [
    'o',
    's',
    'D',
    '^',
    'v',
    'p',
    '*'
]

STYLE_LIST = list(
    zip(
        LINE_STYLES,
        MARKERS
    )
)


# ============================================================
# Output directories
# ============================================================

BASE_DIR = Path('.')

RESULT_DIR = (
    BASE_DIR /
    'results_development'
)

FIGURE_DIR = (
    RESULT_DIR /
    'figures'
)

MODEL_DIR = (
    BASE_DIR /
    'models'
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Data loading
# ============================================================

df = pd.read_csv(
    'CrAg_train.csv',
    encoding='utf-8'
)

num_cols = [
    'CL',
    'GLU',
    'Protein',
    'RBC',
    'AGE'
]

ohe_cols = [
    'Color',
    'Transparency',
    'SEX',
    'DEPT',
    'DIAGNOSIS'
]

ord_cols = [
    'SER-T',
    'Ink staining'
]

feature_cols = (
    num_cols +
    ohe_cols +
    ord_cols
)

X = df[
    feature_cols
].copy()


# ============================================================
# Preprocessing definition
# ============================================================
# The preprocessing object is not fitted globally.
#
# A fresh clone is inserted into each model-specific
# imbalanced-learn Pipeline. Therefore, preprocessing is fitted
# using the corresponding 80% training partition only.
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            SimpleImputer(
                strategy='median'
            ),
            num_cols
        ),
        (
            'ohe',
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ),
            ohe_cols
        ),
        (
            'ord',
            OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ),
            ord_cols
        )
    ],
    remainder='drop'
)


# ============================================================
# Prespecified CrAg clinical titer targets
# ============================================================

thresholds = [
    20,
    40,
    80,
    160,
    320,
    640,
    1280
]

results = {}


# ============================================================
# Optimal RF parameters from the preceding expanded
# hyperparameter search
# ============================================================
# These parameter sets were identified previously using
# cross-validated F1 score and are fixed in the present
# development-stage analysis.
# ============================================================

best_params_by_thr = {

    20: {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced'
    },

    40: {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'class_weight': 'balanced'
    },

    80: {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced'
    },

    160: {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'class_weight': 'balanced_subsample'
    },

    320: {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 10,
        'min_samples_leaf': 1,
        'class_weight': 'balanced_subsample'
    },

    640: {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 4,
        'class_weight': 'balanced'
    },

    1280: {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced_subsample'
    }
}


# ============================================================
# Formal logistic calibration
# ============================================================

def calculate_logistic_calibration(
    y_true,
    y_prob
):
    """
    Estimate formal calibration intercept and slope using
    individual-level logistic recalibration:

        logit[P(Y=1)] = alpha + beta * logit(predicted risk)

    Ideal calibration:
        alpha = 0
        beta  = 1
    """

    y_true = np.asarray(
        y_true
    )

    y_prob = np.asarray(
        y_prob
    )

    eps = 1e-6

    p = np.clip(
        y_prob,
        eps,
        1 - eps
    )

    logit_p = np.log(
        p / (1 - p)
    ).reshape(
        -1,
        1
    )

    calibration_model = LogisticRegression(
        penalty=None,
        solver='lbfgs',
        max_iter=2000
    )

    calibration_model.fit(
        logit_p,
        y_true
    )

    intercept = (
        calibration_model
        .intercept_[0]
    )

    slope = (
        calibration_model
        .coef_[0][0]
    )

    return (
        intercept,
        slope
    )


# ============================================================
# Model fitting and internal validation
# ============================================================
# For each prespecified CrAg titer target, a separate
# stratified 4:1 training/internal-validation split is
# performed using random_state=42.
#
# Consistent with the original safeguarded LLM-generated
# workflow (Scheme A), the model used for subsequent temporal
# validation is fitted only on the corresponding 80% training
# partition. No refitting on the complete 2024 development
# dataset is performed.
# ============================================================

for thr in thresholds:

    print(
        '\n' +
        '=' * 65
    )

    print(
        f'CrAg target: CSF-T >= 1:{thr}'
    )

    print(
        '=' * 65
    )

    # Define binary outcome.
    y = (
        df['CSF-T'] >= thr
    ).astype(int)

    print(
        f'Overall event rate: '
        f'{y.mean():.2%}'
    )

    # Prespecified stratified 4:1 split.
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print(
        f'Training N: '
        f'{len(y_train)}'
    )

    print(
        f'Internal validation N: '
        f'{len(y_val)}'
    )

    print(
        f'Internal validation event rate: '
        f'{y_val.mean():.2%}'
    )

    # Use an independent preprocessing object for each target.
    pipeline = ImbPipeline(
        steps=[
            (
                'prep',
                clone(
                    preprocessor
                )
            ),
            (
                'smote',
                SMOTE(
                    random_state=42
                )
            ),
            (
                'clf',
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=1
                )
            )
        ]
    )

    params = (
        best_params_by_thr[
            thr
        ]
    )

    pipeline.set_params(
        **{
            f'clf__{key}': value
            for key, value
            in params.items()
        }
    )

    # Fit on the 80% training partition only.
    pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_prob = (
        pipeline
        .predict_proba(
            X_val
        )[:, 1]
    )

    # IMPORTANT:
    # Preserve the original Random Forest prediction rule for
    # development-stage threshold-dependent metrics.
    #
    # This intentionally uses pipeline.predict() rather than
    # manually applying y_prob >= 0.50, because tie handling at
    # exactly 0.50 can otherwise produce a one-case difference.
    y_pred = (
        pipeline
        .predict(
            X_val
        )
    )

    # --------------------------------------------------------
    # Threshold-independent performance
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_val,
        y_prob
    )

    ap = average_precision_score(
        y_val,
        y_prob
    )

    brier = brier_score_loss(
        y_val,
        y_prob
    )

    logloss = log_loss(
        y_val,
        y_prob
    )

    # --------------------------------------------------------
    # Threshold-dependent classification performance
    # --------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_val,
            y_pred
        ).ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    f1 = f1_score(
        y_val,
        y_pred
    )

    youden_index = (
        sensitivity +
        specificity -
        1
    )

    # --------------------------------------------------------
    # Formal logistic calibration
    # --------------------------------------------------------

    (
        formal_cal_intercept,
        formal_cal_slope
    ) = calculate_logistic_calibration(
        y_val,
        y_prob
    )

    # Feature names after fitted preprocessing.
    feature_names = (
        pipeline
        .named_steps['prep']
        .get_feature_names_out()
    )

    print(
        f'AUROC = '
        f'{auc:.4f}'
    )

    print(
        f'Average precision = '
        f'{ap:.4f}'
    )

    print(
        f'Brier score = '
        f'{brier:.4f}'
    )

    print(
        f'Log loss = '
        f'{logloss:.4f}'
    )

    print(
        f'Sensitivity = '
        f'{sensitivity:.4f}'
    )

    print(
        f'Specificity = '
        f'{specificity:.4f}'
    )

    print(
        f'PPV = '
        f'{ppv:.4f}'
    )

    print(
        f'NPV = '
        f'{npv:.4f}'
    )

    print(
        f'Accuracy = '
        f'{accuracy:.4f}'
    )

    print(
        f'F1 = '
        f'{f1:.4f}'
    )

    print(
        f'Youden index = '
        f'{youden_index:.4f}'
    )

    print(
        f'Formal calibration intercept = '
        f'{formal_cal_intercept:.4f}'
    )

    print(
        f'Formal calibration slope = '
        f'{formal_cal_slope:.4f}'
    )

    results[
        thr
    ] = {

        'y_val':
            y_val.to_numpy(),

        'y_prob':
            y_prob,

        'y_pred':
            y_pred,

        'model':
            pipeline,

        'auc':
            auc,

        'ap':
            ap,

        'brier':
            brier,

        'log_loss':
            logloss,

        'sensitivity':
            sensitivity,

        'specificity':
            specificity,

        'ppv':
            ppv,

        'npv':
            npv,

        'accuracy':
            accuracy,

        'f1':
            f1,

        'youden_index':
            youden_index,

        'formal_calibration_intercept':
            formal_cal_intercept,

        'formal_calibration_slope':
            formal_cal_slope,

        'event_rate':
            y_val.mean(),

        'params':
            params,

        'feature_names':
            feature_names,

        'confusion':
            (
                tn,
                fp,
                fn,
                tp
            )
    }


# ============================================================
# Internal-validation summary table
# ============================================================

summary_rows = []

for thr, res in results.items():

    (
        tn,
        fp,
        fn,
        tp
    ) = res['confusion']

    summary_rows.append({

        'CrAg_titer_target':
            f'>=1:{thr}',

        'N_internal_validation':
            len(
                res['y_val']
            ),

        'Events':
            int(
                np.sum(
                    res['y_val']
                )
            ),

        'Event_rate':
            res[
                'event_rate'
            ],

        'Sensitivity':
            res[
                'sensitivity'
            ],

        'Specificity':
            res[
                'specificity'
            ],

        'PPV':
            res[
                'ppv'
            ],

        'NPV':
            res[
                'npv'
            ],

        'Accuracy':
            res[
                'accuracy'
            ],

        'F1':
            res[
                'f1'
            ],

        'AUROC':
            res[
                'auc'
            ],

        'Average_precision':
            res[
                'ap'
            ],

        'Brier_score':
            res[
                'brier'
            ],

        'Log_loss':
            res[
                'log_loss'
            ],

        'Youden_index':
            res[
                'youden_index'
            ],

        'Formal_calibration_intercept':
            res[
                'formal_calibration_intercept'
            ],

        'Formal_calibration_slope':
            res[
                'formal_calibration_slope'
            ],

        'TN':
            tn,

        'FP':
            fp,

        'FN':
            fn,

        'TP':
            tp
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    RESULT_DIR /
    'CrAg_multithreshold_internal_validation_metrics.csv',
    index=False,
    encoding='utf-8-sig'
)

print(
    '\n' +
    '=' * 80
)

print(
    'Summary of internal-validation performance'
)

print(
    '=' * 80
)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x:
        f'{x:.4f}'
    )
)


# ============================================================
# Single-target plotting
# ============================================================

def plot_single_curves(
    thr,
    res,
    style_idx
):

    y_val = np.asarray(
        res['y_val']
    )

    y_prob = np.asarray(
        res['y_prob']
    )

    linestyle, marker = (
        STYLE_LIST[
            style_idx %
            len(STYLE_LIST)
        ]
    )

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_val,
        y_prob
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.plot(
        fpr,
        tpr,
        linestyle=linestyle,
        color='black',
        linewidth=2,
        label=(
            f'AUROC = '
            f'{res["auc"]:.4f}'
        )
    )

    ax.plot(
        [0, 1],
        [0, 1],
        'k--',
        alpha=0.5,
        label='Random'
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_xticks(
        np.arange(
            0,
            1.01,
            0.2
        )
    )

    ax.set_yticks(
        np.arange(
            0,
            1.01,
            0.2
        )
    )

    ax.set_xlabel(
        'False Positive Rate'
    )

    ax.set_ylabel(
        'True Positive Rate'
    )

    ax.set_title(
        f'ROC Curve - CrAg Target ≥1:{thr}'
    )

    ax.legend(
        loc='lower right'
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        f'ROC_thr{thr}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Empirical precision-recall curve
    # --------------------------------------------------------

    precision, recall, _ = (
        precision_recall_curve(
            y_val,
            y_prob
        )
    )

    prevalence = (
        y_val.mean()
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.plot(
        recall,
        precision,
        linestyle=linestyle,
        color='black',
        linewidth=2,
        label=(
            f'AP = '
            f'{res["ap"]:.4f}'
        )
    )

    ax.axhline(
        prevalence,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label=(
            'No-skill baseline = '
            f'{prevalence:.4f}'
        )
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_xticks(
        np.arange(
            0,
            1.01,
            0.2
        )
    )

    ax.set_yticks(
        np.arange(
            0,
            1.01,
            0.2
        )
    )

    ax.set_xlabel(
        'Recall'
    )

    ax.set_ylabel(
        'Precision'
    )

    ax.set_title(
        f'Precision-Recall Curve - CrAg Target ≥1:{thr}'
    )

    ax.legend(
        loc='lower left'
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        f'PR_thr{thr}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Calibration plot
    # --------------------------------------------------------

    frac_pos, mean_pred = (
        calibration_curve(
            y_val,
            y_prob,
            n_bins=10,
            strategy='uniform'
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.plot(
        mean_pred,
        frac_pos,
        linestyle=linestyle,
        marker=marker,
        color='black',
        linewidth=2,
        markersize=6,
        label='Observed calibration'
    )

    ax.plot(
        [0, 1],
        [0, 1],
        'k--',
        alpha=0.5,
        label='Perfect calibration'
    )

    # Descriptive linear fit through the binned calibration
    # points. This is retained as a visual aid only and is not
    # interpreted as the formal calibration slope.
    valid_mask = (
        ~np.isnan(mean_pred)
        &
        ~np.isnan(frac_pos)
    )

    descriptive_slope = np.nan
    descriptive_intercept = np.nan
    descriptive_r2 = np.nan

    if np.sum(
        valid_mask
    ) > 1:

        descriptive_model = (
            LinearRegression()
        )

        descriptive_model.fit(
            mean_pred[
                valid_mask
            ].reshape(
                -1,
                1
            ),
            frac_pos[
                valid_mask
            ]
        )

        descriptive_slope = (
            descriptive_model
            .coef_[0]
        )

        descriptive_intercept = (
            descriptive_model
            .intercept_
        )

        descriptive_r2 = (
            descriptive_model
            .score(
                mean_pred[
                    valid_mask
                ].reshape(
                    -1,
                    1
                ),
                frac_pos[
                    valid_mask
                ]
            )
        )

        x_line = np.linspace(
            0,
            1,
            200
        )

        y_line = (
            descriptive_model
            .predict(
                x_line.reshape(
                    -1,
                    1
                )
            )
        )

        ax.plot(
            x_line,
            y_line,
            'r--',
            linewidth=2,
            label=(
                'Descriptive linear fit: '
                f'y={descriptive_slope:.2f}x'
                f'{descriptive_intercept:+.2f}, '
                f'R²={descriptive_r2:.3f}'
            )
        )

    textstr = (
        f'Brier Score: '
        f'{res["brier"]:.4f}\n'
        f'Log Loss: '
        f'{res["log_loss"]:.4f}'
    )

    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='wheat',
            alpha=0.5
        )
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_xticks(
        np.arange(
            0,
            1.01,
            0.2
        )
    )

    ax.set_yticks(
        np.arange(
            0,
            1.01,
            0.2
        )
    )

    ax.set_xlabel(
        'Mean Predicted Probability'
    )

    ax.set_ylabel(
        'Fraction of Positives'
    )

    ax.set_title(
        f'Calibration Curve - CrAg Target ≥1:{thr}'
    )

    ax.legend(
        loc='lower right',
        fontsize=9
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        f'Cal_thr{thr}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    res[
        'descriptive_calibration_slope'
    ] = descriptive_slope

    res[
        'descriptive_calibration_intercept'
    ] = descriptive_intercept

    res[
        'descriptive_calibration_r2'
    ] = descriptive_r2

    # --------------------------------------------------------
    # Probability distribution
    # --------------------------------------------------------

    pos_probs = (
        y_prob[
            y_val == 1
        ]
    )

    neg_probs = (
        y_prob[
            y_val == 0
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.hist(
        neg_probs,
        bins=20,
        alpha=0.5,
        density=True,
        label='Negative',
        color='gray'
    )

    ax.hist(
        pos_probs,
        bins=20,
        alpha=0.5,
        density=True,
        label='Positive',
        color='black'
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_xticks(
        np.arange(
            0,
            1.01,
            0.2
        )
    )

    ax.set_xlabel(
        'Predicted Probability'
    )

    ax.set_ylabel(
        'Density'
    )

    ax.set_title(
        f'Probability Distribution - CrAg Target ≥1:{thr}'
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        f'Dist_thr{thr}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # Single-target DCA
    # --------------------------------------------------------

    thresholds_dca = np.linspace(
        0.01,
        0.99,
        99
    )

    model_nb = []

    for p in thresholds_dca:

        y_pred_dca = (
            y_prob >= p
        ).astype(int)

        tp_dca = np.sum(
            (y_pred_dca == 1)
            &
            (y_val == 1)
        )

        fp_dca = np.sum(
            (y_pred_dca == 1)
            &
            (y_val == 0)
        )

        n = len(
            y_val
        )

        nb = (
            tp_dca / n
            -
            fp_dca / n
            *
            p / (1 - p)
        )

        model_nb.append(
            nb
        )

    model_nb = np.asarray(
        model_nb
    )

    best_idx = np.argmax(
        model_nb
    )

    best_probability_threshold = (
        thresholds_dca[
            best_idx
        ]
    )

    best_net_benefit = (
        model_nb[
            best_idx
        ]
    )

    prevalence = (
        y_val.mean()
    )

    treat_all_nb = (
        prevalence
        -
        (1 - prevalence)
        *
        thresholds_dca
        /
        (1 - thresholds_dca)
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.plot(
        thresholds_dca,
        model_nb,
        linestyle=linestyle,
        color='black',
        linewidth=2,
        label='Model'
    )

    ax.plot(
        thresholds_dca,
        treat_all_nb,
        '--',
        color='gray',
        linewidth=1.5,
        label='Treat all'
    )

    ax.axhline(
        0,
        color='black',
        linestyle=':',
        linewidth=1.5,
        label='Treat none'
    )

    # Preserve the original red maximum-NB marker in the
    # single-target DCA plot.
    ax.plot(
        best_probability_threshold,
        best_net_benefit,
        marker='o',
        markersize=8,
        color='red',
        linestyle='None',
        label=(
            f'Max NB={best_net_benefit:.4f} '
            f'at p={best_probability_threshold:.2f}'
        )
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        -0.05,
        0.30
    )

    ax.set_xticks(
        np.arange(
            0,
            1.01,
            0.2
        )
    )

    ax.set_xlabel(
        'Threshold Probability'
    )

    ax.set_ylabel(
        'Net Benefit'
    )

    ax.set_title(
        f'Decision Curve Analysis - CrAg Target ≥1:{thr}'
    )

    ax.legend(
        loc='best'
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        f'DCA_thr{thr}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    res[
        'maximum_net_benefit'
    ] = best_net_benefit

    res[
        'maximum_nb_probability_threshold'
    ] = best_probability_threshold


# ============================================================
# Feature importance
# ============================================================

def plot_feature_importance(
    thr,
    res,
    top_n=20
):

    classifier = (
        res['model']
        .named_steps['clf']
    )

    importances = (
        classifier
        .feature_importances_
    )

    feature_names = (
        res[
            'feature_names'
        ]
    )

    feature_importance = sorted(
        zip(
            feature_names,
            importances
        ),
        key=lambda x:
        x[1],
        reverse=True
    )

    feature_importance = (
        feature_importance[
            :top_n
        ]
    )

    names, values = zip(
        *feature_importance
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    y_position = np.arange(
        len(names)
    )

    ax.barh(
        y_position,
        values,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_yticks(
        y_position
    )

    ax.set_yticklabels(
        names,
        fontsize=10
    )

    ax.set_xlabel(
        'Importance'
    )

    ax.set_title(
        f'Feature Importance - CrAg Target ≥1:{thr}'
    )

    ax.invert_yaxis()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        f'FI_thr{thr}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )


# ============================================================
# Generate individual-target figures
# ============================================================

for idx, (
    thr,
    res
) in enumerate(
    results.items()
):

    plot_single_curves(
        thr,
        res,
        idx
    )

    plot_feature_importance(
        thr,
        res,
        top_n=20
    )


# ============================================================
# Summary figures across CrAg titer targets
# ============================================================

def plot_summary_curves(
    curve_type
):

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    # --------------------------------------------------------
    # Summary ROC
    # --------------------------------------------------------

    if curve_type == 'roc':

        for idx, (
            thr,
            res
        ) in enumerate(
            results.items()
        ):

            linestyle = (
                LINE_STYLES[
                    idx %
                    len(LINE_STYLES)
                ]
            )

            fpr, tpr, _ = (
                roc_curve(
                    res[
                        'y_val'
                    ],
                    res[
                        'y_prob'
                    ]
                )
            )

            ax.plot(
                fpr,
                tpr,
                linestyle=linestyle,
                color='black',
                linewidth=1.8,
                label=(
                    f'≥1:{thr} '
                    f'(AUROC='
                    f'{res["auc"]:.4f})'
                )
            )

        ax.plot(
            [0, 1],
            [0, 1],
            'k--',
            alpha=0.5,
            label='Random'
        )

        ax.set_xlim(
            0,
            1
        )

        ax.set_ylim(
            0,
            1
        )

        ax.set_xticks(
            np.arange(
                0,
                1.01,
                0.2
            )
        )

        ax.set_yticks(
            np.arange(
                0,
                1.01,
                0.2
            )
        )

        ax.set_xlabel(
            'False Positive Rate'
        )

        ax.set_ylabel(
            'True Positive Rate'
        )

        ax.set_title(
            'ROC Curves Across CrAg Titer Targets'
        )

        ax.legend(
            loc='lower right',
            fontsize=9
        )

    # --------------------------------------------------------
    # Summary PR
    # --------------------------------------------------------

    elif curve_type == 'pr':

        for idx, (
            thr,
            res
        ) in enumerate(
            results.items()
        ):

            linestyle = (
                LINE_STYLES[
                    idx %
                    len(LINE_STYLES)
                ]
            )

            precision, recall, _ = (
                precision_recall_curve(
                    res[
                        'y_val'
                    ],
                    res[
                        'y_prob'
                    ]
                )
            )

            ax.plot(
                recall,
                precision,
                linestyle=linestyle,
                color='black',
                linewidth=1.8,
                label=(
                    f'≥1:{thr} '
                    f'(AP='
                    f'{res["ap"]:.4f})'
                )
            )

        ax.set_xlim(
            0,
            1
        )

        ax.set_ylim(
            0,
            1
        )

        ax.set_xticks(
            np.arange(
                0,
                1.01,
                0.2
            )
        )

        ax.set_yticks(
            np.arange(
                0,
                1.01,
                0.2
            )
        )

        ax.set_xlabel(
            'Recall'
        )

        ax.set_ylabel(
            'Precision'
        )

        ax.set_title(
            'Precision-Recall Curves Across CrAg Titer Targets'
        )

        ax.legend(
            loc='lower left',
            fontsize=9
        )

    # --------------------------------------------------------
    # Summary calibration
    # --------------------------------------------------------

    elif curve_type == 'cal':

        for idx, (
            thr,
            res
        ) in enumerate(
            results.items()
        ):

            linestyle = (
                LINE_STYLES[
                    idx %
                    len(LINE_STYLES)
                ]
            )

            frac_pos, mean_pred = (
                calibration_curve(
                    res[
                        'y_val'
                    ],
                    res[
                        'y_prob'
                    ],
                    n_bins=10,
                    strategy='uniform'
                )
            )

            ax.plot(
                mean_pred,
                frac_pos,
                linestyle=linestyle,
                color='black',
                linewidth=1.8,
                label=(
                    f'≥1:{thr} '
                    f'(Brier='
                    f'{res["brier"]:.4f})'
                )
            )

        ax.plot(
            [0, 1],
            [0, 1],
            'k--',
            alpha=0.5,
            label='Perfect calibration'
        )

        ax.set_xlim(
            0,
            1
        )

        ax.set_ylim(
            0,
            1
        )

        ax.set_xticks(
            np.arange(
                0,
                1.01,
                0.2
            )
        )

        ax.set_yticks(
            np.arange(
                0,
                1.01,
                0.2
            )
        )

        ax.set_xlabel(
            'Mean Predicted Probability'
        )

        ax.set_ylabel(
            'Fraction of Positives'
        )

        ax.set_title(
            'Calibration Curves Across CrAg Titer Targets'
        )

        ax.legend(
            loc='lower right',
            fontsize=9
        )

    # --------------------------------------------------------
    # Summary DCA — Figure 6a
    # --------------------------------------------------------

    elif curve_type == 'dca':

        thresholds_dca = np.linspace(
            0.01,
            0.99,
            99
        )

        maximum_nb_info = []

        # ----------------------------------------------------
        # Treat-none strategy
        # ----------------------------------------------------

        ax.axhline(
            0,
            color='black',
            linestyle=':',
            linewidth=1.5,
            label='Treat none'
        )

        # ----------------------------------------------------
        # Target-specific treat-all curves
        # ----------------------------------------------------
        # Each CrAg titer target has its own observed event
        # prevalence, so each outcome requires a corresponding
        # target-specific treat-all curve.
        # ----------------------------------------------------

        for idx, (
            thr,
            res
        ) in enumerate(
            results.items()
        ):

            y_val = np.asarray(
                res[
                    'y_val'
                ]
            )

            prevalence = (
                y_val.mean()
            )

            linestyle = (
                LINE_STYLES[
                    idx %
                    len(LINE_STYLES)
                ]
            )

            treat_all_nb = (
                prevalence
                -
                (1 - prevalence)
                *
                thresholds_dca
                /
                (1 - thresholds_dca)
            )

            ax.plot(
                thresholds_dca,
                treat_all_nb,
                linestyle=linestyle,
                color='gray',
                alpha=0.55,
                linewidth=1.3,
                label=(
                    'Treat all '
                    '(target-specific)'
                    if idx == 0
                    else '_nolegend_'
                )
            )

        # ----------------------------------------------------
        # Model net-benefit curves
        # ----------------------------------------------------

        for idx, (
            thr,
            res
        ) in enumerate(
            results.items()
        ):

            y_val = np.asarray(
                res[
                    'y_val'
                ]
            )

            y_prob = np.asarray(
                res[
                    'y_prob'
                ]
            )

            linestyle = (
                LINE_STYLES[
                    idx %
                    len(LINE_STYLES)
                ]
            )

            model_nb = []

            for p in thresholds_dca:

                y_pred_dca = (
                    y_prob >= p
                ).astype(int)

                tp_dca = np.sum(
                    (
                        y_pred_dca == 1
                    )
                    &
                    (
                        y_val == 1
                    )
                )

                fp_dca = np.sum(
                    (
                        y_pred_dca == 1
                    )
                    &
                    (
                        y_val == 0
                    )
                )

                n = len(
                    y_val
                )

                nb = (
                    tp_dca / n
                    -
                    fp_dca / n
                    *
                    p / (1 - p)
                )

                model_nb.append(
                    nb
                )

            model_nb = np.asarray(
                model_nb
            )

            best_idx = np.argmax(
                model_nb
            )

            best_nb = (
                model_nb[
                    best_idx
                ]
            )

            best_probability_threshold = (
                thresholds_dca[
                    best_idx
                ]
            )

            res[
                'maximum_net_benefit'
            ] = best_nb

            res[
                'maximum_nb_probability_threshold'
            ] = (
                best_probability_threshold
            )

            maximum_nb_info.append(
                (
                    thr,
                    best_nb,
                    best_probability_threshold
                )
            )

            # Left-side legend identifies model curves only.
            # Maximum net-benefit values are reported separately
            # in the upper-right inset to avoid duplication.
            ax.plot(
                thresholds_dca,
                model_nb,
                linestyle=linestyle,
                color='black',
                linewidth=1.8,
                label=f'≥1:{thr}'
            )

        # ----------------------------------------------------
        # Upper-right maximum-NB information box
        # ----------------------------------------------------

        inset_lines = [
            'Maximum net benefit'
        ]

        for (
            thr,
            best_nb,
            best_probability_threshold
        ) in maximum_nb_info:

            inset_lines.append(
                f'≥1:{thr}: '
                f'{best_nb:.4f} '
                f'(p={best_probability_threshold:.2f})'
            )

        inset_text = '\n'.join(
            inset_lines
        )

        ax.text(
            0.03,
            0.97,
            inset_text,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                edgecolor='black',
                alpha=0.90
            )
        )

        ax.set_xlim(
            0,
            1
        )

        ax.set_ylim(
            -0.05,
            0.30
        )

        ax.set_xticks(
            np.arange(
                0,
                1.01,
                0.2
            )
        )

        ax.set_xlabel(
            'Threshold Probability'
        )

        ax.set_ylabel(
            'Net Benefit'
        )

        ax.set_title(
            'Decision Curve Analysis Across CrAg Titer Targets'
        )

        # The legend explains line identity and reference
        # strategies only; it does not repeat maximum-NB values.
        ax.legend(
            loc='lower right',
            fontsize=9,
            ncol=2
        )

    else:

        raise ValueError(
            "curve_type must be one of "
            "'roc', 'pr', 'cal', or 'dca'."
        )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        f'Summary_{curve_type}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )


# ============================================================
# Generate summary figures
# ============================================================

for curve_type in [
    'roc',
    'pr',
    'cal',
    'dca'
]:

    plot_summary_curves(
        curve_type
    )


# ============================================================
# Save models
# ============================================================
# These saved models correspond to the 80% training partitions
# and are not refitted on the complete 2024 development dataset.
# ============================================================

for thr, res in results.items():

    model_path = (
        MODEL_DIR /
        f'model_thr{thr}.pkl'
    )

    joblib.dump(
        res[
            'model'
        ],
        model_path
    )

    print(
        f'CrAg ≥1:{thr} model saved to: '
        f'{model_path.resolve()}'
    )


# ============================================================
# Save metadata
# ============================================================

metadata = {

    'num_cols':
        num_cols,

    'ohe_cols':
        ohe_cols,

    'ord_cols':
        ord_cols,

    'feature_cols':
        feature_cols,

    'thresholds':
        thresholds,

    'best_params_by_thr':
        best_params_by_thr,

    'split_ratio':
        '80% training / 20% internal validation',

    'split_random_state':
        42,

    'smote_random_state':
        42,

    'model_random_state':
        42,

    'primary_clinical_target':
        'CSF CrAg titer >= 1:80',

    'modeling_scheme':
        (
            'Scheme A: models fitted on the corresponding '
            '80% training partitions and retained without '
            'full-development refitting.'
        ),

    'classification_rule_for_internal_validation':
        (
            'pipeline.predict(), retained to reproduce the '
            'original Random Forest classification behavior.'
        ),

    'dca_probability_range':
        '0.01 to 0.99',

    'calibration_note':
        (
            'Red dashed linear fits through binned calibration '
            'points are descriptive visual aids only. Formal '
            'calibration intercepts and slopes are estimated '
            'using individual-level logistic recalibration.'
        )
}

joblib.dump(
    metadata,
    MODEL_DIR /
    'EFL_data.pkl'
)


# ============================================================
# Save final metrics after all plots have been generated
# ============================================================

final_rows = []

for thr, res in results.items():

    (
        tn,
        fp,
        fn,
        tp
    ) = res[
        'confusion'
    ]

    final_rows.append({

        'CrAg_titer_target':
            f'>=1:{thr}',

        'Sensitivity':
            res[
                'sensitivity'
            ],

        'Specificity':
            res[
                'specificity'
            ],

        'PPV':
            res[
                'ppv'
            ],

        'NPV':
            res[
                'npv'
            ],

        'Accuracy':
            res[
                'accuracy'
            ],

        'F1':
            res[
                'f1'
            ],

        'AUROC':
            res[
                'auc'
            ],

        'Average_precision':
            res[
                'ap'
            ],

        'Brier_score':
            res[
                'brier'
            ],

        'Log_loss':
            res[
                'log_loss'
            ],

        'Youden_index':
            res[
                'youden_index'
            ],

        'Formal_calibration_intercept':
            res[
                'formal_calibration_intercept'
            ],

        'Formal_calibration_slope':
            res[
                'formal_calibration_slope'
            ],

        'Descriptive_binned_linear_fit_slope':
            res.get(
                'descriptive_calibration_slope',
                np.nan
            ),

        'Descriptive_binned_linear_fit_intercept':
            res.get(
                'descriptive_calibration_intercept',
                np.nan
            ),

        'Descriptive_binned_linear_fit_R2':
            res.get(
                'descriptive_calibration_r2',
                np.nan
            ),

        'Maximum_DCA_net_benefit':
            res.get(
                'maximum_net_benefit',
                np.nan
            ),

        'Probability_threshold_at_maximum_DCA_net_benefit':
            res.get(
                'maximum_nb_probability_threshold',
                np.nan
            ),

        'TN':
            tn,

        'FP':
            fp,

        'FN':
            fn,

        'TP':
            tp
    })


final_metrics_df = pd.DataFrame(
    final_rows
)

final_metrics_df.to_csv(
    RESULT_DIR /
    'CrAg_multithreshold_final_metrics.csv',
    index=False,
    encoding='utf-8-sig'
)


# ============================================================
# Completion message
# ============================================================

print(
    '\n' +
    '=' * 80
)

print(
    'Development-stage analysis complete.'
)

print(
    '=' * 80
)

print(
    f'Figures saved to: '
    f'{FIGURE_DIR.resolve()}'
)

print(
    f'Models saved to: '
    f'{MODEL_DIR.resolve()}'
)

print(
    f'Metrics saved to: '
    f'{RESULT_DIR.resolve()}'
)

print(
    '\nAll figures and model outputs were '
    'generated successfully.'
)