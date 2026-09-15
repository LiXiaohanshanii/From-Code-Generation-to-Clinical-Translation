"""
Statistical analysis for the 200-run TRUST LLM code-generation benchmark.

Manuscript:
    From Code Generation to Clinical Translation: Evaluating LLM-Assisted
    Clinical Machine Learning in Laboratory Medicine

Purpose
-------
This script reproduces the benchmark-level statistical analyses for:
    1. continuous code-generation outcomes (Token, Pylint, Code_lines,
       Comment_density);
    2. technical executability (Pass@1);
    3. methodological adequacy (MAR).

Design
------
Factorial design: Prompt × Mode × LLM.
Continuous outcomes are analyzed using three-way factorial ANOVA with
sum-to-zero contrasts and Type III sums of squares. Benjamini-Hochberg FDR
correction is applied within each continuous outcome across the seven tested
ANOVA effects (three main effects, three two-way interactions, and one
three-way interaction; intercept excluded).

Binary outcomes are analyzed without relying on unstable high-order logistic
maximum-likelihood inference. The overall Thinking-versus-Fast association is
quantified using a Mantel-Haenszel common odds ratio stratified by Prompt × LLM,
with a 95% confidence interval and Cochran-Mantel-Haenszel test. Two-sided Fisher
exact tests are additionally used for Thinking-versus-Fast simple effects within
each Prompt × LLM stratum, with Benjamini-Hochberg FDR correction within each
binary outcome. A saturated factorial logistic model is attempted only as a
diagnostic/sensitivity analysis; if it fails to converge, an L2-penalized model
is saved descriptively and is not used for inferential p-values. Because Pass@1
and MAR are paired outcomes measured on the same generated scripts, their overall
rates are compared using the exact McNemar test.

Input
-----
Expected file: data.csv (change INPUT_FILE below if needed)
Required columns:
    Prompt, Mode, LLM, Pass@1, MAR,
    Token, Pylint, Code_lines, Comment_density

Outputs
-------
All outputs are written to ./results/ and include conventional and HC3-robust
Type III ANOVA tables, Mantel-Haenszel adjusted odds ratios, exact McNemar and
Fisher-test tables, descriptive Wilson intervals, post-hoc tables, and
diagnostic/summary figures.

Important
---------
Patient-level clinical data are not used by this script and should not be
included in the public repository.
"""

from __future__ import annotations

import re
import warnings
from itertools import combinations
from pathlib import Path

import chardet
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact, kruskal, levene, norm, rankdata
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import StratifiedTable, mcnemar

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_FILE = Path("200_runs_data.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTINUOUS_VARS = ["Token", "Pylint", "Code_lines", "Comment_density"]
BINARY_VARS = ["Pass@1", "MAR"]
FACTORS = ["Prompt", "Mode", "LLM"]
MODE_ORDER = ["Fast", "Thinking"]
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV file with simple encoding detection and conservative fallbacks."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    raw = path.read_bytes()
    detected = chardet.detect(raw).get("encoding")
    candidates = [detected, "utf-8-sig", "utf-8", "gbk", "gb2312", "cp936", "latin-1"]

    tried = set()
    last_error: Exception | None = None
    for encoding in candidates:
        if not encoding or encoding.lower() in tried:
            continue
        tried.add(encoding.lower())
        try:
            print(f"Reading {path} using encoding: {encoding}")
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown", b"", 0, 1,
        f"Unable to decode {path}. Last error: {last_error}"
    )

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize benchmark column names without changing analytical values."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)

    rename_dict: dict[str, str] = {}
    for col in df.columns:
        if "模板" in col:
            rename_dict[col] = "Prompt"
        elif "模式" in col:
            rename_dict[col] = "Mode"
        elif "LLMs" in col or "模型" in col:
            rename_dict[col] = "LLM"
        elif "Methodology" in col:
            rename_dict[col] = "MAR"
    return df.rename(columns=rename_dict)

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields and factor/outcome coding."""
    required = FACTORS + BINARY_VARS + CONTINUOUS_VARS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Mode"] = df["Mode"].map({"快速": "Fast", "思考": "Thinking"}).fillna(df["Mode"])

    unexpected_modes = sorted(set(df["Mode"].dropna()) - set(MODE_ORDER))
    if unexpected_modes:
        raise ValueError(f"Unexpected Mode values: {unexpected_modes}")

    for col in BINARY_VARS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
        unexpected = sorted(set(df[col].dropna()) - {0, 1})
        if unexpected:
            raise ValueError(f"{col} must be coded 0/1; found: {unexpected}")

    for col in CONTINUOUS_VARS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["Prompt"] = df["Prompt"].astype("category")
    df["Mode"] = pd.Categorical(df["Mode"], categories=MODE_ORDER, ordered=True)
    df["LLM"] = df["LLM"].astype("category")

    if df[FACTORS + BINARY_VARS + CONTINUOUS_VARS].isna().any().any():
        print("Warning: missing values are present; statsmodels/scipy will use available observations as applicable.")

    print(f"Data shape: {df.shape}")
    print(df.head())
    return df

def bh_fdr(p_values: np.ndarray, include_mask: np.ndarray | None = None) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, optionally on a selected subset."""
    p_values = np.asarray(p_values, dtype=float)
    if include_mask is None:
        include_mask = np.ones(len(p_values), dtype=bool)
    include_mask = np.asarray(include_mask, dtype=bool) & ~np.isnan(p_values)

    corrected = np.full(len(p_values), np.nan, dtype=float)
    if include_mask.any():
        _, adjusted, _, _ = multipletests(p_values[include_mask], method="fdr_bh")
        corrected[include_mask] = adjusted
    return corrected

def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

# -----------------------------------------------------------------------------
# 1. Load and validate benchmark data
# -----------------------------------------------------------------------------
df = validate_data(normalize_columns(read_csv_robust(INPUT_FILE)))

# -----------------------------------------------------------------------------
# 2. Continuous outcomes: three-way factorial ANOVA
# -----------------------------------------------------------------------------
# Sum contrasts are used because Type III tests are requested for the factorial design
# FDR correction is performed separately within each outcome across the seven non-intercept ANOVA effects

ANOVA_FORMULA = (
    "{outcome} ~ C(Prompt, Sum) + C(Mode, Sum) + C(LLM, Sum) + "
    "C(Prompt, Sum):C(Mode, Sum) + "
    "C(Prompt, Sum):C(LLM, Sum) + "
    "C(Mode, Sum):C(LLM, Sum) + "
    "C(Prompt, Sum):C(Mode, Sum):C(LLM, Sum)"
)

TWO_WAY_PATTERN = re.compile(r"^C\([^)]+\):C\([^)]+\)$")
THREE_WAY_PATTERN = re.compile(r"^C\([^)]+\):C\([^)]+\):C\([^)]+\)$")

anova_results: dict[str, pd.DataFrame] = {}
interaction_p_fdr: dict[str, dict[str, float]] = {}
three_way_p_fdr: dict[str, float] = {}

for outcome in CONTINUOUS_VARS:
    print(f"\n=== Type III three-way factorial ANOVA: {outcome} ===")
    model = ols(ANOVA_FORMULA.format(outcome=outcome), data=df).fit()
    table = anova_lm(model, typ=3)

    is_non_intercept = table.index.to_numpy() != "Intercept"
    table["p_fdr"] = bh_fdr(table["PR(>F)"].to_numpy(), is_non_intercept)
    anova_results[outcome] = table
    table.to_csv(OUTPUT_DIR / f"ANOVA_{outcome}.csv")
    print(table.round(4))

# HC3 heteroskedasticity-robust Type III ANOVA. When Levene's test indicates heteroskedasticity
# this robust table becomes the primary inferential table

    table_hc3 = None
    try:
        table_hc3 = anova_lm(model, typ=3, robust="hc3")
        table_hc3["p_fdr"] = bh_fdr(table_hc3["PR(>F)"].to_numpy(), is_non_intercept)
        table_hc3.to_csv(OUTPUT_DIR / f"ANOVA_HC3_{outcome}.csv")
        print(f"\nHC3-robust Type III ANOVA: {outcome}")
        print(table_hc3.round(4))
    except Exception as exc:
        save_text(OUTPUT_DIR / f"ANOVA_HC3_{outcome}_FAILED.txt", str(exc))
        print(f"HC3 robust ANOVA unavailable for {outcome}: {exc}")

    interaction_p_fdr[outcome] = {
        term: float(table.loc[term, "p_fdr"])
        for term in table.index
        if TWO_WAY_PATTERN.match(term)
    }
    three_terms = [term for term in table.index if THREE_WAY_PATTERN.match(term)]
    three_way_p_fdr[outcome] = (
        float(table.loc[three_terms[0], "p_fdr"]) if three_terms else np.nan
    )

# Levene test across all Prompt × Mode × LLM cells.

    levene_p = np.nan
    groups = [
        group[outcome].dropna().to_numpy()
        for _, group in df.groupby(FACTORS, observed=True)
    ]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        stat, p_value = levene(*groups)
        levene_p = float(p_value)
        print(f"Levene test: W={stat:.4f}, p={p_value:.6g}")
        pd.DataFrame({"Statistic": [stat], "P_value": [p_value]}).to_csv(
            OUTPUT_DIR / f"Levene_{outcome}.csv", index=False
        )

# Select the inferential table used to trigger follow-up analyses and annotate figures
# HC3 is primary whenever Levene p < 0.05
# conventional Type III ANOVA is retained
# Both tables are always saved when available

    primary_table = table_hc3 if (table_hc3 is not None and not np.isnan(levene_p) and levene_p < 0.05) else table
    primary_label = "HC3" if primary_table is table_hc3 else "Conventional"
    anova_results[outcome] = primary_table
    interaction_p_fdr[outcome] = {
        term: float(primary_table.loc[term, "p_fdr"])
        for term in primary_table.index
        if TWO_WAY_PATTERN.match(term)
    }
    three_terms_primary = [term for term in primary_table.index if THREE_WAY_PATTERN.match(term)]
    three_way_p_fdr[outcome] = (
        float(primary_table.loc[three_terms_primary[0], "p_fdr"])
        if three_terms_primary else np.nan
    )
    pd.DataFrame({
        "Outcome": [outcome],
        "Primary_inference": [primary_label],
        "Levene_p": [levene_p],
    }).to_csv(OUTPUT_DIR / f"PrimaryInference_{outcome}.csv", index=False)
    print(f"Primary inferential table for {outcome}: {primary_label}")

# -----------------------------------------------------------------------------
# 3. Binary outcomes: factorial logistic model + stratified Fisher tests
# -----------------------------------------------------------------------------
def fit_factorial_logistic(data: pd.DataFrame, response: str) -> None:
    """
    Fit Prompt × Mode × LLM logistic model
    If the standard maximum-likelihood model does not converge, an L2-penalized
    model is saved only as a stability/sensitivity output
    Penalized coefficients are not treated as standard inferential estimates and
    no p-values are reported for that fallback model
    """
    safe_response = response.replace("@", "_at_")
    working = data.copy()
    working[safe_response] = working[response]
    formula = f"{safe_response} ~ C(Prompt, Sum) * C(Mode, Sum) * C(LLM, Sum)"

    try:
        model = smf.logit(formula, data=working).fit(disp=0, maxiter=200)
        if not bool(model.mle_retvals.get("converged", False)):
            raise RuntimeError("Maximum-likelihood logistic model did not converge.")

        params = model.params
        conf = model.conf_int()
        results = pd.DataFrame(
            {
                "Coefficient": params,
                "OR": np.exp(params),
                "CI95_low": np.exp(conf[0]),
                "CI95_high": np.exp(conf[1]),
                "P_value": model.pvalues,
            }
        )
        results.index.name = "Term"
        results.to_csv(OUTPUT_DIR / f"Logistic_{response}_Standard.csv")
        save_text(OUTPUT_DIR / f"Logistic_{response}_Standard.txt", str(model.summary()))
        print(f"{response}: standard logistic model converged.")
        return

    except Exception as exc:
        print(f"{response}: standard logistic model unavailable ({exc}).")
        print("Fitting L2-penalized fallback for stability only; no inferential p-values will be reported.")

    try:
        model_reg = smf.glm(
            formula, data=working, family=sm.families.Binomial()
        ).fit_regularized(method="elastic_net", alpha=0.1, L1_wt=0.0, maxiter=500)

        results = pd.DataFrame(
            {
                "Coefficient": model_reg.params,
                "Penalized_OR": np.exp(model_reg.params),
                "Inferential_P_value": np.nan,
            }
        )
        results.index.name = "Term"
        results.to_csv(OUTPUT_DIR / f"Logistic_{response}_L2_sensitivity_only.csv")
        save_text(
            OUTPUT_DIR / f"Logistic_{response}_L2_sensitivity_only.txt",
            "L2-penalized logistic regression fallback.\n"
            "These coefficients are shrinkage estimates and are not used for standard inferential p-values.\n",
        )
    except Exception as exc:
        save_text(
            OUTPUT_DIR / f"Logistic_{response}_FAILED.txt",
            f"Both standard and penalized logistic models failed.\n{exc}\n",
        )
        print(f"{response}: penalized fallback also failed ({exc}).")

def mantel_haenszel_mode_effect(data: pd.DataFrame, response: str) -> pd.DataFrame:
    """
    Estimate the common Thinking-versus-Fast odds ratio across Prompt × LLM strata
    Each 2×2 table is oriented as:
        rows = [Thinking, Fast]
        columns = [success = 1, failure = 0]
    StratifiedTable applies a 0.5 continuity correction to strata containing zero cells,
    which is important for ceiling outcomes such as Pass@1.
    The resulting common OR is therefore a Mantel-Haenszel estimate adjusted
    for Prompt × LLM strata, not a coefficient from the saturated logistic model.
    """
    tables: list[np.ndarray] = []
    stratum_labels: list[str] = []

    for prompt in data["Prompt"].dropna().unique():
        for llm in data["LLM"].dropna().unique():
            sub = data[(data["Prompt"] == prompt) & (data["LLM"] == llm)]
            rows = []
            valid = True
            for mode in ["Thinking", "Fast"]:
                vals = sub.loc[sub["Mode"] == mode, response]
                success = int((vals == 1).sum())
                failure = int((vals == 0).sum())
                if success + failure == 0:
                    valid = False
                    break
                rows.append([success, failure])
            if valid:
                tables.append(np.asarray(rows, dtype=float))
                stratum_labels.append(f"Prompt={prompt}|LLM={llm}")

    if not tables:
        return pd.DataFrame()

    st = StratifiedTable(np.stack(tables, axis=2), shift_zeros=True)
    mh_test = st.test_null_odds(correction=False)
    ci_low, ci_high = st.oddsratio_pooled_confint(alpha=0.05)

    out = pd.DataFrame({
        "Response": [response],
        "Comparison": ["Thinking vs Fast"],
        "Stratification": ["Prompt × LLM"],
        "N_strata": [len(tables)],
        "MH_common_OR": [float(st.oddsratio_pooled)],
        "CI95_low": [float(ci_low)],
        "CI95_high": [float(ci_high)],
        "CMH_statistic": [float(mh_test.statistic)],
        "CMH_p_value": [float(mh_test.pvalue)],
        "Zero_cell_correction": ["0.5 within zero-cell strata"],
    })
    out.to_csv(OUTPUT_DIR / f"MantelHaenszel_Mode_{response}.csv", index=False)
    save_text(
        OUTPUT_DIR / f"MantelHaenszel_Mode_{response}_strata.txt",
        "\n".join(stratum_labels) + "\n",
    )
    return out

def exact_mcnemar_pass_vs_mar(data: pd.DataFrame) -> pd.DataFrame:
    """Exact paired comparison of Pass@1 and MAR on the same generated scripts."""
    paired = data[["Pass@1", "MAR"]].dropna().astype(int)
    table = pd.crosstab(paired["Pass@1"], paired["MAR"]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    arr = table.to_numpy(dtype=int)
    result = mcnemar(arr, exact=True)
    out = pd.DataFrame({
        "N_paired": [int(arr.sum())],
        "Pass0_MAR0": [int(arr[0, 0])],
        "Pass0_MAR1": [int(arr[0, 1])],
        "Pass1_MAR0": [int(arr[1, 0])],
        "Pass1_MAR1": [int(arr[1, 1])],
        "Discordant_Pass0_MAR1": [int(arr[0, 1])],
        "Discordant_Pass1_MAR0": [int(arr[1, 0])],
        "Exact_McNemar_statistic": [float(result.statistic)],
        "Exact_McNemar_p_value": [float(result.pvalue)],
    })
    out.to_csv(OUTPUT_DIR / "McNemar_PassAt1_vs_MAR.csv", index=False)
    return out

def descriptive_binary_rates(data: pd.DataFrame, response: str) -> pd.DataFrame:
    """Aggregate binary outcome by LLM and Mode and calculate Wilson 95% CIs."""
    grouped = (
        data.groupby(["LLM", "Mode"], observed=True)[response]
        .agg(rate="mean", n="count", success="sum")
        .reset_index()
    )
    low, high = proportion_confint(
        count=grouped["success"], nobs=grouped["n"], alpha=0.05, method="wilson"
    )
    grouped["ci95_low"] = low
    grouped["ci95_high"] = high
    grouped.to_csv(OUTPUT_DIR / f"Descriptive_{response}_by_LLM_Mode.csv", index=False)
    return grouped

def fisher_mode_simple_effects(data: pd.DataFrame, response: str) -> pd.DataFrame:
    """
    Compare Thinking versus Fast within each Prompt × LLM stratum.
    Table orientation is explicitly:
        rows = [Thinking, Fast]
        columns = [success = 1, failure= 0]
    Therefore the Fisher odds ratio is Thinking versus Fast.
    """
    rows: list[dict[str, object]] = []

    for prompt in data["Prompt"].dropna().unique():
        for llm in data["LLM"].dropna().unique():
            sub = data[(data["Prompt"] == prompt) & (data["LLM"] == llm)]

            counts: dict[str, tuple[int, int]] = {}
            for mode in MODE_ORDER:
                mode_values = sub.loc[sub["Mode"] == mode, response]
                success = int((mode_values == 1).sum())
                failure = int((mode_values == 0).sum())
                counts[mode] = (success, failure)

            fast_success, fast_failure = counts["Fast"]
            thinking_success, thinking_failure = counts["Thinking"]

            if (fast_success + fast_failure == 0) or (thinking_success + thinking_failure == 0):
                continue

            table = np.array(
                [
                    [thinking_success, thinking_failure],
                    [fast_success, fast_failure],
                ],
                dtype=int,
            )
            odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

            rows.append(
                {
                    "Prompt": str(prompt),
                    "LLM": str(llm),
                    "Comparison": "Thinking vs Fast",
                    "Thinking_success": thinking_success,
                    "Thinking_failure": thinking_failure,
                    "Fast_success": fast_success,
                    "Fast_failure": fast_failure,
                    "OR_Thinking_vs_Fast": odds_ratio,
                    "p_raw": p_value,
                    "N": int(table.sum()),
                }
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        results["p_fdr"] = bh_fdr(results["p_raw"].to_numpy())
    results.to_csv(OUTPUT_DIR / f"Fisher_Mode_SimpleEffects_{response}.csv", index=False)
    return results

print("\n=== Paired comparison: Pass@1 vs MAR ===")
mcnemar_results = exact_mcnemar_pass_vs_mar(df)
print(mcnemar_results.round(6))

for response in BINARY_VARS:
    print(f"\n=== Binary outcome: {response} ===")

# Primary adjusted Mode analysis: common OR across Prompt × LLM strata.

    mh_results = mantel_haenszel_mode_effect(df, response)
    print("Mantel-Haenszel adjusted Mode effect:")
    print(mh_results.round(6) if not mh_results.empty else "No MH analysis available.")

# Prespecified stratum-specific exact simple effects with BH-FDR correction.

    fisher_results = fisher_mode_simple_effects(df, response)
    print("Stratified Fisher exact tests:")
    print(fisher_results.round(4) if not fisher_results.empty else "No Fisher tests available.")

# Descriptive rates and Wilson confidence intervals.

    grouped_binary = descriptive_binary_rates(df, response)

# Diagnostic/sensitivity only: saturated factorial logistic model.
# Its inferential output is not used if MLE does not converge.

    fit_factorial_logistic(df, response)

#-----------------------------------------------------------------------------
# 4. Manuscript figures retained from this analysis script
# -----------------------------------------------------------------------------
# Only figures that are explicitly used in the manuscript/supplement are generated:
#   1) Figure 2c: Token by LLM and reasoning Mode (mean ± 1 SD).
#   2) Supplementary Figure 1: secondary engineering metrics.
# Diagnostic residual plots, binary-rate plots, and redundant per-outcome
# boxplots/interaction plots are intentionally not generated.

def plot_mean_sd_interaction(data: pd.DataFrame, outcome: str, ax, title: str) -> None:
    """Plot mean ± 1 SD for Fast and Thinking modes across LLMs."""
    summary = (
        data.groupby(["LLM", "Mode"], observed=True)[outcome]
        .agg(mean="mean", sd="std")
        .reset_index()
    )
    models = list(data["LLM"].cat.categories if hasattr(data["LLM"], "cat") else data["LLM"].unique())
    x = np.arange(len(models), dtype=float)
    for mode in MODE_ORDER:
        sub = summary[summary["Mode"] == mode].set_index("LLM").reindex(models)
        ax.errorbar(
            x,
            sub["mean"].to_numpy(dtype=float),
            yerr=sub["sd"].to_numpy(dtype=float),
            marker="o",
            capsize=4,
            label=mode,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha="right")
    ax.set_xlabel("LLMs")
    ax.set_ylabel(outcome)
    ax.set_title(title,fontweight='bold',fontsize=14)
    ax.legend(title="Mode")

# Figure 2c: Token interaction plot. Error bars are ±1 SD to match the manuscript caption.

fig, ax = plt.subplots(figsize=(9, 8))
plot_mean_sd_interaction(
    df,
    "Token",
    ax,
    "Token consumption by LLM and reasoning mode",
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "Figure2c_Token_Mode_x_LLM.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Supplementary Figure 1: concise visualization of the three secondary engineering metrics.
# a: Pylint by LLM and Mode (main effects; no significant Mode × LLM interaction).
# b: Code lines, Prompt × LLM interaction.
# c: Code lines, Mode × LLM interaction.
# d: Comment density, Mode × LLM interaction.

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.boxplot(data=df, x="LLM", y="Pylint", hue="Mode", ax=axes[0, 0])
axes[0, 0].set_title("A. Pylint score by LLM and mode")
axes[0, 0].set_xlabel("LLMs")
axes[0, 0].set_ylabel("Pylint score")
axes[0, 0].tick_params(axis="x", rotation=0)
axes[0, 0].legend(title="Mode")

sns.pointplot(
    data=df,
    x="LLM",
    y="Code_lines",
    hue="Prompt",
    errorbar=("ci", 95),
    capsize=0.2,
    dodge=0.3,
    seed=RANDOM_SEED,
    ax=axes[0, 1],
)
axes[0, 1].set_title("B. Code lines: Prompt × LLM")
axes[0, 1].set_xlabel("LLMs")
axes[0, 1].set_ylabel("Code lines")
axes[0, 1].tick_params(axis="x", rotation=0)
axes[0, 1].legend(title="Prompt")

sns.pointplot(
    data=df,
    x="LLM",
    y="Code_lines",
    hue="Mode",
    hue_order=MODE_ORDER,
    errorbar=("ci", 95),
    capsize=0.2,
    dodge=0.4,
    seed=RANDOM_SEED,
    ax=axes[1, 0],
)
axes[1, 0].set_title("C. Code lines: Mode × LLM")
axes[1, 0].set_xlabel("LLMs")
axes[1, 0].set_ylabel("Code lines")
axes[1, 0].tick_params(axis="x", rotation=0)
axes[1, 0].legend(title="Mode")

sns.pointplot(
    data=df,
    x="LLM",
    y="Comment_density",
    hue="Mode",
    hue_order=MODE_ORDER,
    errorbar=("ci", 95),
    capsize=0.2,
    dodge=0.4,
    seed=RANDOM_SEED,
    ax=axes[1, 1],
)
axes[1, 1].set_title("D. Comment density: Mode × LLM")
axes[1, 1].set_xlabel("LLMs")
axes[1, 1].set_ylabel("Comment density")
axes[1, 1].tick_params(axis="x", rotation=0)
axes[1, 1].legend(title="Mode")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "Supplementary_Figure1_EngineeringMetrics.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# 5. Non-parametric post-hoc analyses for significant continuous effects
# -----------------------------------------------------------------------------
def dunn_posthoc(groups: list[np.ndarray], group_names: list[str]) -> pd.DataFrame:
    """Dunn pairwise tests with tie correction and BH-FDR adjustment."""
    if len(groups) < 2:
        return pd.DataFrame()

    pairs = list(combinations(range(len(groups)), 2))
    all_data = np.concatenate(groups)
    ranks = rankdata(all_data)
    sizes = [len(group) for group in groups]
    n_total = sum(sizes)

    rank_sums: list[float] = []
    start = 0
    for size in sizes:
        rank_sums.append(float(ranks[start : start + size].sum()))
        start += size

    _, tie_counts = np.unique(all_data, return_counts=True)
    tie_term = (
        np.sum(tie_counts**3 - tie_counts) / (12 * (n_total - 1))
        if n_total > 1
        else 0.0
    )
    rank_variance = n_total * (n_total + 1) / 12 - tie_term

    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    z_values: list[float] = []

    for i, j in pairs:
        mean_i = rank_sums[i] / sizes[i]
        mean_j = rank_sums[j] / sizes[j]
        se = np.sqrt(rank_variance * (1 / sizes[i] + 1 / sizes[j]))
        z_value = 0.0 if se == 0 else (mean_i - mean_j) / se
        p_value = 2 * (1 - norm.cdf(abs(z_value)))
        z_values.append(z_value)
        p_values.append(p_value)

    adjusted = bh_fdr(np.asarray(p_values, dtype=float))
    for k, (i, j) in enumerate(pairs):
        rows.append(
            {
                "Group1": group_names[i],
                "Group2": group_names[j],
                "Z": z_values[k],
                "p_raw": p_values[k],
                "p_fdr": adjusted[k],
            }
        )

    return pd.DataFrame(rows)

def collect_groups(data: pd.DataFrame, outcome: str, factors: list[str]) -> tuple[list[np.ndarray], list[str]]:
    """Collect non-empty outcome arrays for combinations of factor levels."""
    groups: list[np.ndarray] = []
    labels: list[str] = []

    if len(factors) == 1:
        factor = factors[0]
        for level in data[factor].dropna().unique():
            values = data.loc[data[factor] == level, outcome].dropna().to_numpy()
            if len(values) > 0:
                groups.append(values)
                labels.append(f"{factor}={level}")
        return groups, labels

    factor1, factor2 = factors
    for level1 in data[factor1].dropna().unique():
        for level2 in data[factor2].dropna().unique():
            mask = (data[factor1] == level1) & (data[factor2] == level2)
            values = data.loc[mask, outcome].dropna().to_numpy()
            if len(values) > 0:
                groups.append(values)
                labels.append(f"{factor1}={level1}, {factor2}={level2}")
    return groups, labels

def run_kw_dunn(groups: list[np.ndarray], labels: list[str]) -> tuple[float, float, pd.DataFrame] | None:
    """Run Kruskal-Wallis, followed by Dunn only when the omnibus test is significant."""
    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        return None
    try:
        h_stat, p_kw = kruskal(*groups)
    except ValueError:
        return None
    if p_kw < 0.05:
        return float(h_stat), float(p_kw), dunn_posthoc(groups, labels)
    return float(h_stat), float(p_kw), pd.DataFrame()

FACTOR_TERM_MAP = {
    "C(Prompt, Sum)": "Prompt",
    "C(Mode, Sum)": "Mode",
    "C(LLM, Sum)": "LLM",
}

for outcome in CONTINUOUS_VARS:
    print(f"\n=== Post-hoc analysis: {outcome} ===")
    anova_table = anova_results[outcome]
    p_three = three_way_p_fdr.get(outcome, np.nan)

    if not np.isnan(p_three) and p_three < 0.05:
# strategy: stratify by Prompt and compare Mode × LLM cells using Kruskal-Wallis followed by Dunn tests.

        for prompt in df["Prompt"].dropna().unique():
            sub = df[df["Prompt"] == prompt]
            groups, labels = collect_groups(sub, outcome, ["Mode", "LLM"])
            result = run_kw_dunn(groups, labels)
            if result is None:
                continue
            h_stat, p_kw, dunn = result
            pd.DataFrame({"H": [h_stat], "P_value": [p_kw]}).to_csv(
                OUTPUT_DIR / f"KW_{outcome}_Prompt_{prompt}_Mode_x_LLM.csv", index=False
            )
            if not dunn.empty:
                dunn.to_csv(
                    OUTPUT_DIR / f"Dunn_{outcome}_Prompt_{prompt}_Mode_x_LLM.csv",
                    index=False,
                )
        continue

    significant_two_way = {
        term: p_value
        for term, p_value in interaction_p_fdr.get(outcome, {}).items()
        if not np.isnan(p_value) and p_value < 0.05
    }

    if significant_two_way:
        for term in significant_two_way:
            raw_parts = term.split(":")
            factor1 = FACTOR_TERM_MAP.get(raw_parts[0], raw_parts[0])
            factor2 = FACTOR_TERM_MAP.get(raw_parts[1], raw_parts[1])
            if factor1 not in df.columns or factor2 not in df.columns:
                continue

# strategy: stratify by the first factor and compare levels of the second factor.

            for level in df[factor1].dropna().unique():
                sub = df[df[factor1] == level]
                groups, labels = collect_groups(sub, outcome, [factor2])
                result = run_kw_dunn(groups, labels)
                if result is None:
                    continue
                h_stat, p_kw, dunn = result
                safe_term = term.replace(":", "_x_").replace("(", "").replace(")", "").replace(", ", "_")
                pd.DataFrame({"H": [h_stat], "P_value": [p_kw]}).to_csv(
                    OUTPUT_DIR / f"KW_{outcome}_{safe_term}_{factor1}_{level}.csv",
                    index=False,
                )
                if not dunn.empty:
                    dunn.to_csv(
                        OUTPUT_DIR / f"Dunn_{outcome}_{safe_term}_{factor1}_{level}.csv",
                        index=False,
                    )
        continue

# Main-effect post-hoc comparisons only when no higher-order interaction
# is significant after the prespecified within-outcome FDR correction.

    for term, factor in FACTOR_TERM_MAP.items():
        if term not in anova_table.index:
            continue
        p_main = float(anova_table.loc[term, "p_fdr"])
        if np.isnan(p_main) or p_main >= 0.05:
            continue

        levels = list(df[factor].dropna().unique())
        if len(levels) <= 2:
            medians = df.groupby(factor, observed=True)[outcome].median()
            medians.to_csv(OUTPUT_DIR / f"Median_{outcome}_{factor}.csv")
            continue

        groups, labels = collect_groups(df, outcome, [factor])
        result = run_kw_dunn(groups, labels)
        if result is None:
            continue
        h_stat, p_kw, dunn = result
        pd.DataFrame({"H": [h_stat], "P_value": [p_kw]}).to_csv(
            OUTPUT_DIR / f"KW_MainEffect_{outcome}_{factor}.csv", index=False
        )
        if not dunn.empty:
            dunn.to_csv(
                OUTPUT_DIR / f"Dunn_MainEffect_{outcome}_{factor}.csv", index=False
            )


print(f"\nAnalysis complete. Outputs saved to: {OUTPUT_DIR.resolve()}")