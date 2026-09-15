from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ============================================================
# 1. 基本配置
# ============================================================

DATA_FILE = Path("train_data.csv")
TARGET_COLUMN = "TRUST"
TARGET_THRESHOLD = 16

TEST_SIZE = 0.20
RANDOM_STATE = 42

CATEGORICAL_ONEHOT_COLUMNS = [
    "SEX",
    "DEPT",
    "DIAGNOSIS",
]

CATEGORICAL_ORDINAL_COLUMNS = [
    "TPPA",
]

CONTINUOUS_COLUMNS = [
    "AGE",
    "TP",
    "HIV",
    "WBC",
    "RBC",
    "PLT",
    "NC",
    "LY",
    "NLR",
]

FEATURE_COLUMNS = (
    CATEGORICAL_ONEHOT_COLUMNS
    + CATEGORICAL_ORDINAL_COLUMNS
    + CONTINUOUS_COLUMNS
)

# TPPA 的实际类别顺序未在题目中给出。
# 保持为 None 时，OrdinalEncoder 会根据训练数据自动确定类别顺序。
#
# 若已知 TPPA 的明确医学顺序，可修改为类似：
# TPPA_ORDER = ["阴性", "弱阳性", "阳性"]
TPPA_ORDER: list | None = None


# ============================================================
# 2. 数据读取及检查
# ============================================================

def load_and_prepare_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """读取数据，检查字段，并生成二分类目标变量。"""

    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}\n"
            "请将 train_data.csv 放在当前 PyCharm 项目的运行目录中，"
            "或修改 DATA_FILE 为正确路径。"
        )

    df = pd.read_csv(file_path, encoding="utf-8")

    # 清除列名前后的空格
    df.columns = df.columns.astype(str).str.strip()

    # 兼容题目中可能出现的 DIAGONSIS 拼写
    if "DIAGNOSIS" not in df.columns and "DIAGONSIS" in df.columns:
        df = df.rename(columns={"DIAGONSIS": "DIAGNOSIS"})
        print("提示：已将列名 DIAGONSIS 自动更正为 DIAGNOSIS。")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要列："
            f"{missing_columns}\n"
            f"当前数据列为：{df.columns.tolist()}"
        )

    if df.columns[-1] != TARGET_COLUMN:
        print(
            f"警告：数据集最后一列是 {df.columns[-1]!r}，"
            f"而不是目标列 {TARGET_COLUMN!r}。"
            "程序仍将按列名 TRUST 作为目标列。"
        )

    # TRUST 转换为数值。
    # 无法转换的值将变为 NaN，并删除对应样本。
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    invalid_target_count = int(df[TARGET_COLUMN].isna().sum())

    if invalid_target_count > 0:
        print(
            f"警告：TRUST 中有 {invalid_target_count} 个缺失或非数值记录，"
            "这些记录将被删除。"
        )
        df = df.dropna(subset=[TARGET_COLUMN]).copy()

    if df.empty:
        raise ValueError("删除无效目标值后，数据集为空。")

    # 连续变量强制转换为数值。
    # 非法字符串将转为 NaN，随后在 Pipeline 中使用中位数填补。
    for column in CONTINUOUS_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 检查分类变量缺失值
    categorical_columns = (
        CATEGORICAL_ONEHOT_COLUMNS
        + CATEGORICAL_ORDINAL_COLUMNS
    )

    categorical_missing = df[categorical_columns].isna().sum()
    categorical_missing = categorical_missing[
        categorical_missing > 0
    ]

    if not categorical_missing.empty:
        raise ValueError(
            "题目设定分类变量不存在缺失值，但检测到以下缺失：\n"
            f"{categorical_missing.to_string()}"
        )

    X = df[FEATURE_COLUMNS].copy()

    # TRUST >= 16 为正类1，否则为负类0
    y = (df[TARGET_COLUMN] >= TARGET_THRESHOLD).astype(int)

    class_counts = y.value_counts().sort_index()

    if y.nunique() != 2:
        raise ValueError(
            "转换后的目标变量不是二分类数据。\n"
            f"类别计数为：{class_counts.to_dict()}\n"
            "请检查 TRUST 列是否同时包含小于16和大于等于16的记录。"
        )

    if class_counts.min() < 2:
        raise ValueError(
            "少数类别样本数不足2个，无法进行分层训练集/测试集划分。"
        )

    return X, y


# ============================================================
# 3. 构建预处理器
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """构建分类变量编码和连续变量缺失值处理流程。"""

    onehot_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )

    if TPPA_ORDER is None:
        ordinal_transformer = OrdinalEncoder(
            categories="auto",
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )
    else:
        ordinal_transformer = OrdinalEncoder(
            categories=[TPPA_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )

    continuous_transformer = Pipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                onehot_transformer,
                CATEGORICAL_ONEHOT_COLUMNS,
            ),
            (
                "ordinal",
                ordinal_transformer,
                CATEGORICAL_ORDINAL_COLUMNS,
            ),
            (
                "continuous",
                continuous_transformer,
                CONTINUOUS_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# 4. 根据少数类样本数设置交叉验证和 SMOTE
# ============================================================

def determine_cv_and_smote(
    y_train: pd.Series,
) -> tuple[StratifiedKFold, int]:
    """
    根据训练集少数类数量，动态设置交叉验证折数和 SMOTE 的
    k_neighbors，防止小样本数据因近邻数量不足而报错。
    """

    minority_count = int(y_train.value_counts().min())

    # 最多使用5折；每一折至少应尽可能包含少数类样本
    cv_splits = min(5, minority_count)

    if cv_splits < 2:
        raise ValueError(
            "训练集中少数类别样本不足，无法执行交叉验证。"
        )

    # 在某个交叉验证训练折中可能出现的最小少数类样本数
    minimum_fold_minority = (
        minority_count
        - math.ceil(minority_count / cv_splits)
    )

    if minimum_fold_minority < 2:
        raise ValueError(
            "训练集中的少数类别样本过少，无法在交叉验证内部安全执行 SMOTE。"
            f"当前训练集少数类样本数：{minority_count}。"
        )

    # SMOTE 要求 k_neighbors 小于当前少数类样本数
    smote_k_neighbors = min(5, minimum_fold_minority - 1)

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return cv, smote_k_neighbors


# ============================================================
# 5. 模型评估
# ============================================================

def evaluate_model(
    model: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """计算测试集上的二分类评估指标。"""

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "AUC": roc_auc_score(y_test, y_probability),
    }

    result = pd.DataFrame(
        {
            "评价指标": list(metrics.keys()),
            "测试集结果": list(metrics.values()),
        }
    )

    return result


# ============================================================
# 6. 主程序
# ============================================================

def main() -> None:
    X, y = load_and_prepare_data(DATA_FILE)

    print("=" * 60)
    print("数据集信息")
    print("=" * 60)
    print(f"样本总数：{len(X)}")
    print(f"特征数量：{X.shape[1]}")
    print("\n二分类目标分布：")
    print(f"0类（TRUST < {TARGET_THRESHOLD}）：{int((y == 0).sum())}")
    print(f"1类（TRUST >= {TARGET_THRESHOLD}）：{int((y == 1).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n训练集样本数：", len(X_train))
    print("测试集样本数：", len(X_test))

    cv, smote_k_neighbors = determine_cv_and_smote(y_train)

    print(f"交叉验证折数：{cv.n_splits}")
    print(f"SMOTE k_neighbors：{smote_k_neighbors}")

    preprocessor = build_preprocessor()

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,  # 明确禁止随机森林使用多进程
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=smote_k_neighbors,
                ),
            ),
            ("random_forest", random_forest),
        ]
    )

    # Pipeline 内模型参数必须使用“步骤名__参数名”的格式
    parameter_grid = {
        "random_forest__n_estimators": [100, 200],
        "random_forest__max_depth": [10],
        "random_forest__min_samples_split": [2],
        "random_forest__min_samples_leaf": [1],
        "random_forest__class_weight": ["balanced"],
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=1,  # 明确禁止网格搜索使用多进程
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    print("\n" + "=" * 60)
    print("开始进行随机森林超参数搜索")
    print("=" * 60)

    grid_search.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("超参数搜索结果")
    print("=" * 60)
    print("最佳参数：")

    for parameter_name, parameter_value in grid_search.best_params_.items():
        clean_name = parameter_name.replace("random_forest__", "")
        print(f"  {clean_name}: {parameter_value}")

    print(
        f"\n最佳交叉验证 AUC："
        f"{grid_search.best_score_:.4f}"
    )

    evaluation_result = evaluate_model(
        grid_search,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("独立测试集评估结果")
    print("=" * 60)

    print(
        evaluation_result.to_string(
            index=False,
            formatters={
                "测试集结果": lambda value: f"{value:.4f}"
            },
        )
    )


if __name__ == "__main__":
    main()