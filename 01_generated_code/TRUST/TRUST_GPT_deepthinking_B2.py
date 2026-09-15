# -*- coding: utf-8 -*-
"""
使用随机森林预测 TRUST 是否 >= 16

运行环境：
    Python 3.10+
    PyCharm 2025.2.3

依赖安装：
    pip install pandas numpy scikit-learn imbalanced-learn
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline


# =========================
# 1. 基本参数
# =========================

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_FILE = Path(__file__).resolve().parent / "train_data.csv"

TARGET_COLUMN = "TRUST"

ONEHOT_COLUMNS = [
    "SEX",
    "DEPT",
    "DIAGNOSIS",
]

ORDINAL_COLUMNS = [
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
    ONEHOT_COLUMNS
    + ORDINAL_COLUMNS
    + CONTINUOUS_COLUMNS
)


def create_onehot_encoder() -> OneHotEncoder:
    """
    创建独热编码器。

    sparse_output=False 适用于较新的 scikit-learn；
    sparse=False 用于兼容旧版本。
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def load_and_validate_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取数据、检查字段并生成二分类目标变量。

    二分类标签：
        TRUST >= 16：1
        TRUST < 16：0
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"没有找到数据文件：{file_path}\n"
            "请将 train_data.csv 放在本脚本所在目录。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    if data.empty:
        raise ValueError("train_data.csv 中没有数据。")

    # 删除字段名前后的空格，避免因隐藏空格导致字段识别失败
    data.columns = data.columns.str.strip()

    # 兼容任务描述中 DIAGONSIS 的常见拼写错误
    if "DIAGNOSIS" not in data.columns and "DIAGONSIS" in data.columns:
        warnings.warn(
            "检测到字段 DIAGONSIS，已自动重命名为 DIAGNOSIS。",
            stacklevel=2,
        )
        data = data.rename(columns={"DIAGONSIS": "DIAGNOSIS"})

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要字段："
            + ", ".join(missing_columns)
        )

    if data.columns[-1] != TARGET_COLUMN:
        warnings.warn(
            f"数据集最后一列为 {data.columns[-1]!r}，而不是 TRUST。"
            "程序仍将使用名为 TRUST 的字段作为目标列。",
            stacklevel=2,
        )

    # 只保留建模所需字段
    data = data[required_columns].copy()

    # 将连续变量转换为数值。
    # 非法内容会转换为 NaN，之后由中位数填充。
    for column in CONTINUOUS_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # 如果 TPPA 全部可以转换成数值，则转换成数值。
    # 这样 OrdinalEncoder 会按照数值大小建立顺序，而不是字符串顺序。
    tppa_numeric = pd.to_numeric(
        data["TPPA"],
        errors="coerce",
    )
    if tppa_numeric.notna().all():
        data["TPPA"] = tppa_numeric

    # TRUST 必须能够转换为数值
    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    invalid_target_count = int(trust_numeric.isna().sum())
    if invalid_target_count > 0:
        raise ValueError(
            f"目标列 TRUST 中有 {invalid_target_count} 个值无法转换为数值。"
            "请检查 TRUST 是否均为 1、2、4、8、16 等数值。"
        )

    X = data[FEATURE_COLUMNS].copy()
    y = (trust_numeric >= 16).astype(np.int8)

    class_counts = y.value_counts().sort_index()

    if y.nunique() != 2:
        raise ValueError(
            "转换后的目标变量不是二分类数据。"
            "请确认 TRUST 中同时存在小于16和大于等于16的样本。"
        )

    print("原始数据概况")
    print("-" * 50)
    print(f"样本数量：{len(data)}")
    print(f"特征数量：{len(FEATURE_COLUMNS)}")
    print(f"TRUST < 16 的样本数：{class_counts.get(0, 0)}")
    print(f"TRUST >= 16 的样本数：{class_counts.get(1, 0)}")
    print()

    return X, y


def determine_smote_and_cv_parameters(
    y_train: pd.Series,
) -> tuple[int, int]:
    """
    根据训练集中少数类样本数确定交叉验证折数和
    SMOTE 的 k_neighbors，降低小样本情况下的报错风险。
    """
    minority_count = int(y_train.value_counts().min())

    if minority_count < 3:
        raise ValueError(
            "训练集中的少数类样本少于3个，无法稳定执行"
            "分层交叉验证和SMOTE。请增加少数类样本。"
        )

    # 最多使用5折，但每折中必须包含少数类样本
    cv_splits = min(5, minority_count)

    # 估计每个交叉验证训练折中最少的少数类样本数
    maximum_validation_minority = int(
        np.ceil(minority_count / cv_splits)
    )
    minimum_training_minority = (
        minority_count - maximum_validation_minority
    )

    if minimum_training_minority < 2:
        raise ValueError(
            "交叉验证训练折中的少数类样本过少，无法应用SMOTE。"
        )

    smote_k_neighbors = min(
        5,
        minimum_training_minority - 1,
    )

    return cv_splits, smote_k_neighbors


def build_model_pipeline(
    smote_k_neighbors: int,
) -> ImbalancedPipeline:
    """
    构建预处理、SMOTE和随机森林组成的完整流水线。
    """
    continuous_pipeline = SklearnPipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    onehot_pipeline = SklearnPipeline(
        steps=[
            (
                "onehot_encoder",
                create_onehot_encoder(),
            ),
        ]
    )

    ordinal_pipeline = SklearnPipeline(
        steps=[
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                continuous_pipeline,
                CONTINUOUS_COLUMNS,
            ),
            (
                "onehot",
                onehot_pipeline,
                ONEHOT_COLUMNS,
            ),
            (
                "ordinal",
                ordinal_pipeline,
                ORDINAL_COLUMNS,
            ),
        ],
        remainder="drop",
        # 强制输出稠密矩阵，便于SMOTE处理
        sparse_threshold=0.0,
    )

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    model_pipeline = ImbalancedPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=smote_k_neighbors,
                ),
            ),
            ("model", random_forest),
        ]
    )

    return model_pipeline


def evaluate_model(
    fitted_model: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    在独立测试集上计算模型评价指标。
    """
    y_pred = fitted_model.predict(X_test)
    y_probability = fitted_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "AUC": roc_auc_score(
            y_test,
            y_probability,
        ),
    }

    return metrics


def main() -> None:
    # =========================
    # 2. 读取并处理目标变量
    # =========================
    X, y = load_and_validate_data(DATA_FILE)

    # =========================
    # 3. 划分训练集和测试集
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("数据集划分")
    print("-" * 50)
    print(f"训练集样本数：{len(X_train)}")
    print(f"测试集样本数：{len(X_test)}")
    print(
        "训练集类别分布：",
        y_train.value_counts().sort_index().to_dict(),
    )
    print(
        "测试集类别分布：",
        y_test.value_counts().sort_index().to_dict(),
    )
    print()

    if y_test.nunique() != 2:
        raise ValueError(
            "测试集中未同时包含两个类别，无法计算AUC。"
            "请增加样本量或调整测试集比例。"
        )

    cv_splits, smote_k_neighbors = (
        determine_smote_and_cv_parameters(y_train)
    )

    print("训练参数")
    print("-" * 50)
    print(f"分层交叉验证折数：{cv_splits}")
    print(f"SMOTE k_neighbors：{smote_k_neighbors}")
    print()

    # =========================
    # 4. 构建流水线
    # =========================
    model_pipeline = build_model_pipeline(
        smote_k_neighbors=smote_k_neighbors,
    )

    # 参数名称前需要加流水线步骤名称 model__
    parameter_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10],
        "model__min_samples_split": [2],
        "model__min_samples_leaf": [1],
        "model__class_weight": ["balanced"],
    }

    cross_validation = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 使用AUC选择最佳参数，同时记录其他指标
    scoring = {
        "accuracy": "accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring=scoring,
        refit="roc_auc",
        cv=cross_validation,
        n_jobs=1,  # 不使用多进程
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    # =========================
    # 5. 模型训练和参数调优
    # =========================
    print("开始模型训练与超参数搜索……")
    grid_search.fit(X_train, y_train)

    print()
    print("超参数搜索结果")
    print("-" * 50)

    readable_best_params = {
        key.replace("model__", ""): value
        for key, value in grid_search.best_params_.items()
    }

    print(f"最佳参数：{readable_best_params}")
    print(
        "最佳交叉验证平均AUC："
        f"{grid_search.best_score_:.4f}"
    )

    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_

    print(
        "最佳模型交叉验证平均Accuracy："
        f"{cv_results['mean_test_accuracy'][best_index]:.4f}"
    )
    print(
        "最佳模型交叉验证平均Recall："
        f"{cv_results['mean_test_recall'][best_index]:.4f}"
    )
    print(
        "最佳模型交叉验证平均Precision："
        f"{cv_results['mean_test_precision'][best_index]:.4f}"
    )
    print(
        "最佳模型交叉验证平均F1-score："
        f"{cv_results['mean_test_f1'][best_index]:.4f}"
    )
    print()

    # =========================
    # 6. 独立测试集评估
    # =========================
    test_metrics = evaluate_model(
        fitted_model=grid_search,
        X_test=X_test,
        y_test=y_test,
    )

    print("独立测试集评估结果")
    print("-" * 50)

    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name:<10}: {metric_value:.4f}")


if __name__ == "__main__":
    main()