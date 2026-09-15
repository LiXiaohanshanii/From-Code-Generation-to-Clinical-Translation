from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. 基本参数
# ============================================================

DATA_PATH = Path("train_data.csv")
TARGET_COLUMN = "TRUST"

TEST_SIZE = 0.20
RANDOM_STATE = 42

ONE_HOT_COLUMNS = [
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]

ORDINAL_COLUMNS = [
    "TPPA"
]

NUMERIC_COLUMNS = [
    "AGE",
    "TP",
    "HIV",
    "WBC",
    "RBC",
    "PLT",
    "NC",
    "LY",
    "NLR"
]

FEATURE_COLUMNS = ONE_HOT_COLUMNS + ORDINAL_COLUMNS + NUMERIC_COLUMNS


# ============================================================
# 2. 工具函数
# ============================================================

def parse_titer(value):
    """
    将 TRUST 滴度转换为数值。

    支持以下常见形式：
    16
    "16"
    "1:16"
    "1：16"
    ">=16"
    "≥16"

    无法解析时返回 NaN。
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace("：", ":")

    # 对于 1:16 形式，取冒号后面的数字
    if ":" in text:
        text = text.split(":")[-1].strip()

    match = re.search(r"\d+(?:\.\d+)?", text)

    if match is None:
        return np.nan

    return float(match.group())


def load_and_validate_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取数据、检查字段、处理目标变量并返回 X 和 y。
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}\n"
            "请确认 train_data.csv 与当前 Python 脚本位于同一目录。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    # 清理字段名前后的空格
    data.columns = data.columns.astype(str).str.strip()

    # 兼容需求描述中可能出现的 DIAGONSIS 拼写
    if "DIAGNOSIS" not in data.columns and "DIAGONSIS" in data.columns:
        warnings.warn(
            "检测到字段 DIAGONSIS，已自动更名为 DIAGNOSIS。",
            UserWarning
        )
        data = data.rename(columns={"DIAGONSIS": "DIAGNOSIS"})

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要字段：\n"
            f"{missing_columns}\n\n"
            f"当前数据字段为：\n{data.columns.tolist()}"
        )

    # 检查 TRUST 是否为最后一列
    if data.columns[-1] != TARGET_COLUMN:
        warnings.warn(
            f"目标字段 {TARGET_COLUMN} 不是数据集最后一列。"
            "程序仍将按字段名进行建模。",
            UserWarning
        )

    # 分类变量按要求应无缺失值
    categorical_columns = ONE_HOT_COLUMNS + ORDINAL_COLUMNS
    categorical_missing = data[categorical_columns].isna().sum()
    categorical_missing = categorical_missing[
        categorical_missing > 0
    ]

    if not categorical_missing.empty:
        raise ValueError(
            "以下分类变量存在缺失值，但任务要求分类变量无缺失值：\n"
            f"{categorical_missing.to_string()}"
        )

    # 分类字段统一转换为字符串，避免数字和字符串混合导致编码错误
    for column in categorical_columns:
        data[column] = data[column].astype(str).str.strip()

    # 连续变量转换为数值
    # 非法字符将转换为 NaN，之后由中位数填充
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # TRUST 滴度转换为数值
    trust_numeric = data[TARGET_COLUMN].apply(parse_titer)

    invalid_target_count = int(trust_numeric.isna().sum())

    if invalid_target_count > 0:
        warnings.warn(
            f"目标字段 {TARGET_COLUMN} 中有 "
            f"{invalid_target_count} 条记录无法转换为数值，"
            "这些记录将被删除。",
            UserWarning
        )

        valid_mask = trust_numeric.notna()
        data = data.loc[valid_mask].copy()
        trust_numeric = trust_numeric.loc[valid_mask]

    if data.empty:
        raise ValueError("删除无效目标值后，数据集为空。")

    # 二分类目标：TRUST >= 16 为 1，否则为 0
    y = (trust_numeric >= 16).astype(int)

    X = data[FEATURE_COLUMNS].copy()

    class_counts = y.value_counts().sort_index()

    if y.nunique() != 2:
        raise ValueError(
            "目标变量转换后没有同时包含两个类别。\n"
            f"当前类别分布：\n{class_counts.to_string()}"
        )

    if class_counts.min() < 2:
        raise ValueError(
            "少数类别样本数少于 2，无法进行分层训练集/测试集划分。\n"
            f"当前类别分布：\n{class_counts.to_string()}"
        )

    return X, y


def create_preprocessor() -> ColumnTransformer:
    """
    创建数据预处理器。

    SEX、DEPT、DIAGNOSIS：
        One-HotEncoder

    TPPA：
        OrdinalEncoder

    连续变量：
        中位数填充
    """
    one_hot_pipeline = SklearnPipeline(
        steps=[
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64
                )
            )
        ]
    )

    ordinal_pipeline = SklearnPipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64
                )
            )
        ]
    )

    numeric_pipeline = SklearnPipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot_categorical",
                one_hot_pipeline,
                ONE_HOT_COLUMNS
            ),
            (
                "ordinal_categorical",
                ordinal_pipeline,
                ORDINAL_COLUMNS
            ),
            (
                "continuous",
                numeric_pipeline,
                NUMERIC_COLUMNS
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


def create_cv_and_smote_neighbors(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """
    根据训练集少数类别数量自动确定交叉验证折数和 SMOTE 的
    k_neighbors，降低少数类别样本较少时发生报错的风险。
    """
    minority_count = int(y_train.value_counts().min())

    # 最多使用 5 折；少数类样本较少时自动降低折数
    n_splits = min(5, minority_count)

    if n_splits < 2:
        raise ValueError(
            "训练集中少数类别样本不足，无法执行交叉验证。"
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_splits = list(cv.split(X_train, y_train))

    # 计算所有交叉验证训练折中最小的少数类样本数
    minimum_fold_minority_count = min(
        int(y_train.iloc[train_indices].value_counts().min())
        for train_indices, _ in cv_splits
    )

    # SMOTE 要求 k_neighbors 小于少数类样本数
    smote_k_neighbors = min(
        5,
        minimum_fold_minority_count - 1
    )

    if smote_k_neighbors < 1:
        raise ValueError(
            "交叉验证训练折中的少数类别样本不足，"
            "无法应用 SMOTE。"
        )

    return cv_splits, smote_k_neighbors


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict[str, float]:
    """
    在独立测试集上计算模型评估指标。
    """
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1-score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "AUC": roc_auc_score(
            y_test,
            y_probability
        )
    }

    return metrics


# ============================================================
# 3. 主程序
# ============================================================

def main():
    # --------------------------------------------------------
    # 3.1 读取数据
    # --------------------------------------------------------
    X, y = load_and_validate_data(DATA_PATH)

    print("=" * 60)
    print("数据读取完成")
    print("=" * 60)
    print(f"总样本数：{len(X)}")
    print(f"特征数量：{X.shape[1]}")

    print("\n二分类标签定义：")
    print("0：TRUST < 16")
    print("1：TRUST >= 16")

    print("\n全部数据类别分布：")
    print(y.value_counts().sort_index().rename(
        index={
            0: "TRUST < 16",
            1: "TRUST >= 16"
        }
    ))

    # --------------------------------------------------------
    # 3.2 分层划分训练集和测试集
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n训练集样本数：", len(X_train))
    print("测试集样本数：", len(X_test))

    print("\n训练集类别分布：")
    print(y_train.value_counts().sort_index().rename(
        index={
            0: "TRUST < 16",
            1: "TRUST >= 16"
        }
    ))

    print("\n测试集类别分布：")
    print(y_test.value_counts().sort_index().rename(
        index={
            0: "TRUST < 16",
            1: "TRUST >= 16"
        }
    ))

    # 测试集中必须同时存在两个类别，才能计算 AUC
    if y_test.nunique() != 2:
        raise ValueError(
            "测试集中未同时包含两个类别，无法计算 AUC。"
            "请增加样本量或调整 TEST_SIZE。"
        )

    # --------------------------------------------------------
    # 3.3 创建交叉验证划分并确定 SMOTE 参数
    # --------------------------------------------------------
    cv_splits, smote_k_neighbors = (
        create_cv_and_smote_neighbors(
            X_train,
            y_train
        )
    )

    print(f"\n交叉验证折数：{len(cv_splits)}")
    print(f"SMOTE k_neighbors：{smote_k_neighbors}")

    # --------------------------------------------------------
    # 3.4 创建预处理器
    # --------------------------------------------------------
    preprocessor = create_preprocessor()

    # --------------------------------------------------------
    # 3.5 创建随机森林分类器
    # --------------------------------------------------------
    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,

        # 明确不使用多进程
        n_jobs=1
    )

    # --------------------------------------------------------
    # 3.6 构建包含 SMOTE 的完整建模管道
    #
    # 执行顺序：
    # 原始数据
    # -> 中位数填充及分类编码
    # -> SMOTE
    # -> 随机森林
    #
    # SMOTE 只会作用于训练集或交叉验证训练折。
    # --------------------------------------------------------
    model_pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=smote_k_neighbors
                )
            ),
            (
                "classifier",
                random_forest
            )
        ]
    )

    # --------------------------------------------------------
    # 3.7 设置超参数搜索范围
    # --------------------------------------------------------
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"]
    }

    # --------------------------------------------------------
    # 3.8 网格搜索
    #
    # refit="roc_auc"：
    # 以交叉验证 AUC 最高的模型作为最终最佳模型。
    #
    # n_jobs=1：
    # 不使用多进程。
    # --------------------------------------------------------
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring={
            "accuracy": "accuracy",
            "recall": "recall",
            "precision": "precision",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        refit="roc_auc",
        cv=cv_splits,
        n_jobs=1,
        verbose=1,
        return_train_score=False,
        error_score="raise"
    )

    print("\n" + "=" * 60)
    print("开始进行随机森林超参数调优")
    print("=" * 60)

    grid_search.fit(X_train, y_train)

    # 最佳模型已经使用整个训练集重新拟合
    best_model = grid_search.best_estimator_

    print("\n" + "=" * 60)
    print("超参数调优完成")
    print("=" * 60)

    print("\n最佳参数：")
    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        clean_name = parameter_name.replace(
            "classifier__",
            ""
        )
        print(f"{clean_name}: {parameter_value}")

    print(
        "\n最佳交叉验证 AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # --------------------------------------------------------
    # 3.9 在独立测试集上评估
    # --------------------------------------------------------
    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    print("\n" + "=" * 60)
    print("独立测试集评估结果")
    print("=" * 60)

    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name:<12}: {metric_value:.4f}")

    # 可在后续预测代码中直接使用 best_model
    #
    # 示例：
    # new_predictions = best_model.predict(new_data)
    # new_probabilities = best_model.predict_proba(new_data)[:, 1]


if __name__ == "__main__":
    main()