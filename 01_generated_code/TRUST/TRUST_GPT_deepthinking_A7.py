from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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


# =========================
# 全局参数
# =========================

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "SEX",
    "AGE",
    "DEPT",
    "DIAGNOSIS",
    "TPPA",
    "TP",
    "HIV",
    "WBC",
    "RBC",
    "PLT",
    "NC",
    "LY",
    "NLR",
]

TARGET_COLUMN = "TRUST"

# 独热编码变量
ONE_HOT_COLUMNS = [
    "SEX",
    "DEPT",
    "DIAGNOSIS",
]

# 序数编码变量
ORDINAL_COLUMNS = [
    "TPPA",
]

# 连续变量
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


# 如果明确知道 TPPA 各类别的医学顺序，可以在这里填写。
# 例如：
# TPPA_ORDER = ["阴性", "弱阳性", "阳性", "强阳性"]
#
# 设置为 None 时，OrdinalEncoder 会根据训练数据中的取值自动确定顺序。
TPPA_ORDER: list | None = None


def load_and_prepare_data(
    csv_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取并检查数据，将 TRUST 转换为二分类目标变量。

    TRUST >= 16：正类，编码为 1
    TRUST < 16：负类，编码为 0
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{csv_path}\n"
            "请确认 train_data.csv 与当前 Python 脚本位于同一目录。"
        )

    data = pd.read_csv(
        csv_path,
        encoding="utf-8",
    )

    # 清除列名两侧可能存在的空格
    data.columns = data.columns.astype(str).str.strip()

    # 兼容需求描述中可能出现的 DIAGONSIS 拼写
    if (
        "DIAGNOSIS" not in data.columns
        and "DIAGONSIS" in data.columns
    ):
        data = data.rename(
            columns={"DIAGONSIS": "DIAGNOSIS"}
        )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV 文件缺少以下必要列：{missing_columns}"
        )

    # 只保留建模所需字段
    data = data[required_columns].copy()

    # 将 TRUST 转换为数值
    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    # 删除目标列为空或无法转换为数值的记录
    invalid_target_count = int(
        trust_numeric.isna().sum()
    )

    if invalid_target_count > 0:
        print(
            f"提示：删除 {invalid_target_count} 行 "
            "TRUST 为空或无法转换为数值的记录。"
        )

        valid_mask = trust_numeric.notna()

        data = data.loc[valid_mask].copy()
        trust_numeric = trust_numeric.loc[valid_mask]

    if data.empty:
        raise ValueError(
            "清理 TRUST 后没有可用于建模的数据。"
        )

    X = data[FEATURE_COLUMNS].copy()

    # TRUST >= 16 为正类
    y = (trust_numeric >= 16).astype(int)
    y.name = "TRUST_GE_16"

    # 将连续变量强制转换为数值
    # 无法转换的内容会变成 NaN，随后使用中位数填补
    for column in CONTINUOUS_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # 按需求，分类变量不应包含缺失值
    categorical_columns = (
        ONE_HOT_COLUMNS + ORDINAL_COLUMNS
    )

    categorical_missing = (
        X[categorical_columns]
        .isna()
        .sum()
    )

    categorical_missing = categorical_missing[
        categorical_missing > 0
    ]

    if not categorical_missing.empty:
        raise ValueError(
            "检测到分类变量缺失值，但需求指定分类变量无缺失值："
            f"{categorical_missing.to_dict()}"
        )

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    创建数据预处理器。

    SEX、DEPT、DIAGNOSIS：独热编码
    TPPA：序数编码
    连续变量：中位数填补
    """

    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    ordinal_parameters = {
        "handle_unknown": "use_encoded_value",
        "unknown_value": -1,
        "dtype": np.float64,
    }

    if TPPA_ORDER is not None:
        ordinal_parameters["categories"] = [
            TPPA_ORDER
        ]

    ordinal_encoder = OrdinalEncoder(
        **ordinal_parameters
    )

    continuous_imputer = SimpleImputer(
        strategy="median"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot",
                one_hot_encoder,
                ONE_HOT_COLUMNS,
            ),
            (
                "ordinal",
                ordinal_encoder,
                ORDINAL_COLUMNS,
            ),
            (
                "continuous",
                continuous_imputer,
                CONTINUOUS_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def choose_cv_and_smote_k(
    y_train: pd.Series,
) -> tuple[int, int]:
    """
    根据训练集中少数类样本数量，选择交叉验证折数和
    SMOTE 的 k_neighbors。

    这样可以降低小样本数据在交叉验证过程中因少数类
    样本不足而报错的风险。
    """

    class_counts = y_train.value_counts()

    if len(class_counts) != 2:
        raise ValueError(
            "训练集必须同时包含 0 和 1 两个类别。"
            f"当前类别分布：{class_counts.to_dict()}"
        )

    minority_count = int(class_counts.min())

    if minority_count < 3:
        raise ValueError(
            "训练集中的少数类样本少于 3 个，"
            "无法在交叉验证内部可靠执行 SMOTE。"
            "请增加少数类样本或调整测试集比例。"
        )

    # 默认最多使用 5 折
    cv_splits = min(5, minority_count)

    # 估算某个交叉验证训练折中少数类样本数的保守下界
    minimum_minority_in_cv_train = (
        minority_count
        - math.ceil(minority_count / cv_splits)
    )

    if minimum_minority_in_cv_train < 2:
        raise ValueError(
            "交叉验证训练折中的少数类样本不足，"
            "SMOTE 无法执行。"
        )

    smote_k_neighbors = min(
        5,
        minimum_minority_in_cv_train - 1,
    )

    return cv_splits, smote_k_neighbors


def main() -> None:
    """
    主程序：
    1. 读取数据
    2. 划分训练集和测试集
    3. 创建预处理、SMOTE 和随机森林管道
    4. 执行网格搜索
    5. 在独立测试集上评估
    """

    # train_data.csv 应与本脚本位于同一目录
    csv_path = (
        Path(__file__).resolve().parent
        / "train_data.csv"
    )

    X, y = load_and_prepare_data(csv_path)

    print("完整数据的目标类别分布：")

    class_distribution = (
        y.value_counts()
        .sort_index()
        .rename(
            index={
                0: "TRUST < 16",
                1: "TRUST >= 16",
            }
        )
    )

    print(class_distribution)

    if y.nunique() != 2:
        raise ValueError(
            "TRUST 转换后未形成两个类别。"
            f"当前类别：{sorted(y.unique().tolist())}"
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "每个类别至少需要 2 个样本，"
            "才能进行分层训练集和测试集划分。"
        )

    # 分层划分，保持训练集与测试集中的类别比例
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    cv_splits, smote_k_neighbors = (
        choose_cv_and_smote_k(y_train)
    )

    print("\n训练集类别分布：")
    print(
        y_train.value_counts()
        .sort_index()
        .rename(
            index={
                0: "TRUST < 16",
                1: "TRUST >= 16",
            }
        )
    )

    print("\n测试集类别分布：")
    print(
        y_test.value_counts()
        .sort_index()
        .rename(
            index={
                0: "TRUST < 16",
                1: "TRUST >= 16",
            }
        )
    )

    # 使用 imbalanced-learn 的 Pipeline。
    # SMOTE 只在每个交叉验证训练折内执行，避免数据泄漏。
    pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=smote_k_neighbors,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 指定的随机森林超参数
    param_grid = {
        "classifier__n_estimators": [
            100,
            200,
        ],
        "classifier__max_depth": [
            10,
        ],
        "classifier__min_samples_split": [
            2,
        ],
        "classifier__min_samples_leaf": [
            1,
        ],
        "classifier__class_weight": [
            "balanced",
        ],
    }

    stratified_cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 以 ROC AUC 作为超参数选择指标
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=stratified_cv,
        n_jobs=1,  # 不使用多进程
        refit=True,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    print("\n开始进行网格搜索和交叉验证……")

    grid_search.fit(
        X_train,
        y_train,
    )

    best_model = grid_search.best_estimator_

    # 预测类别
    y_pred = best_model.predict(X_test)

    # 预测正类概率，用于计算 ROC AUC
    classifier = best_model.named_steps[
        "classifier"
    ]

    positive_class_position = np.where(
        classifier.classes_ == 1
    )[0]

    if len(positive_class_position) == 0:
        raise ValueError(
            "模型中未找到正类标签 1，无法计算 ROC AUC。"
        )

    positive_class_index = int(
        positive_class_position[0]
    )

    y_probability = best_model.predict_proba(
        X_test
    )[:, positive_class_index]

    if y_test.nunique() < 2:
        raise ValueError(
            "测试集只包含一个类别，无法计算 ROC AUC。"
            "请适当调整 TEST_SIZE。"
        )

    # 计算模型评估指标
    evaluation_metrics = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
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
        "ROC AUC": roc_auc_score(
            y_test,
            y_probability,
        ),
    }

    print("\n最佳超参数：")

    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        displayed_name = parameter_name.replace(
            "classifier__",
            "",
        )

        print(
            f"  {displayed_name}: "
            f"{parameter_value}"
        )

    print(
        "\n最佳交叉验证 ROC AUC："
        f"{grid_search.best_score_:.4f}"
    )

    print(
        f"交叉验证折数：{cv_splits}"
    )

    print(
        "SMOTE k_neighbors："
        f"{smote_k_neighbors}"
    )

    metrics_table = pd.DataFrame(
        {
            "指标": list(
                evaluation_metrics.keys()
            ),
            "得分": list(
                evaluation_metrics.values()
            ),
        }
    )

    print("\n独立测试集评估结果：")

    print(
        metrics_table.to_string(
            index=False,
            formatters={
                "得分": "{:.4f}".format
            },
        )
    )


if __name__ == "__main__":
    main()