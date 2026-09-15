"""
随机森林二分类模型
目标：预测 TRUST 是否 >= 16

适用环境：
- Python 3.10+
- PyCharm 2025.2.3
- pandas
- scikit-learn
- imbalanced-learn
"""

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ============================================================
# 1. 参数设置
# ============================================================

DATA_FILE = Path("train_data.csv")

TARGET_COLUMN = "TRUST"

ONE_HOT_COLUMNS = [
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
    ONE_HOT_COLUMNS
    + ORDINAL_COLUMNS
    + CONTINUOUS_COLUMNS
)

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# TPPA 的顺序设置：
# 如果 TPPA 有明确的医学顺序，建议将下面的 "auto" 改成实际顺序。
#
# 示例：
# TPPA_CATEGORIES = [["阴性", "弱阳性", "阳性"]]
#
# 如果 TPPA 是数字分类，例如 0、1、2：
# TPPA_CATEGORIES = [[0, 1, 2]]
#
# 当前设置会让 OrdinalEncoder 自动根据训练数据确定类别顺序。
TPPA_CATEGORIES = "auto"


# ============================================================
# 2. 读取并检查数据
# ============================================================

def load_data(file_path: Path) -> pd.DataFrame:
    """读取并检查原始数据。"""

    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}\n"
            "请将 train_data.csv 放在当前项目运行目录下。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    # 去除列名首尾可能存在的空格
    data.columns = data.columns.str.strip()

    # 兼容需求描述中可能出现的 DIAGONSIS 拼写
    if "DIAGNOSIS" not in data.columns and "DIAGONSIS" in data.columns:
        data = data.rename(columns={"DIAGONSIS": "DIAGNOSIS"})
        print(
            "提示：检测到列名 DIAGONSIS，"
            "已自动重命名为 DIAGNOSIS。"
        )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要列："
            f"{missing_columns}\n"
            f"当前实际列名：{data.columns.tolist()}"
        )

    if data.columns[-1] != TARGET_COLUMN:
        print(
            f"提示：目标列 {TARGET_COLUMN} 不是数据集最后一列，"
            "但程序仍会按照列名读取该目标列。"
        )

    # 只保留本次建模需要使用的列
    data = data[required_columns].copy()

    return data


# ============================================================
# 3. 构造自变量和二分类目标变量
# ============================================================

def prepare_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    将 TRUST 转换为数值，并构造二分类目标：
    TRUST >= 16 -> 1
    TRUST < 16  -> 0
    """

    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    invalid_target_count = int(trust_numeric.isna().sum())

    if invalid_target_count > 0:
        print(
            f"警告：目标列中有 {invalid_target_count} 条记录"
            "无法转换为数值，这些记录将被删除。"
        )

        valid_mask = trust_numeric.notna()
        data = data.loc[valid_mask].copy()
        trust_numeric = trust_numeric.loc[valid_mask]

    if data.empty:
        raise ValueError("删除无效目标值后，数据集为空。")

    x = data[FEATURE_COLUMNS].copy()
    y = (trust_numeric >= 16).astype(int)

    class_counts = y.value_counts().sort_index()

    print("\n二分类目标分布：")
    print(f"TRUST < 16  （类别 0）：{class_counts.get(0, 0)}")
    print(f"TRUST >= 16 （类别 1）：{class_counts.get(1, 0)}")

    if y.nunique() < 2:
        raise ValueError(
            "目标变量只有一个类别，无法训练二分类模型。"
        )

    return x, y


# ============================================================
# 4. 构建数据预处理器
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """创建分类变量编码和连续变量缺失值处理流程。"""

    # SEX、DEPT、DIAGNOSIS：独热编码
    one_hot_pipeline = Pipeline(
        steps=[
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # TPPA：序数编码
    ordinal_pipeline = Pipeline(
        steps=[
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    categories=TPPA_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    # 连续变量：中位数填充
    continuous_pipeline = Pipeline(
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
                "one_hot_features",
                one_hot_pipeline,
                ONE_HOT_COLUMNS,
            ),
            (
                "ordinal_features",
                ordinal_pipeline,
                ORDINAL_COLUMNS,
            ),
            (
                "continuous_features",
                continuous_pipeline,
                CONTINUOUS_COLUMNS,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# 5. 构建包含 SMOTE 的模型管道
# ============================================================

def build_model_pipeline() -> ImbPipeline:
    """
    建立完整管道：
    数据预处理 -> SMOTE -> 随机森林
    """

    preprocessor = build_preprocessor()

    smote = SMOTE(
        random_state=RANDOM_STATE,
    )

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    model_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("classifier", random_forest),
        ]
    )

    return model_pipeline


# ============================================================
# 6. 模型评估
# ============================================================

def evaluate_model(
    model: GridSearchCV,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """在独立测试集上计算指定评估指标。"""

    y_pred = model.predict(x_test)
    y_probability = model.predict_proba(x_test)[:, 1]

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


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    data = load_data(DATA_FILE)
    x, y = prepare_features_and_target(data)

    # 先划分训练集和测试集。
    # SMOTE 只会在训练集的交叉验证折内执行，不会处理测试集。
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n数据集划分：")
    print(f"训练集样本数：{len(x_train)}")
    print(f"测试集样本数：{len(x_test)}")

    model_pipeline = build_model_pipeline()

    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    cross_validator = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 同时计算多项交叉验证指标，以 AUC 选择最佳模型。
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
        cv=cross_validator,
        n_jobs=1,  # 明确禁用多进程
        return_train_score=False,
        error_score="raise",
        verbose=1,
    )

    print("\n开始进行网格搜索和交叉验证……")
    grid_search.fit(x_train, y_train)

    print("\n最佳超参数：")
    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        clean_name = parameter_name.replace(
            "classifier__",
            "",
        )
        print(f"{clean_name}: {parameter_value}")

    print(
        "\n最佳交叉验证平均 AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # 输出最佳参数对应的各项交叉验证得分
    best_index = grid_search.best_index_

    print("\n最佳模型的交叉验证平均指标：")
    print(
        "Accuracy : "
        f"{grid_search.cv_results_['mean_test_accuracy'][best_index]:.4f}"
    )
    print(
        "Recall   : "
        f"{grid_search.cv_results_['mean_test_recall'][best_index]:.4f}"
    )
    print(
        "Precision: "
        f"{grid_search.cv_results_['mean_test_precision'][best_index]:.4f}"
    )
    print(
        "F1-score : "
        f"{grid_search.cv_results_['mean_test_f1'][best_index]:.4f}"
    )
    print(
        "AUC      : "
        f"{grid_search.cv_results_['mean_test_roc_auc'][best_index]:.4f}"
    )

    test_metrics = evaluate_model(
        grid_search,
        x_test,
        y_test,
    )

    print("\n独立测试集评估结果：")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name:<9}: {metric_value:.4f}")


if __name__ == "__main__":
    main()