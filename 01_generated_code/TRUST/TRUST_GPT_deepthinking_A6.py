from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ============================================================
# 1. 全局参数
# ============================================================

RANDOM_STATE = 42
DATA_FILE = Path("train_data.csv")

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
NUMERIC_COLUMNS = [
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

ALL_FEATURE_COLUMNS = (
    ONE_HOT_COLUMNS
    + ORDINAL_COLUMNS
    + NUMERIC_COLUMNS
)


# ============================================================
# 2. 读取并检查数据
# ============================================================

def load_and_validate_data(file_path: Path) -> pd.DataFrame:
    """
    读取CSV文件并检查必要字段。

    同时兼容可能误写为 DIAGONSIS 的列名。
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}\n"
            "请将 train_data.csv 放在当前项目的工作目录中，"
            "或者修改 DATA_FILE。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    # 统一清理列名两侧空格
    data.columns = data.columns.astype(str).str.strip()

    # 兼容题目中可能出现的拼写错误
    if "DIAGONSIS" in data.columns and "DIAGNOSIS" not in data.columns:
        data = data.rename(columns={"DIAGONSIS": "DIAGNOSIS"})
        print("提示：已将列名 DIAGONSIS 自动更正为 DIAGNOSIS。")

    required_columns = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据中缺少以下必要列："
            f"{missing_columns}\n"
            f"实际列名为：{data.columns.tolist()}"
        )

    if data.empty:
        raise ValueError("数据集为空，无法训练模型。")

    # 只保留本任务需要的列，避免无关字段进入模型
    data = data[required_columns].copy()

    return data


# ============================================================
# 3. 构造特征和二分类目标
# ============================================================

def prepare_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    将 TRUST 转为数值，并构造二分类目标：
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
            f"警告：目标列中有 {invalid_target_count} 行无法转换为数值，"
            "这些行将被删除。"
        )

        valid_mask = trust_numeric.notna()
        data = data.loc[valid_mask].copy()
        trust_numeric = trust_numeric.loc[valid_mask]

    if data.empty:
        raise ValueError("删除目标值无效的记录后，数据集为空。")

    X = data[ALL_FEATURE_COLUMNS].copy()
    y = (trust_numeric >= 16).astype(int)

    class_counts = y.value_counts().sort_index()

    print("\n二分类目标分布：")
    print(f"TRUST < 16  （类别0）：{class_counts.get(0, 0)}")
    print(f"TRUST >= 16 （类别1）：{class_counts.get(1, 0)}")

    if y.nunique() < 2:
        raise ValueError(
            "目标变量只有一个类别，无法训练二分类模型。"
        )

    # 确保连续变量按数值处理。
    # 无法转换的内容会变成 NaN，随后由中位数填充。
    for column in NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # 分类变量统一转换为字符串，避免混合数据类型导致编码失败
    for column in ONE_HOT_COLUMNS + ORDINAL_COLUMNS:
        X[column] = X[column].astype(str)

    return X, y


# ============================================================
# 4. 创建预处理器
# ============================================================

def create_preprocessor() -> ColumnTransformer:
    """
    创建数据预处理器。

    - SEX、DEPT、DIAGNOSIS：独热编码
    - TPPA：序数编码
    - 连续变量：中位数填充
    """

    one_hot_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    ordinal_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    numeric_transformer = Pipeline(
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
                "one_hot",
                one_hot_transformer,
                ONE_HOT_COLUMNS,
            ),
            (
                "ordinal",
                ordinal_transformer,
                ORDINAL_COLUMNS,
            ),
            (
                "numeric",
                numeric_transformer,
                NUMERIC_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# 5. 创建完整建模管道
# ============================================================

def create_model_pipeline() -> Pipeline:
    """
    将预处理、SMOTE和随机森林放入同一个管道。

    在交叉验证时，SMOTE只会作用于每个训练折，
    不会对验证折或最终测试集进行过采样。
    """
    preprocessor = create_preprocessor()

    smote = SMOTE(
        random_state=RANDOM_STATE,
    )

    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,  # 明确禁止随机森林内部多进程
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("classifier", classifier),
        ]
    )

    return model_pipeline


# ============================================================
# 6. 模型评估
# ============================================================

def evaluate_model(
    model: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    在独立测试集上计算各项评价指标。
    """
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

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
    data = load_and_validate_data(DATA_FILE)
    X, y = prepare_features_and_target(data)

    # 分层划分训练集和测试集，保持类别比例
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n数据集划分：")
    print(f"训练集样本数：{len(X_train)}")
    print(f"测试集样本数：{len(X_test)}")

    model_pipeline = create_model_pipeline()

    # 参数名称前需添加管道步骤名 classifier__
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

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

    print("\n开始进行网格搜索和交叉验证……")
    grid_search.fit(X_train, y_train)

    print("\n最优超参数：")
    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        clean_name = parameter_name.replace(
            "classifier__",
            "",
        )
        print(f"{clean_name}: {parameter_value}")

    print(
        "\n最优模型的交叉验证平均AUC："
        f"{grid_search.best_score_:.4f}"
    )

    test_metrics = evaluate_model(
        grid_search,
        X_test,
        y_test,
    )

    print("\n独立测试集评估结果：")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name:<10}: {metric_value:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n程序运行失败：{error}")
        raise