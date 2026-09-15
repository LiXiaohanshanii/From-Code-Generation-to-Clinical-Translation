from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# =========================
# 基本配置
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20
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

FEATURE_COLUMNS = ONE_HOT_COLUMNS + ORDINAL_COLUMNS + CONTINUOUS_COLUMNS


def load_and_prepare_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取数据并创建二分类目标变量。

    原始目标：
        TRUST = 1、2、4、8、16、32……

    新目标：
        TRUST >= 16 -> 1
        TRUST < 16  -> 0
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}\n"
            "请将 train_data.csv 放在当前项目工作目录中。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    # 兼容可能出现的拼写错误 DIAGONSIS
    if "DIAGNOSIS" not in data.columns and "DIAGONSIS" in data.columns:
        data = data.rename(columns={"DIAGONSIS": "DIAGNOSIS"})
        print("提示：已将列名 DIAGONSIS 自动更正为 DIAGNOSIS。")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要列："
            + ", ".join(missing_columns)
        )

    # 只保留建模所需列
    data = data[required_columns].copy()

    # 将连续变量转换为数值。
    # 无法转换的内容会变成 NaN，随后由中位数填充。
    for column in CONTINUOUS_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # TRUST 转换为数值
    trust_numeric = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")

    if trust_numeric.isna().any():
        invalid_count = int(trust_numeric.isna().sum())
        raise ValueError(
            f"目标列 {TARGET_COLUMN} 中有 {invalid_count} 个值无法转换为数值。"
        )

    # 构建二分类目标
    target = (trust_numeric >= 16).astype(int)
    features = data[FEATURE_COLUMNS].copy()

    # 检查是否同时存在两个类别
    if target.nunique() != 2:
        raise ValueError(
            "转换后的目标变量不包含两个类别。"
            "请检查 TRUST 列中是否同时存在小于16和大于等于16的样本。"
        )

    return features, target


def build_pipeline() -> Pipeline:
    """
    构建数据预处理、SMOTE和随机森林流水线。
    """

    # 连续变量：中位数填充
    continuous_transformer = Pipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    # SEX、DEPT、DIAGNOSIS：独热编码
    one_hot_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    # TPPA：序数编码
    #
    # categories="auto" 会根据训练数据自动识别类别。
    # 若TPPA有明确的医学等级顺序，可改为：
    # OrdinalEncoder(
    #     categories=[["阴性", "弱阳性", "阳性", "强阳性"]],
    #     handle_unknown="use_encoded_value",
    #     unknown_value=-1
    # )
    ordinal_transformer = OrdinalEncoder(
        categories="auto",
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot_categorical",
                one_hot_transformer,
                ONE_HOT_COLUMNS,
            ),
            (
                "ordinal_categorical",
                ordinal_transformer,
                ORDINAL_COLUMNS,
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

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,  # 明确禁止多进程
    )

    # SMOTE位于预处理之后、模型之前。
    # 在GridSearchCV中，每个训练折单独执行SMOTE，可防止数据泄漏。
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(random_state=RANDOM_STATE),
            ),
            ("classifier", model),
        ]
    )

    return pipeline


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    在独立测试集上评估模型。
    """
    y_pred = model.predict(x_test)
    y_probability = model.predict_proba(x_test)[:, 1]

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

    return metrics


def main() -> None:
    # 1. 读取并准备数据
    x, y = load_and_prepare_data(DATA_FILE)

    print("=" * 60)
    print("原始数据基本信息")
    print("=" * 60)
    print(f"样本数量：{len(x)}")
    print(f"特征数量：{x.shape[1]}")
    print("\n二分类目标分布：")
    print(
        pd.DataFrame(
            {
                "样本数": y.value_counts().sort_index(),
                "比例": y.value_counts(
                    normalize=True
                ).sort_index(),
            }
        ).rename(
            index={
                0: "TRUST < 16",
                1: "TRUST >= 16",
            }
        )
    )

    # 2. 划分训练集和测试集
    # stratify=y 保持训练集、测试集中的类别比例基本一致
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n训练集样本数：", len(x_train))
    print("测试集样本数：", len(x_test))

    # 检查少数类样本数，默认SMOTE至少需要6个少数类样本
    minority_count = int(y_train.value_counts().min())

    if minority_count < 6:
        raise ValueError(
            "训练集中的少数类样本少于6个，默认SMOTE无法执行。"
            "可以增加数据量，或根据样本数量调小SMOTE的k_neighbors参数。"
        )

    # 3. 构建流水线
    pipeline = build_pipeline()

    # 4. 设置超参数搜索范围
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    # 分层五折交叉验证
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 5. 网格搜索
    #
    # refit="roc_auc"：
    # 使用交叉验证AUC选择最佳参数，并自动使用最佳参数
    # 在完整训练集上重新拟合模型。
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring={
            "accuracy": "accuracy",
            "recall": "recall",
            "precision": "precision",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="roc_auc",
        cv=cross_validation,
        n_jobs=1,  # 不使用多进程
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    print("\n" + "=" * 60)
    print("开始进行超参数搜索")
    print("=" * 60)

    grid_search.fit(x_train, y_train)

    print("\n" + "=" * 60)
    print("超参数搜索结果")
    print("=" * 60)
    print("最佳参数：")

    for parameter, value in grid_search.best_params_.items():
        clean_name = parameter.replace("classifier__", "")
        print(f"  {clean_name}: {value}")

    print(
        f"\n最佳交叉验证AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # 6. 独立测试集评估
    best_model = grid_search.best_estimator_
    test_metrics = evaluate_model(
        best_model,
        x_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("独立测试集评估结果")
    print("=" * 60)

    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name:<12}: {metric_value:.4f}")


if __name__ == "__main__":
    main()