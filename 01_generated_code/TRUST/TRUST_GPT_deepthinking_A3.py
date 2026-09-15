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
# 1. 基本配置
# =========================
DATA_FILE = Path("train_data.csv")
TARGET_COLUMN = "TRUST"
RANDOM_STATE = 42

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


def load_and_prepare_data(
    file_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取数据并生成二分类目标：
    TRUST >= 16 记为1，否则记为0。
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}"
        )

    data = pd.read_csv(
        file_path,
        encoding="utf-8",
    )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据中缺少以下列："
            f"{missing_columns}\n"
            f"当前数据列为：{data.columns.tolist()}"
        )

    # 仅保留模型需要的列，避免其他无关列进入模型。
    data = data[required_columns].copy()

    # 将目标列转换为数值。
    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    # 无法转换为数值的目标记录不能用于监督学习，因此删除。
    invalid_target_count = int(trust_numeric.isna().sum())

    if invalid_target_count > 0:
        print(
            f"警告：目标列中有 {invalid_target_count} 条记录"
            "无法转换为数值，已删除。"
        )

        valid_mask = trust_numeric.notna()
        data = data.loc[valid_mask].copy()
        trust_numeric = trust_numeric.loc[valid_mask]

    if data.empty:
        raise ValueError("删除无效目标值后，数据集为空。")

    # 构造二分类目标：
    # 1 = TRUST >= 16
    # 0 = TRUST < 16
    y = (trust_numeric >= 16).astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "目标变量只有一个类别，无法进行二分类建模。"
        )

    X = data[FEATURE_COLUMNS].copy()

    # 连续变量统一转换为数值。
    # 无法转换的值会变为NaN，随后由中位数填充。
    for column in CONTINUOUS_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X, y


def build_model_pipeline() -> Pipeline:
    """
    构建预处理、SMOTE和随机森林分类管道。
    """

    # 独热编码：
    # handle_unknown="ignore"用于处理测试集或新数据中的未知类别。
    # sparse_output=False确保输出为稠密数组，便于SMOTE处理。
    one_hot_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    # TPPA序数编码。
    # 未知类别编码为-1，防止预测时因新类别报错。
    ordinal_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    # 连续变量使用中位数填充。
    continuous_transformer = SimpleImputer(
        strategy="median",
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
                "continuous",
                continuous_transformer,
                CONTINUOUS_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                ),
            ),
            ("classifier", classifier),
        ]
    )

    return pipeline


def evaluate_model(
    model: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    在独立测试集上计算模型评估指标。
    """
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
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
    # 2. 读取并处理数据
    # =========================
    X, y = load_and_prepare_data(DATA_FILE)

    print("数据集基本信息：")
    print(f"总样本数：{len(X)}")
    print(f"特征数：{X.shape[1]}")
    print("\n二分类目标分布：")
    print(y.value_counts().sort_index())
    print("\n二分类目标比例：")
    print(y.value_counts(normalize=True).sort_index())

    # =========================
    # 3. 划分训练集和测试集
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\n训练集样本数：{len(X_train)}")
    print(f"测试集样本数：{len(X_test)}")

    # =========================
    # 4. 构建模型管道
    # =========================
    pipeline = build_model_pipeline()

    # 参数名称需要使用：
    # 管道步骤名__模型参数名
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

    # 以AUC作为最优模型选择指标。
    # n_jobs=1明确禁止多进程。
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=1,
        refit=True,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    # =========================
    # 5. 超参数搜索和模型训练
    # =========================
    grid_search.fit(
        X_train,
        y_train,
    )

    print("\n最优超参数：")
    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    print(
        "\n交叉验证最优平均AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # =========================
    # 6. 独立测试集评估
    # =========================
    test_metrics = evaluate_model(
        model=grid_search,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n独立测试集评估结果：")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()