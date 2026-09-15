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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# =========================
# 基本设置
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_FILE = Path("train_data.csv")
TARGET_COLUMN = "TRUST"

# 注意：这里使用的是 DIAGNOSIS。
# 题目后半部分出现的 DIAGONSIS 应为拼写错误。
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


def parse_titer(series: pd.Series) -> pd.Series:
    """
    将滴度转换为数值。

    支持：
    1. 数值形式：1、2、4、8、16
    2. 字符串形式："1"、"16"
    3. 比例形式："1:16" 或 "1：16"

    无法转换的值将变为 NaN。
    """
    numeric_result = pd.to_numeric(series, errors="coerce")

    failed_mask = numeric_result.isna() & series.notna()

    if failed_mask.any():
        text_values = (
            series.astype("string")
            .str.strip()
            .str.replace("：", ":", regex=False)
        )

        # 提取字符串末尾的数字，例如从“1:16”中提取16
        extracted_values = text_values.str.extract(
            r"(\d+(?:\.\d+)?)\s*$",
            expand=False,
        )

        numeric_result.loc[failed_mask] = pd.to_numeric(
            extracted_values.loc[failed_mask],
            errors="coerce",
        )

    return numeric_result


def load_and_prepare_data(
    file_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取数据，检查字段，并构造二分类目标：
    TRUST >= 16 记为1，否则记为0。
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{file_path.resolve()}"
        )

    data = pd.read_csv(
        file_path,
        encoding="utf-8",
    )

    # 去除列名前后的空格
    data.columns = data.columns.str.strip()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            "数据集中缺少以下列："
            f"{missing_columns}\n"
            "请特别检查 DIAGNOSIS 的拼写。"
        )

    if data.columns[-1] != TARGET_COLUMN:
        warnings.warn(
            f"数据集最后一列是 {data.columns[-1]!r}，"
            f"不是 {TARGET_COLUMN!r}。代码仍将使用 "
            f"{TARGET_COLUMN!r} 作为目标列。",
            stacklevel=2,
        )

    # 将TRUST滴度转换为数值
    trust_numeric = parse_titer(data[TARGET_COLUMN])

    invalid_target_count = int(trust_numeric.isna().sum())

    if invalid_target_count > 0:
        warnings.warn(
            f"目标列中有 {invalid_target_count} 个值无法转换为滴度，"
            "这些记录将被删除。",
            stacklevel=2,
        )

    valid_target_mask = trust_numeric.notna()

    data = data.loc[valid_target_mask].copy()
    trust_numeric = trust_numeric.loc[valid_target_mask]

    # 二分类标签：TRUST >= 16 为1，否则为0
    target = (trust_numeric >= 16).astype(int)
    target.name = "TRUST_GE_16"

    features = data.loc[:, FEATURE_COLUMNS].copy()

    # 检查分类变量缺失值
    categorical_columns = ONE_HOT_COLUMNS + ORDINAL_COLUMNS
    categorical_missing = (
        features[categorical_columns]
        .isna()
        .sum()
    )

    categorical_missing = categorical_missing[
        categorical_missing > 0
    ]

    if not categorical_missing.empty:
        raise ValueError(
            "分类变量中检测到缺失值，但当前要求为分类变量无缺失值：\n"
            f"{categorical_missing.to_string()}"
        )

    # 分类变量统一转换为字符串，避免同一列中数字和字符串混合
    features[categorical_columns] = (
        features[categorical_columns].astype(str)
    )

    # 连续变量转换为数值
    # 无法转换的内容变为NaN，之后由中位数填充
    for column in CONTINUOUS_COLUMNS:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    class_counts = target.value_counts().sort_index()

    if len(class_counts) != 2:
        raise ValueError(
            "目标变量转换后没有同时包含两个类别。\n"
            "请检查TRUST列是否同时包含小于16和大于等于16的记录。\n"
            f"当前类别分布：\n{class_counts.to_string()}"
        )

    if class_counts.min() < 2:
        raise ValueError(
            "少数类别样本数不足，无法进行分层训练集和测试集划分。\n"
            f"当前类别分布：\n{class_counts.to_string()}"
        )

    return features, target


def create_cv_and_smote_k(
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """
    根据训练集中少数类别样本量动态确定：
    1. 分层交叉验证折数
    2. SMOTE的k_neighbors

    这样可以降低少数类别样本较少时SMOTE报错的风险。
    """
    minority_count = int(target_train.value_counts().min())
    number_of_splits = min(5, minority_count)

    if number_of_splits < 2:
        raise ValueError(
            "训练集中的少数类别样本不足，无法进行交叉验证。"
        )

    cv_splitter = StratifiedKFold(
        n_splits=number_of_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_splits = list(
        cv_splitter.split(features_train, target_train)
    )

    # 计算所有交叉验证训练折中最小的少数类别样本数
    minimum_fold_minority_count = min(
        int(
            target_train.iloc[train_indices]
            .value_counts()
            .min()
        )
        for train_indices, _ in cv_splits
    )

    if minimum_fold_minority_count < 2:
        raise ValueError(
            "交叉验证的某个训练折中少数类别样本少于2个，"
            "无法执行SMOTE。请增加少数类别样本量。"
        )

    smote_k_neighbors = min(
        5,
        minimum_fold_minority_count - 1,
    )

    return cv_splits, smote_k_neighbors


def build_pipeline(
    smote_k_neighbors: int,
) -> Pipeline:
    """
    构建数据预处理、SMOTE和随机森林流水线。
    """
    one_hot_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    ordinal_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
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
        sparse_threshold=0.0,
    )

    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,  # 不使用多进程
    )

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
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
                classifier,
            ),
        ]
    )

    return model_pipeline


def evaluate_model(
    model: Pipeline,
    features_test: pd.DataFrame,
    target_test: pd.Series,
) -> pd.DataFrame:
    """
    在独立测试集上计算指定的分类评价指标。
    """
    predicted_class = model.predict(features_test)
    predicted_probability = model.predict_proba(
        features_test
    )[:, 1]

    evaluation_results = pd.DataFrame(
        {
            "指标": [
                "Accuracy",
                "Recall",
                "Precision",
                "F1-score",
                "AUC",
            ],
            "数值": [
                accuracy_score(
                    target_test,
                    predicted_class,
                ),
                recall_score(
                    target_test,
                    predicted_class,
                    zero_division=0,
                ),
                precision_score(
                    target_test,
                    predicted_class,
                    zero_division=0,
                ),
                f1_score(
                    target_test,
                    predicted_class,
                    zero_division=0,
                ),
                roc_auc_score(
                    target_test,
                    predicted_probability,
                ),
            ],
        }
    )

    return evaluation_results


def main() -> None:
    # 1. 读取并处理数据
    features, target = load_and_prepare_data(DATA_FILE)

    print("=" * 60)
    print("完整数据集类别分布")
    print("0：TRUST < 16")
    print("1：TRUST >= 16")
    print(target.value_counts().sort_index().to_string())
    print("=" * 60)

    # 2. 划分训练集和测试集
    features_train, features_test, target_train, target_test = (
        train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )
    )

    print("\n训练集类别分布：")
    print(
        target_train.value_counts()
        .sort_index()
        .to_string()
    )

    print("\n测试集类别分布：")
    print(
        target_test.value_counts()
        .sort_index()
        .to_string()
    )

    # 3. 确定交叉验证折数和安全的SMOTE参数
    cv_splits, smote_k_neighbors = create_cv_and_smote_k(
        features_train,
        target_train,
    )

    print(f"\n交叉验证折数：{len(cv_splits)}")
    print(
        f"SMOTE k_neighbors：{smote_k_neighbors}"
    )

    # 4. 构建流水线
    pipeline = build_pipeline(smote_k_neighbors)

    # 5. 设置超参数搜索范围
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    # 以AUC作为网格搜索最优模型的选择标准
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv_splits,
        n_jobs=1,  # 不使用多进程
        refit=True,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    # 6. 模型训练与超参数搜索
    grid_search.fit(
        features_train,
        target_train,
    )

    print("\n" + "=" * 60)
    print("最优超参数")
    print("=" * 60)

    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        clean_name = parameter_name.replace(
            "classifier__",
            "",
        )
        print(f"{clean_name}: {parameter_value}")

    print(
        "\n交叉验证最优平均AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # 7. 在独立测试集上评估最优模型
    best_model = grid_search.best_estimator_

    evaluation_results = evaluate_model(
        best_model,
        features_test,
        target_test,
    )

    print("\n" + "=" * 60)
    print("独立测试集评估结果")
    print("=" * 60)
    print(
        evaluation_results.to_string(
            index=False,
            formatters={
                "数值": lambda value: f"{value:.4f}"
            },
        )
    )


if __name__ == "__main__":
    main()