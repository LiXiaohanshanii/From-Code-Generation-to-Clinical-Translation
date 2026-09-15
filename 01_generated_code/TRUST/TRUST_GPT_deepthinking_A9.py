from pathlib import Path
import math

import numpy as np
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
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# =========================
# 基本设置
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20

# train_data.csv 与本 Python 文件位于同一目录
DATA_FILE = Path(__file__).resolve().parent / "train_data.csv"

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

TARGET_COLUMN = "TRUST"

# 如果已知 TPPA 的真实临床顺序，请在这里填写。
# 示例：
# TPPA_ORDER = ["阴性", "弱阳性", "阳性", "强阳性"]
#
# 设置为 None 时，OrdinalEncoder 会根据数据自动确定类别顺序。
TPPA_ORDER = None


def convert_numeric_strict(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    将一列转换为数值型。

    原始缺失值会保留为 NaN；
    非缺失但无法转换为数值的内容会触发异常，
    避免错误数据被当作缺失值进行中位数填充。
    """
    converted = pd.to_numeric(series, errors="coerce")

    invalid_mask = series.notna() & converted.isna()

    if invalid_mask.any():
        invalid_examples = (
            series.loc[invalid_mask]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        raise ValueError(
            f"列 {column_name!r} 中存在无法转换为数值的内容，"
            f"示例：{invalid_examples}"
        )

    return converted


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """读取数据、检查字段并生成二分类目标变量。"""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{DATA_FILE}\n"
            "请确认 train_data.csv 与本 Python 文件位于同一目录。"
        )

    data = pd.read_csv(
        DATA_FILE,
        encoding="utf-8",
    )

    if data.empty:
        raise ValueError("train_data.csv 中没有可用数据。")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"数据中缺少以下字段：{missing_columns}\n"
            f"当前数据字段为：{data.columns.tolist()}"
        )

    if data.columns[-1] != TARGET_COLUMN:
        raise ValueError(
            f"目标列应当是数据集最后一列 {TARGET_COLUMN!r}，"
            f"但当前最后一列是 {data.columns[-1]!r}。"
        )

    # 分类变量按要求不允许存在缺失值
    categorical_columns = ONEHOT_COLUMNS + ORDINAL_COLUMNS
    categorical_missing = data[categorical_columns].isna().sum()
    categorical_missing = categorical_missing[
        categorical_missing > 0
    ]

    if not categorical_missing.empty:
        raise ValueError(
            "以下分类变量存在缺失值，但任务设定中分类变量应无缺失值：\n"
            f"{categorical_missing.to_string()}"
        )

    # 将分类变量统一转换为字符串，避免同一列混合数字和文本
    for column in categorical_columns:
        data[column] = data[column].astype(str)

    # 连续变量转换为数值；原有缺失值后续由中位数填充
    for column in CONTINUOUS_COLUMNS:
        data[column] = convert_numeric_strict(
            data[column],
            column,
        )

    # TRUST 必须能够转换为数值
    trust_numeric = convert_numeric_strict(
        data[TARGET_COLUMN],
        TARGET_COLUMN,
    )

    if trust_numeric.isna().any():
        missing_count = int(trust_numeric.isna().sum())
        raise ValueError(
            f"目标列 {TARGET_COLUMN!r} 中存在 {missing_count} 个缺失值。"
        )

    # 二分类目标：
    # 0 表示 TRUST < 16
    # 1 表示 TRUST >= 16
    target = (trust_numeric >= 16).astype(int)
    target.name = "TRUST_GE_16"

    class_counts = target.value_counts().sort_index()

    if target.nunique() != 2:
        raise ValueError(
            "转换后的目标变量不包含两个类别。"
            "请确认 TRUST 中同时存在小于16和大于等于16的样本。\n"
            f"类别计数：\n{class_counts.to_string()}"
        )

    if class_counts.min() < 2:
        raise ValueError(
            "至少有一个类别的样本数少于2，无法进行分层训练集/测试集划分。\n"
            f"类别计数：\n{class_counts.to_string()}"
        )

    features = data[FEATURE_COLUMNS].copy()

    print("数据读取完成")
    print(f"样本数：{len(data)}")
    print(f"特征数：{len(FEATURE_COLUMNS)}")
    print("转换后的目标变量分布：")
    print(f"  TRUST < 16 ：{class_counts.get(0, 0)}")
    print(f"  TRUST >= 16：{class_counts.get(1, 0)}")

    return features, target


def build_ordinal_encoder() -> OrdinalEncoder:
    """创建 TPPA 的序数编码器。"""
    common_parameters = {
        "handle_unknown": "use_encoded_value",
        "unknown_value": -1,
    }

    if TPPA_ORDER is None:
        return OrdinalEncoder(
            **common_parameters,
        )

    categories = [
        [str(value) for value in TPPA_ORDER]
    ]

    return OrdinalEncoder(
        categories=categories,
        **common_parameters,
    )


def calculate_cv_and_smote_parameters(
    y_train: pd.Series,
) -> tuple[int, int]:
    """
    根据训练集中少数类样本数，确定交叉验证折数和
    SMOTE 的 k_neighbors。

    这样可以降低小样本数据中 SMOTE 因邻居数不足而报错的概率。
    """
    train_class_counts = y_train.value_counts()
    minority_count = int(train_class_counts.min())

    if minority_count < 3:
        raise ValueError(
            "训练集中少数类样本数少于3，"
            "无法在交叉验证内部稳定执行 SMOTE。\n"
            f"训练集类别计数：\n{train_class_counts.to_string()}"
        )

    cv_splits = min(5, minority_count)

    # 在某个交叉验证训练折中可能出现的最少少数类样本数
    minimum_minority_in_cv_training = (
        minority_count
        - math.ceil(minority_count / cv_splits)
    )

    smote_k_neighbors = min(
        5,
        minimum_minority_in_cv_training - 1,
    )

    if smote_k_neighbors < 1:
        raise ValueError(
            "少数类样本数不足，无法为 SMOTE 设置有效的近邻数。"
        )

    return cv_splits, smote_k_neighbors


def build_model_pipeline(
    smote_k_neighbors: int,
) -> Pipeline:
    """构建预处理、SMOTE和随机森林组成的完整管道。"""
    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    ordinal_encoder = build_ordinal_encoder()

    continuous_transformer = SimpleImputer(
        strategy="median",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                onehot_encoder,
                ONEHOT_COLUMNS,
            ),
            (
                "ordinal",
                ordinal_encoder,
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

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,

        # 明确禁用多进程
        n_jobs=1,
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
                random_forest,
            ),
        ]
    )

    return model_pipeline


def train_and_evaluate() -> None:
    """完成数据划分、超参数搜索和测试集评估。"""
    features, target = load_and_prepare_data()

    try:
        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )
    except ValueError as error:
        raise ValueError(
            "训练集/测试集分层划分失败。"
            "请检查样本总数及两个类别的样本数。"
        ) from error

    print("\n数据集划分完成")
    print(f"训练集样本数：{len(x_train)}")
    print(f"测试集样本数：{len(x_test)}")
    print("训练集类别分布：")
    print(y_train.value_counts().sort_index().to_string())

    cv_splits, smote_k_neighbors = (
        calculate_cv_and_smote_parameters(y_train)
    )

    print(f"\n交叉验证折数：{cv_splits}")
    print(f"SMOTE k_neighbors：{smote_k_neighbors}")

    pipeline = build_model_pipeline(
        smote_k_neighbors=smote_k_neighbors,
    )

    # 完全按照题目提供的参数范围进行搜索
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    # 网格搜索过程中同时计算各项指标，
    # 使用 AUC 选择并重新训练最佳模型
    scoring = {
        "accuracy": "accuracy",
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
    }

    cross_validation = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring=scoring,
        refit="roc_auc",
        cv=cross_validation,

        # 明确禁用多进程
        n_jobs=1,

        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    print("\n开始网格搜索和模型训练……")
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    print("\n最佳参数：")
    for parameter, value in grid_search.best_params_.items():
        clean_name = parameter.replace(
            "classifier__",
            "",
        )
        print(f"  {clean_name}: {value}")

    print(
        "\n最佳交叉验证平均AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # 独立测试集预测
    y_pred = best_model.predict(x_test)
    y_probability = best_model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    if y_test.nunique() == 2:
        auc = roc_auc_score(
            y_test,
            y_probability,
        )
    else:
        auc = np.nan

    print("\n独立测试集评估结果")
    print("=" * 35)
    print(f"Accuracy ：{accuracy:.4f}")
    print(f"Recall   ：{recall:.4f}")
    print(f"Precision：{precision:.4f}")
    print(f"F1-score ：{f1:.4f}")

    if np.isnan(auc):
        print("AUC       ：无法计算（测试集仅包含一个类别）")
    else:
        print(f"AUC       ：{auc:.4f}")


def main() -> None:
    try:
        train_and_evaluate()
    except Exception as error:
        print("\n程序运行失败：")
        print(error)
        raise


if __name__ == "__main__":
    main()