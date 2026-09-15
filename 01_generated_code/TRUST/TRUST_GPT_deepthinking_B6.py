from pathlib import Path
import warnings

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


# =========================
# 基本参数
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20
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

# 若 TPPA 存在明确的临床等级顺序，可在此填写。
# 示例：
# TPPA_ORDER = ["阴性", "弱阳性", "阳性"]
# TPPA_ORDER = [1, 2, 4, 8, 16, 32]
#
# 设置为 None 时，OrdinalEncoder 根据数据中的类别自动确定顺序。
TPPA_ORDER = None


def make_one_hot_encoder() -> OneHotEncoder:
    """
    创建独热编码器。

    sparse_output=False：
    将独热编码结果转换为稠密数组，便于后续执行 SMOTE。

    try/except 用于兼容不同版本的 scikit-learn。
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        # 兼容较旧版本的 scikit-learn
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def load_and_validate_data(
    csv_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取并校验数据。

    将 TRUST 转换为二分类目标：
        0：TRUST < 16
        1：TRUST >= 16
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{csv_path}"
        )

    data = pd.read_csv(
        csv_path,
        encoding="utf-8",
    )

    # 去除列名前后的空格
    data.columns = data.columns.astype(str).str.strip()

    # 兼容需求描述中可能出现的 DIAGONSIS 拼写错误
    if (
        "DIAGNOSIS" not in data.columns
        and "DIAGONSIS" in data.columns
    ):
        warnings.warn(
            "检测到列名 DIAGONSIS，"
            "已自动重命名为 DIAGNOSIS。",
            stacklevel=2,
        )

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
            f"数据缺少以下必要列：{missing_columns}"
        )

    if data.columns[-1] != TARGET_COLUMN:
        warnings.warn(
            f"{TARGET_COLUMN} 不是数据集最后一列。"
            f"程序仍将按照列名 {TARGET_COLUMN!r} 读取目标变量。",
            stacklevel=2,
        )

    # 将 TRUST 转换为数值
    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    if trust_numeric.isna().any():
        invalid_values = (
            data.loc[
                trust_numeric.isna(),
                TARGET_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "TRUST 列存在缺失值或无法转换为数值的内容，"
            f"示例：{invalid_values[:10]}"
        )

    # 只选取指定特征
    X = data.loc[:, FEATURE_COLUMNS].copy()

    # 构建二分类目标
    y = (trust_numeric >= 16).astype(int)
    y.name = "TRUST_GE_16"

    class_counts = y.value_counts().sort_index()

    if set(class_counts.index) != {0, 1}:
        raise ValueError(
            "目标变量必须同时包含 TRUST < 16 和 "
            "TRUST >= 16 两个类别。"
            f"当前类别计数：{class_counts.to_dict()}"
        )

    if class_counts.min() < 2:
        raise ValueError(
            "每个类别至少需要两个样本，"
            "才能进行分层训练集和测试集划分。"
            f"当前类别计数：{class_counts.to_dict()}"
        )

    print(
        "原始目标类别计数"
        "（0：TRUST < 16，1：TRUST >= 16）："
    )
    print(class_counts.to_string())

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    构建数据预处理器。

    1. SEX、DEPT、DIAGNOSIS：独热编码
    2. TPPA：序数编码
    3. 连续变量：中位数填充
    """
    continuous_pipeline = Pipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    ordinal_encoder_parameters = {
        "handle_unknown": "use_encoded_value",
        "unknown_value": -1,
    }

    if TPPA_ORDER is not None:
        ordinal_encoder_parameters["categories"] = [
            TPPA_ORDER
        ]

    ordinal_encoder = OrdinalEncoder(
        **ordinal_encoder_parameters
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot",
                make_one_hot_encoder(),
                ONE_HOT_COLUMNS,
            ),
            (
                "ordinal",
                ordinal_encoder,
                ORDINAL_COLUMNS,
            ),
            (
                "continuous",
                continuous_pipeline,
                CONTINUOUS_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def choose_cv_and_smote_k(
    y_train: pd.Series,
) -> tuple[StratifiedKFold, int]:
    """
    根据训练集中的少数类样本数，设置：

    1. 分层交叉验证折数
    2. SMOTE 的 k_neighbors

    这样可以降低少数类样本较少时，SMOTE 在某个交叉验证
    训练折中因邻居数量不足而报错的风险。
    """
    minority_count = int(
        y_train.value_counts().min()
    )

    if minority_count < 3:
        raise ValueError(
            "训练集中的少数类样本不足 3 个，"
            "无法同时可靠执行分层交叉验证和 SMOTE。"
        )

    # 最多使用 5 折交叉验证
    n_splits = min(5, minority_count)

    # 估算交叉验证中验证折最多包含的少数类样本数
    largest_validation_minority_count = int(
        np.ceil(minority_count / n_splits)
    )

    # 估算最小训练折中的少数类样本数
    minimum_cv_train_minority_count = (
        minority_count
        - largest_validation_minority_count
    )

    if minimum_cv_train_minority_count < 2:
        raise ValueError(
            "交叉验证训练折中的少数类样本不足，"
            "无法执行 SMOTE。"
        )

    smote_k_neighbors = min(
        5,
        minimum_cv_train_minority_count - 1,
    )

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return cross_validation, smote_k_neighbors


def main() -> None:
    # 默认 train_data.csv 与当前脚本位于同一目录
    csv_path = (
        Path(__file__).resolve().parent
        / "train_data.csv"
    )

    # 读取数据并构建二分类目标
    X, y = load_and_validate_data(csv_path)

    # 分层划分训练集和测试集
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    print("\n训练集类别计数：")
    print(
        y_train
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n测试集类别计数：")
    print(
        y_test
        .value_counts()
        .sort_index()
        .to_string()
    )

    cross_validation, smote_k_neighbors = (
        choose_cv_and_smote_k(y_train)
    )

    print(
        f"\n交叉验证折数："
        f"{cross_validation.n_splits}"
    )
    print(
        f"SMOTE k_neighbors："
        f"{smote_k_neighbors}"
    )

    # 使用 imblearn 的 Pipeline。
    # 每次交叉验证时，SMOTE 只作用于当前训练折，
    # 不会作用于验证折或最终测试集。
    model_pipeline = ImbPipeline(
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
                    n_jobs=1,
                ),
            ),
        ]
    )

    # 随机森林超参数范围
    parameter_grid = {
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

    # 以 AUC 作为网格搜索的优化指标
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=1,          # 不使用多进程
        pre_dispatch=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )

    print("\n开始网格搜索和模型训练……")

    grid_search.fit(
        X_train,
        y_train,
    )

    # 获取最优完整管道
    best_model = grid_search.best_estimator_

    # 测试集类别预测
    y_pred = best_model.predict(X_test)

    # TRUST >= 16 的预测概率
    y_probability = best_model.predict_proba(
        X_test
    )[:, 1]

    # 计算评价指标
    evaluation_metrics = {
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

    print("\n最佳超参数：")
    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        print(
            f"{parameter_name}: "
            f"{parameter_value}"
        )

    print(
        "\n最佳交叉验证 AUC："
        f"{grid_search.best_score_:.6f}"
    )

    print(
        "\n独立测试集评估结果"
        "（阳性类别：TRUST >= 16）："
    )

    for metric_name, metric_value in (
        evaluation_metrics.items()
    ):
        print(
            f"{metric_name:<10}: "
            f"{metric_value:.6f}"
        )


if __name__ == "__main__":
    main()