from pathlib import Path
import math

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 全局参数
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# CSV文件与本Python脚本放在同一目录
DATA_FILE = Path(__file__).resolve().parent / "train_data.csv"

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


def load_and_check_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取数据、检查列名，并生成二分类目标变量。

    原始TRUST：
        TRUST >= 16 -> 1
        TRUST < 16  -> 0
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path}\n"
            "请确认train_data.csv与当前Python脚本位于同一目录。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    # 去除列名前后的空格，避免“ AGE”等列名导致匹配失败
    data.columns = data.columns.astype(str).str.strip()

    if data.empty:
        raise ValueError("train_data.csv为空，无法建立模型。")

    # 根据需求，TRUST应为最后一列
    actual_last_column = data.columns[-1]
    if actual_last_column != TARGET_COLUMN:
        raise ValueError(
            f"数据集最后一列应为'{TARGET_COLUMN}'，"
            f"但当前最后一列是'{actual_last_column}'。"
        )

    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要列："
            + ", ".join(missing_columns)
        )

    # 只提取指定特征，避免其他无关列进入模型
    X = data[FEATURE_COLUMNS].copy()

    # 将连续变量转换为数值
    # 无法转换的内容将变成NaN，随后由中位数进行填补
    for column in CONTINUOUS_COLUMNS:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # 按照需求，分类变量应当不存在缺失值
    categorical_columns = ONE_HOT_COLUMNS + ORDINAL_COLUMNS
    categorical_missing = X[categorical_columns].isna().sum()
    categorical_missing = categorical_missing[
        categorical_missing > 0
    ]

    if not categorical_missing.empty:
        missing_details = ", ".join(
            f"{column}: {count}个"
            for column, count in categorical_missing.items()
        )
        raise ValueError(
            "检测到分类变量存在缺失值，但需求中规定分类变量无缺失值。"
            f"具体情况：{missing_details}"
        )

    # TRUST转换为数值
    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    if trust_numeric.isna().any():
        invalid_rows = trust_numeric[
            trust_numeric.isna()
        ].index.tolist()

        # CSV中的行号通常比DataFrame索引大2：
        # 一行为表头，索引从0开始
        csv_row_numbers = [index + 2 for index in invalid_rows[:10]]

        raise ValueError(
            "TRUST列中存在空值或无法转换为数值的内容。"
            f"部分异常CSV行号：{csv_row_numbers}"
        )

    # 构建二分类目标
    y = (trust_numeric >= 16).astype(int)
    y.name = "TRUST_GE_16"

    if y.nunique() != 2:
        class_counts = y.value_counts().to_dict()
        raise ValueError(
            "目标变量转换后没有同时包含两个类别，无法建立二分类模型。"
            f"类别分布：{class_counts}"
        )

    return X, y


def calculate_cv_and_smote_parameters(
    y_train: pd.Series,
) -> tuple[int, int]:
    """
    根据训练集中少数类样本量，自动确定：

    1. 分层交叉验证折数
    2. SMOTE的k_neighbors

    这样可以降低少数类样本较少时SMOTE报错的风险。
    """
    class_counts = y_train.value_counts()
    minority_count = int(class_counts.min())

    # 默认最多进行5折交叉验证
    cv_splits = min(5, minority_count)

    if cv_splits < 2:
        raise ValueError(
            "训练集中少数类样本不足2例，"
            "无法进行分层交叉验证和SMOTE。"
        )

    # 估算每个交叉验证训练折中最少可能包含的少数类样本数
    minimum_minority_in_cv_training = (
        minority_count
        - math.ceil(minority_count / cv_splits)
    )

    if minimum_minority_in_cv_training < 2:
        raise ValueError(
            "少数类样本量过少，无法在交叉验证训练折内执行SMOTE。"
            f"当前训练集少数类样本数：{minority_count}。"
        )

    # SMOTE要求k_neighbors小于少数类样本数
    smote_k_neighbors = min(
        5,
        minimum_minority_in_cv_training - 1,
    )

    return cv_splits, smote_k_neighbors


def build_preprocessor() -> ColumnTransformer:
    """
    构建数据预处理器。

    SEX、DEPT、DIAGNOSIS：
        独热编码

    TPPA：
        序数编码

    连续变量：
        中位数填补
    """
    one_hot_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    ordinal_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

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

    return preprocessor


def evaluate_model(
    model: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    在独立测试集上计算模型评估指标。
    """
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    results = {
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
        "AUC": roc_auc_score(
            y_test,
            y_probability,
        ),
    }

    return results


def main() -> None:
    # --------------------------------------------------------
    # 1. 读取并检查数据
    # --------------------------------------------------------
    X, y = load_and_check_data(DATA_FILE)

    print("=" * 60)
    print("数据读取完成")
    print(f"样本数：{len(X)}")
    print(f"特征数：{X.shape[1]}")

    print("\n二分类目标定义：")
    print("0：TRUST < 16")
    print("1：TRUST >= 16")

    print("\n完整数据集类别分布：")
    print(y.value_counts().sort_index().rename("样本数"))

    # --------------------------------------------------------
    # 2. 划分训练集和测试集
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n训练集样本数：", len(X_train))
    print("测试集样本数：", len(X_test))

    print("\n训练集类别分布（SMOTE前）：")
    print(y_train.value_counts().sort_index().rename("样本数"))

    print("\n测试集类别分布：")
    print(y_test.value_counts().sort_index().rename("样本数"))

    # --------------------------------------------------------
    # 3. 根据少数类样本数设置交叉验证与SMOTE参数
    # --------------------------------------------------------
    cv_splits, smote_k_neighbors = (
        calculate_cv_and_smote_parameters(y_train)
    )

    print(f"\n分层交叉验证折数：{cv_splits}")
    print(f"SMOTE k_neighbors：{smote_k_neighbors}")

    # --------------------------------------------------------
    # 4. 构建预处理器
    # --------------------------------------------------------
    preprocessor = build_preprocessor()

    # --------------------------------------------------------
    # 5. 构建SMOTE和随机森林流水线
    #
    # 使用imblearn的Pipeline可以保证：
    # 每个交叉验证训练折单独进行预处理和SMOTE，
    # 验证折、测试集不参与SMOTE，避免数据泄漏。
    # --------------------------------------------------------
    pipeline = ImbPipeline(
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
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # 6. 设置超参数搜索范围
    # --------------------------------------------------------
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    cross_validation = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 使用AUC作为超参数选择指标
    # n_jobs=1表示不使用多进程
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cross_validation,
        refit=True,
        n_jobs=1,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    # --------------------------------------------------------
    # 7. 训练模型并进行超参数调优
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("开始进行随机森林超参数搜索……")

    grid_search.fit(X_train, y_train)

    print("\n超参数搜索完成。")

    best_parameters = {
        parameter.replace("classifier__", ""): value
        for parameter, value in grid_search.best_params_.items()
    }

    print("\n最佳超参数：")
    for parameter, value in best_parameters.items():
        print(f"{parameter}: {value}")

    print(
        "\n最佳交叉验证AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # --------------------------------------------------------
    # 8. 使用独立测试集评估最佳模型
    # --------------------------------------------------------
    best_model = grid_search.best_estimator_

    test_results = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("独立测试集评估结果")
    print("=" * 60)

    for metric_name, metric_value in test_results.items():
        print(f"{metric_name:<10}: {metric_value:.4f}")


if __name__ == "__main__":
    main()