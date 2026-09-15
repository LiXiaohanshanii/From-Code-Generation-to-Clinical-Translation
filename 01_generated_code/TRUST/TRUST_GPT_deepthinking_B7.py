from pathlib import Path
from math import ceil

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
    make_scorer,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. 参数设置
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# 默认读取与本脚本位于同一目录下的 train_data.csv
DATA_PATH = Path(__file__).resolve().parent / "train_data.csv"

TARGET_COLUMN = "TRUST"

ONE_HOT_FEATURES = [
    "SEX",
    "DEPT",
    "DIAGNOSIS",
]

ORDINAL_FEATURES = [
    "TPPA",
]

CONTINUOUS_FEATURES = [
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

ALL_FEATURES = (
    ONE_HOT_FEATURES
    + ORDINAL_FEATURES
    + CONTINUOUS_FEATURES
)

# 如果TPPA有明确的临床顺序，可在此处填写。
# 示例：
# TPPA_ORDER = ["阴性", "1:80", "1:160", "1:320", "1:640"]
#
# 未指定时：
# 1. 若TPPA全部可以转换为数值，则按数值大小编码；
# 2. 否则由OrdinalEncoder自动确定类别顺序。
TPPA_ORDER = None


# ============================================================
# 2. 读取和检查数据
# ============================================================

def load_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """读取数据、检查字段，并将TRUST转换为二分类目标。"""

    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path}\n"
            "请确认train_data.csv与当前Python脚本位于同一目录。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    # 去除列名首尾可能存在的空格
    data.columns = data.columns.astype(str).str.strip()

    if data.empty:
        raise ValueError("train_data.csv为空，无法构建模型。")

    # 根据题目要求，TRUST应为最后一列
    if data.columns[-1] != TARGET_COLUMN:
        raise ValueError(
            f"数据集最后一列应为{TARGET_COLUMN}，"
            f"实际最后一列为：{data.columns[-1]}"
        )

    missing_columns = [
        column for column in ALL_FEATURES + [TARGET_COLUMN]
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下字段："
            + "、".join(missing_columns)
            + "\n请特别检查DIAGNOSIS的拼写。"
        )

    # TRUST转为数值
    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    invalid_target_count = int(trust_numeric.isna().sum())

    if invalid_target_count > 0:
        print(
            f"警告：TRUST列中有{invalid_target_count}行无法转换为数值，"
            "这些行将被删除。"
        )

    valid_target_mask = trust_numeric.notna()

    data = data.loc[valid_target_mask].copy()
    trust_numeric = trust_numeric.loc[valid_target_mask]

    if data.empty:
        raise ValueError("删除TRUST无效值后没有可用数据。")

    # 只保留指定特征
    X = data[ALL_FEATURES].copy()

    # TRUST >= 16记为1，否则记为0
    y = (trust_numeric >= 16).astype(int)
    y.name = "TRUST_GE_16"

    # 连续变量转为数值。
    # 无法转换的内容会变为NaN，后续使用中位数填充。
    for column in CONTINUOUS_FEATURES:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # 独热编码变量统一转换为字符串，避免同一列混有数字和文本
    for column in ONE_HOT_FEATURES:
        X[column] = X[column].astype(str)

    # TPPA的数据类型处理
    if TPPA_ORDER is not None:
        X["TPPA"] = X["TPPA"].astype(str)
    else:
        tppa_numeric = pd.to_numeric(
            X["TPPA"],
            errors="coerce",
        )

        if tppa_numeric.notna().all():
            X["TPPA"] = tppa_numeric
        else:
            X["TPPA"] = X["TPPA"].astype(str)

    class_counts = y.value_counts().sort_index()

    if len(class_counts) != 2:
        raise ValueError(
            "目标变量转换后没有同时包含两个类别。\n"
            "请确认TRUST列中同时存在小于16和大于等于16的数据。"
        )

    if class_counts.min() < 2:
        raise ValueError(
            "少数类别样本数少于2，无法进行分层训练集/测试集划分。"
        )

    print("=" * 60)
    print(f"数据文件：{file_path}")
    print(f"有效样本数：{len(X)}")
    print(f"特征数量：{X.shape[1]}")
    print("\n目标类别分布：")
    print(f"TRUST < 16 ：{class_counts.get(0, 0)}")
    print(f"TRUST >= 16：{class_counts.get(1, 0)}")
    print("=" * 60)

    return X, y


# ============================================================
# 3. 创建预处理器
# ============================================================

def create_preprocessor() -> ColumnTransformer:
    """创建连续变量填充和分类变量编码预处理器。"""

    # 兼容不同版本的scikit-learn
    try:
        one_hot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        # scikit-learn旧版本使用sparse参数
        one_hot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    if TPPA_ORDER is None:
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    else:
        ordinal_encoder = OrdinalEncoder(
            categories=[
                [str(value) for value in TPPA_ORDER]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                SimpleImputer(strategy="median"),
                CONTINUOUS_FEATURES,
            ),
            (
                "one_hot",
                one_hot_encoder,
                ONE_HOT_FEATURES,
            ),
            (
                "ordinal",
                ordinal_encoder,
                ORDINAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# 4. 根据样本数量设置交叉验证和SMOTE
# ============================================================

def determine_cv_and_smote_parameters(
    y_train: pd.Series,
) -> tuple[int, int]:
    """
    根据训练集中少数类样本数，确定：
    1. 分层交叉验证折数；
    2. SMOTE的k_neighbors。

    这样可以降低小样本数据中SMOTE报错的风险。
    """

    class_counts = y_train.value_counts()

    if len(class_counts) != 2:
        raise ValueError("训练集没有同时包含两个目标类别。")

    minority_count = int(class_counts.min())

    if minority_count < 3:
        raise ValueError(
            "训练集中少数类样本少于3个，"
            "无法稳定地同时执行交叉验证和SMOTE。"
        )

    cv_splits = min(5, minority_count)

    # 估计每次交叉验证的训练折中可能出现的最少少数类样本数
    minimum_cv_train_minority = (
        minority_count
        - ceil(minority_count / cv_splits)
    )

    smote_k_neighbors = min(
        5,
        minimum_cv_train_minority - 1,
    )

    if smote_k_neighbors < 1:
        raise ValueError(
            "少数类样本数量不足，无法为SMOTE设置有效的k_neighbors。"
        )

    return cv_splits, smote_k_neighbors


# ============================================================
# 5. 模型训练及超参数调优
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """使用SMOTE、随机森林和网格搜索训练模型。"""

    cv_splits, smote_k_neighbors = (
        determine_cv_and_smote_parameters(y_train)
    )

    print(f"\n交叉验证折数：{cv_splits}")
    print(f"SMOTE k_neighbors：{smote_k_neighbors}")

    preprocessor = create_preprocessor()

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    # SMOTE位于预处理之后、随机森林之前。
    # 在GridSearchCV中，SMOTE只对每个训练折执行，
    # 不会对验证折或最终测试集执行。
    model_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=smote_k_neighbors,
                ),
            ),
            ("classifier", random_forest),
        ]
    )

    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

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
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring=scoring,
        refit="roc_auc",
        cv=cross_validation,
        n_jobs=1,              # 按要求不使用多进程
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    print("\n开始进行网格搜索和模型训练……")
    grid_search.fit(X_train, y_train)

    return grid_search


# ============================================================
# 6. 模型评估
# ============================================================

def evaluate_model(
    grid_search: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """在独立测试集上计算指定评价指标。"""

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

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

    result_table = pd.DataFrame(
        {
            "评价指标": list(metrics.keys()),
            "测试集结果": list(metrics.values()),
        }
    )

    return result_table


def print_cv_results(grid_search: GridSearchCV) -> None:
    """输出最优模型的交叉验证结果。"""

    best_index = grid_search.best_index_

    metric_names = {
        "accuracy": "Accuracy",
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1-score",
        "roc_auc": "AUC",
    }

    print("\n" + "=" * 60)
    print("最优超参数")
    print("=" * 60)

    for parameter, value in grid_search.best_params_.items():
        clean_name = parameter.replace("classifier__", "")
        print(f"{clean_name}: {value}")

    print("\n" + "=" * 60)
    print("最优模型的平均交叉验证结果")
    print("=" * 60)

    for internal_name, display_name in metric_names.items():
        mean_key = f"mean_test_{internal_name}"
        std_key = f"std_test_{internal_name}"

        mean_value = grid_search.cv_results_[mean_key][best_index]
        std_value = grid_search.cv_results_[std_key][best_index]

        print(
            f"{display_name:<10}: "
            f"{mean_value:.4f} ± {std_value:.4f}"
        )


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n训练集类别分布：")
    print(y_train.value_counts().sort_index())

    print("\n测试集类别分布：")
    print(y_test.value_counts().sort_index())

    grid_search = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    print_cv_results(grid_search)

    test_results = evaluate_model(
        grid_search=grid_search,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n" + "=" * 60)
    print("独立测试集评估结果")
    print("=" * 60)

    print(
        test_results.to_string(
            index=False,
            formatters={
                "测试集结果": lambda value: f"{value:.4f}"
            },
        )
    )


if __name__ == "__main__":
    main()