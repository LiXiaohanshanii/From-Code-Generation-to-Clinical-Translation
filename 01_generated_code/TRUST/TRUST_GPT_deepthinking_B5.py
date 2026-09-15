from pathlib import Path
import math

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
    make_scorer,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline


# ============================================================
# 1. 基本配置
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# train_data.csv 与本脚本放在同一目录
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


# ============================================================
# 2. 读取并检查数据
# ============================================================

def load_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取数据，并将 TRUST 转换为二分类目标：
        TRUST >= 16：1
        TRUST < 16：0
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path}\n"
            "请确认 train_data.csv 与本 Python 脚本位于同一目录。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要列："
            + ", ".join(missing_columns)
        )

    # 只保留本模型所需的列
    data = data[required_columns].copy()

    # 将目标列转换为数值
    target_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce"
    )

    # 删除目标列为空或不能转换为数字的记录
    valid_target_mask = target_numeric.notna()
    invalid_target_count = int((~valid_target_mask).sum())

    if invalid_target_count > 0:
        print(
            f"警告：发现 {invalid_target_count} 条 TRUST 无法转换为数值，"
            "这些记录将被删除。"
        )

    data = data.loc[valid_target_mask].copy()
    target_numeric = target_numeric.loc[valid_target_mask]

    # 连续变量强制转换为数值
    # 非法字符会转为 NaN，随后由中位数填充
    for column in CONTINUOUS_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # 将正负无穷替换为缺失值
    data[CONTINUOUS_COLUMNS] = data[
        CONTINUOUS_COLUMNS
    ].replace([np.inf, -np.inf], np.nan)

    X = data[FEATURE_COLUMNS].reset_index(drop=True)

    # TRUST >= 16 记为阳性类别1，否则为类别0
    y = (
        target_numeric.reset_index(drop=True) >= 16
    ).astype(int)

    class_counts = y.value_counts().sort_index()

    if len(class_counts) < 2:
        raise ValueError(
            "目标变量转换后只有一个类别，无法建立二分类模型。"
        )

    if class_counts.min() < 2:
        raise ValueError(
            "至少有一个类别的样本数少于2，"
            "无法进行分层训练集/测试集划分。"
        )

    return X, y


# ============================================================
# 3. 构建数据预处理器
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    预处理方法：
    1. SEX、DEPT、DIAGNOSIS：独热编码
    2. TPPA：序数编码
    3. 连续变量：中位数填充
    """

    # 连续变量中位数填充
    continuous_transformer = SklearnPipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    # 独热编码
    # sparse_output=False 输出稠密数组，方便后续SMOTE处理
    one_hot_transformer = SklearnPipeline(
        steps=[
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                )
            )
        ]
    )

    # TPPA序数编码
    # 测试集中出现训练集未见类别时编码为-1
    ordinal_transformer = SklearnPipeline(
        steps=[
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous",
                continuous_transformer,
                CONTINUOUS_COLUMNS,
            ),
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
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor


# ============================================================
# 4. 根据少数类别样本数设置交叉验证和SMOTE
# ============================================================

def determine_cv_and_smote_neighbors(
    y_train: pd.Series,
) -> tuple[int, int]:
    """
    根据训练集中少数类别的样本量，自动确定：
    1. 分层交叉验证折数
    2. SMOTE的k_neighbors

    这样可以降低小样本数据中SMOTE报错的概率。
    """
    class_counts = y_train.value_counts()
    minority_count = int(class_counts.min())

    if minority_count < 3:
        raise ValueError(
            "训练集中少数类别样本少于3例，"
            "无法安全地同时执行分层交叉验证和SMOTE。"
        )

    # 默认最多使用5折交叉验证
    n_splits = min(5, minority_count)

    # 计算某一交叉验证训练折中可能出现的最少少数类样本数
    max_validation_minority = math.ceil(
        minority_count / n_splits
    )
    minimum_cv_train_minority = (
        minority_count - max_validation_minority
    )

    if minimum_cv_train_minority < 2:
        raise ValueError(
            "交叉验证训练折中的少数类别样本不足，"
            "无法应用SMOTE。"
        )

    # SMOTE要求k_neighbors小于少数类别样本数
    smote_neighbors = min(
        5,
        minimum_cv_train_minority - 1
    )

    return n_splits, smote_neighbors


# ============================================================
# 5. 构建并调优模型
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    preprocessor = build_preprocessor()

    n_splits, smote_neighbors = (
        determine_cv_and_smote_neighbors(y_train)
    )

    print(f"\n交叉验证折数：{n_splits}")
    print(f"SMOTE k_neighbors：{smote_neighbors}")

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=smote_neighbors,
    )

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,

        # 明确禁用多进程
        n_jobs=1,
    )

    # imblearn的Pipeline可以在交叉验证训练折内部执行SMOTE
    model_pipeline = ImbalancedPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("classifier", random_forest),
        ]
    )

    # 按照指定参数进行超参数调优
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
            zero_division=0
        ),
        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),
        "roc_auc": "roc_auc",
    }

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring=scoring,

        # 根据ROC-AUC选择最佳模型
        refit="roc_auc",

        cv=cross_validation,

        # 不使用多进程
        n_jobs=1,

        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    grid_search.fit(X_train, y_train)

    return grid_search


# ============================================================
# 6. 模型评估
# ============================================================

def evaluate_model(
    grid_search: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    if y_test.nunique() == 2:
        auc = roc_auc_score(
            y_test,
            y_probability
        )
    else:
        auc = np.nan

    print("\n" + "=" * 60)
    print("最佳超参数")
    print("=" * 60)

    for parameter, value in grid_search.best_params_.items():
        clean_name = parameter.replace("classifier__", "")
        print(f"{clean_name}: {value}")

    print("\n" + "=" * 60)
    print("最佳模型的交叉验证结果")
    print("=" * 60)

    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_

    cv_metric_names = {
        "accuracy": "Accuracy",
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
    }

    for metric_key, metric_name in cv_metric_names.items():
        mean_score = cv_results[
            f"mean_test_{metric_key}"
        ][best_index]

        std_score = cv_results[
            f"std_test_{metric_key}"
        ][best_index]

        print(
            f"{metric_name:<12}: "
            f"{mean_score:.4f} ± {std_score:.4f}"
        )

    print("\n" + "=" * 60)
    print("独立测试集评估结果")
    print("=" * 60)
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"F1-score   : {f1:.4f}")

    if np.isnan(auc):
        print(
            "ROC-AUC    : 无法计算，"
            "因为测试集中只包含一个类别。"
        )
    else:
        print(f"ROC-AUC    : {auc:.4f}")


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    print(f"正在读取数据：{DATA_FILE}")

    X, y = load_data(DATA_FILE)

    print(f"总样本数：{len(X)}")
    print("\n目标类别定义：")
    print("0 = TRUST < 16")
    print("1 = TRUST >= 16")

    print("\n完整数据集类别分布：")
    class_distribution = y.value_counts().sort_index()

    for label, count in class_distribution.items():
        proportion = count / len(y)
        print(
            f"类别 {label}：{count} 例，"
            f"占比 {proportion:.2%}"
        )

    # 必须先划分测试集，再在训练过程内部执行SMOTE
    # 测试集不能进行SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\n训练集样本数：{len(X_train)}")
    print(f"测试集样本数：{len(X_test)}")

    print("\n训练集类别分布（SMOTE前）：")
    print(y_train.value_counts().sort_index().to_string())

    grid_search = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    evaluate_model(
        grid_search=grid_search,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()