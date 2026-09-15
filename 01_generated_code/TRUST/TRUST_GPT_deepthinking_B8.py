"""
随机森林二分类模型
目标：预测 TRUST 是否 >= 16

正类（1）：TRUST >= 16
负类（0）：TRUST < 16
"""

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
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. 全局参数
# ============================================================

DATA_PATH = Path("train_data.csv")
TARGET_COLUMN = "TRUST"

RANDOM_STATE = 42
TEST_SIZE = 0.20

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

FEATURE_COLUMNS = (
    ONE_HOT_COLUMNS
    + ORDINAL_COLUMNS
    + CONTINUOUS_COLUMNS
)


# ============================================================
# 2. 读取和检查数据
# ============================================================

def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取CSV文件，检查字段并构建二分类目标变量。

    返回：
        X：模型特征
        y：二分类目标，1表示TRUST >= 16，0表示TRUST < 16
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{csv_path.resolve()}\n"
            "请将 train_data.csv 放在当前PyCharm项目的工作目录中。"
        )

    df = pd.read_csv(
        csv_path,
        encoding="utf-8",
    )

    if df.empty:
        raise ValueError("train_data.csv 中没有数据。")

    # 清除列名前后的空格
    df.columns = df.columns.astype(str).str.strip()

    # 兼容任务描述中可能出现的 DIAGONSIS 拼写
    if (
        "DIAGNOSIS" not in df.columns
        and "DIAGONSIS" in df.columns
    ):
        df = df.rename(
            columns={"DIAGONSIS": "DIAGNOSIS"}
        )
        print(
            "提示：检测到列名 DIAGONSIS，"
            "已自动重命名为 DIAGNOSIS。"
        )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要字段："
            f"{missing_columns}"
        )

    if df.columns[-1] != TARGET_COLUMN:
        print(
            f"警告：数据集最后一列是 {df.columns[-1]!r}，"
            f"不是 {TARGET_COLUMN!r}。"
            "程序仍将按列名 TRUST 作为目标变量。"
        )

    # 按要求，分类变量不应存在缺失值
    categorical_columns = (
        ONE_HOT_COLUMNS + ORDINAL_COLUMNS
    )

    categorical_missing = (
        df[categorical_columns]
        .isna()
        .sum()
    )

    categorical_missing = categorical_missing[
        categorical_missing > 0
    ]

    if not categorical_missing.empty:
        raise ValueError(
            "检测到分类变量存在缺失值，但任务设定为"
            "分类变量无缺失值：\n"
            f"{categorical_missing.to_string()}"
        )

    X = df[FEATURE_COLUMNS].copy()

    # 分类变量统一转为字符串，避免同一列中数字和文本混合
    for column in categorical_columns:
        X[column] = X[column].astype(str).str.strip()

    # 连续变量转为数值
    # 无法转换的内容将变为NaN，之后由中位数进行填充
    for column in CONTINUOUS_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # TRUST必须能够转换为数值
    trust_numeric = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    invalid_target_count = int(
        trust_numeric.isna().sum()
    )

    if invalid_target_count > 0:
        raise ValueError(
            f"目标列 {TARGET_COLUMN} 中有 "
            f"{invalid_target_count} 个值无法转换为数值。"
        )

    # 构建二分类目标
    y = (trust_numeric >= 16).astype(int)
    y.name = "TRUST_GE_16"

    if y.nunique() != 2:
        raise ValueError(
            "转换后的目标变量不包含两个类别。\n"
            "请确认 TRUST 中同时存在小于16和大于等于16的样本。"
        )

    return X, y


# ============================================================
# 3. 根据训练集样本量确定交叉验证和SMOTE参数
# ============================================================

def configure_cv_and_smote(
    y_train: pd.Series,
) -> tuple[StratifiedKFold, int]:
    """
    根据训练集中少数类样本数确定：
    1. 分层交叉验证折数；
    2. SMOTE的k_neighbors。

    这样可以降低少数类样本较少时SMOTE报错的风险。
    """

    class_counts = y_train.value_counts()
    minority_count = int(class_counts.min())

    if minority_count < 3:
        raise ValueError(
            "训练集中的少数类样本不足3例，"
            "无法在交叉验证内部可靠执行SMOTE。\n"
            f"训练集类别数量：{class_counts.to_dict()}"
        )

    cv_splits = min(5, minority_count)

    # 估计交叉验证中某个训练折内最少的少数类样本数
    minimum_fold_minority = (
        minority_count
        - math.ceil(minority_count / cv_splits)
    )

    if minimum_fold_minority < 2:
        raise ValueError(
            "交叉验证训练折中的少数类样本过少，"
            "无法执行SMOTE。"
        )

    # SMOTE要求k_neighbors小于少数类样本数
    smote_k_neighbors = min(
        5,
        minimum_fold_minority - 1,
    )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return cv, smote_k_neighbors


# ============================================================
# 4. 构建预处理器
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    构建数据预处理器：
    1. SEX、DEPT、DIAGNOSIS：独热编码；
    2. TPPA：序数编码；
    3. 连续变量：中位数填充。
    """

    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )

    continuous_transformer = SimpleImputer(
        strategy="median",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot",
                one_hot_encoder,
                ONE_HOT_COLUMNS,
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
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# 5. 构建并调优模型
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    使用SMOTE、随机森林和网格搜索训练模型。
    """

    cv, smote_k_neighbors = configure_cv_and_smote(
        y_train
    )

    print(f"交叉验证折数：{cv.n_splits}")
    print(
        "SMOTE参数 k_neighbors："
        f"{smote_k_neighbors}"
    )

    preprocessor = build_preprocessor()

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=smote_k_neighbors,
    )

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,

        # 明确禁止随机森林内部多进程
        n_jobs=1,
    )

    # imblearn的Pipeline可在预处理后、模型训练前执行SMOTE
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
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

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,

        # 使用AUC选择最佳参数
        scoring="roc_auc",

        cv=cv,
        refit=True,

        # 禁止网格搜索多进程
        n_jobs=1,
        pre_dispatch=1,

        return_train_score=False,
        error_score="raise",
        verbose=1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    return grid_search


# ============================================================
# 6. 独立测试集评估
# ============================================================

def evaluate_model(
    model: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    在未参与训练和参数调优的测试集上评估模型。
    """

    y_pred = model.predict(X_test)

    # 正类TRUST >= 16的预测概率
    y_probability = model.predict_proba(X_test)[:, 1]

    results = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
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

    result_df = pd.DataFrame(
        {
            "Metric": list(results.keys()),
            "Value": list(results.values()),
        }
    )

    return result_df


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    X, y = load_data(DATA_PATH)

    print("=" * 60)
    print("数据读取完成")
    print(f"总样本数：{len(X)}")
    print(f"特征数：{X.shape[1]}")
    print("目标变量定义：1 = TRUST >= 16，0 = TRUST < 16")
    print("\n总体类别分布：")
    print(y.value_counts().sort_index().to_string())

    # 使用分层抽样划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n训练集类别分布：")
    print(y_train.value_counts().sort_index().to_string())

    print("\n测试集类别分布：")
    print(y_test.value_counts().sort_index().to_string())

    print("\n开始执行网格搜索...")
    grid_search = train_model(
        X_train,
        y_train,
    )

    print("\n" + "=" * 60)
    print("参数调优完成")
    print(
        "交叉验证最佳AUC："
        f"{grid_search.best_score_:.6f}"
    )

    print("\n最佳参数：")
    for parameter, value in (
        grid_search.best_params_.items()
    ):
        clean_parameter = parameter.replace(
            "classifier__",
            "",
        )
        print(f"{clean_parameter}: {value}")

    result_df = evaluate_model(
        grid_search,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("独立测试集评估结果")
    print(
        result_df.to_string(
            index=False,
            formatters={
                "Value": lambda value: f"{value:.6f}"
            },
        )
    )


if __name__ == "__main__":
    main()