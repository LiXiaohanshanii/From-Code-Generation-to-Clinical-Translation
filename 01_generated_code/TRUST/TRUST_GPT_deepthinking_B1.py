from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

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
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline


# =========================
# 1. 参数设置
# =========================

DATA_PATH = Path("train_data.csv")
MODEL_OUTPUT_PATH = Path("random_forest_trust_model.joblib")

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

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_and_validate_data(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取并检查数据。

    目标变量转换规则：
        TRUST >= 16 -> 1
        TRUST < 16  -> 0
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}\n"
            "请确认 train_data.csv 位于当前项目工作目录中。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    # 去除列名两侧可能存在的空格
    data.columns = data.columns.astype(str).str.strip()

    # 兼容需求描述中可能出现的 DIAGONSIS 拼写
    if "DIAGNOSIS" not in data.columns and "DIAGONSIS" in data.columns:
        warnings.warn(
            "检测到列名 DIAGONSIS，已自动重命名为 DIAGNOSIS。",
            UserWarning,
        )
        data = data.rename(columns={"DIAGONSIS": "DIAGNOSIS"})

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要字段：\n"
            f"{missing_columns}\n\n"
            f"当前数据字段为：\n{data.columns.tolist()}"
        )

    if data.empty:
        raise ValueError("数据集为空，无法构建模型。")

    if data.columns[-1] != TARGET_COLUMN:
        warnings.warn(
            f"目标列 {TARGET_COLUMN} 不是数据集最后一列，"
            "代码仍会根据列名提取目标变量。",
            UserWarning,
        )

    # 只提取建模需要的特征
    X = data[FEATURE_COLUMNS].copy()

    # 分类变量按照要求不应存在缺失值
    categorical_columns = ONE_HOT_COLUMNS + ORDINAL_COLUMNS
    categorical_missing = X[categorical_columns].isna().sum()
    categorical_missing = categorical_missing[
        categorical_missing > 0
    ]

    if not categorical_missing.empty:
        raise ValueError(
            "以下分类变量存在缺失值，但任务要求分类变量无缺失值：\n"
            f"{categorical_missing.to_dict()}"
        )

    # 将连续变量转换为数值型
    # 非法字符会转换为 NaN，随后由中位数填充
    for column in CONTINUOUS_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # 将 TRUST 转换为数值
    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    invalid_target_count = int(trust_numeric.isna().sum())
    if invalid_target_count > 0:
        raise ValueError(
            f"目标列 {TARGET_COLUMN} 中有 "
            f"{invalid_target_count} 个缺失值或非数值内容。"
        )

    # 二分类目标：TRUST >= 16 为阳性类别 1
    y = (trust_numeric >= 16).astype(int)
    y.name = "TRUST_GE_16"

    class_counts = y.value_counts().sort_index()

    if y.nunique() != 2:
        raise ValueError(
            "转换后的目标变量不是二分类数据。\n"
            f"类别分布：{class_counts.to_dict()}\n"
            "请确认 TRUST 中同时存在小于16和大于等于16的数据。"
        )

    print("=" * 60)
    print("数据读取完成")
    print(f"样本数：{len(X)}")
    print(f"特征数：{X.shape[1]}")
    print("转换后的总体类别分布：")
    print(f"  0（TRUST < 16）：{class_counts.get(0, 0)}")
    print(f"  1（TRUST >= 16）：{class_counts.get(1, 0)}")
    print("=" * 60)

    return X, y


def create_cross_validation_and_smote_k(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[StratifiedKFold, int]:
    """
    根据训练集中少数类别的样本数，自动确定：
    1. 分层交叉验证折数
    2. SMOTE 的 k_neighbors

    这样可以降低小样本情况下 SMOTE 报错的概率。
    """
    class_counts = y_train.value_counts()
    minority_class = class_counts.idxmin()
    minority_count = int(class_counts.min())

    if minority_count < 3:
        raise ValueError(
            "训练集中的少数类别样本不足3例，"
            "无法在交叉验证中稳定执行SMOTE。\n"
            f"训练集类别分布：{class_counts.to_dict()}"
        )

    n_splits = min(5, minority_count)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 计算各交叉验证训练折中少数类别的最小样本数
    minimum_fold_minority_count = np.inf

    for fold_train_indices, _ in cv.split(X_train, y_train):
        fold_y_train = y_train.iloc[fold_train_indices]
        fold_minority_count = int(
            (fold_y_train == minority_class).sum()
        )

        minimum_fold_minority_count = min(
            minimum_fold_minority_count,
            fold_minority_count,
        )

    minimum_fold_minority_count = int(
        minimum_fold_minority_count
    )

    if minimum_fold_minority_count < 2:
        raise ValueError(
            "部分交叉验证训练折中的少数类别样本少于2例，"
            "无法执行SMOTE。"
        )

    # SMOTE要求 k_neighbors 小于少数类别样本数
    smote_k_neighbors = min(
        5,
        minimum_fold_minority_count - 1,
    )

    print(f"交叉验证折数：{n_splits}")
    print(f"SMOTE k_neighbors：{smote_k_neighbors}")

    return cv, smote_k_neighbors


def build_model_pipeline(
    smote_k_neighbors: int,
) -> ImbalancedPipeline:
    """
    构建预处理、SMOTE和随机森林管道。
    """

    # 连续变量：中位数填充
    continuous_transformer = SklearnPipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    # SEX、DEPT、DIAGNOSIS：独热编码
    one_hot_transformer = SklearnPipeline(
        steps=[
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # TPPA：序数编码
    #
    # categories="auto" 表示根据训练数据自动识别类别。
    # handle_unknown 设置用于处理测试集中未在训练集中出现的新类别。
    ordinal_transformer = SklearnPipeline(
        steps=[
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    categories="auto",
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
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

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=smote_k_neighbors,
    )

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    # imblearn.pipeline.Pipeline 能正确处理 SMOTE
    model_pipeline = ImbalancedPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("random_forest", random_forest),
        ]
    )

    return model_pipeline


def evaluate_model(
    model: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    在独立测试集上评估模型。
    """
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

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
        "AUC": roc_auc_score(
            y_test,
            y_probability,
        ),
    }

    print("\n" + "=" * 60)
    print("独立测试集评估结果")
    print("=" * 60)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:<12}: {metric_value:.4f}")

    print("=" * 60)

    return metrics


def main() -> None:
    # 读取数据
    X, y = load_and_validate_data(DATA_PATH)

    # 先划分训练集和测试集，避免测试集参与SMOTE或参数调优
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

    # 根据训练数据确定交叉验证折数和SMOTE参数
    cv, smote_k_neighbors = (
        create_cross_validation_and_smote_k(
            X_train,
            y_train,
        )
    )

    # 构建完整管道
    model_pipeline = build_model_pipeline(
        smote_k_neighbors=smote_k_neighbors,
    )

    # 严格按照指定范围进行超参数调优
    parameter_grid = {
        "random_forest__n_estimators": [100, 200],
        "random_forest__max_depth": [10],
        "random_forest__min_samples_split": [2],
        "random_forest__min_samples_leaf": [1],
        "random_forest__class_weight": ["balanced"],
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,          # 不使用多进程
        refit=True,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    print("\n开始进行网格搜索和模型训练……")
    grid_search.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("参数调优结果")
    print("=" * 60)
    print(f"最佳交叉验证AUC：{grid_search.best_score_:.4f}")
    print("最佳参数：")

    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        print(f"  {parameter_name}: {parameter_value}")

    # 在独立测试集上评估
    evaluate_model(
        model=grid_search,
        X_test=X_test,
        y_test=y_test,
    )

    # 保存最佳完整模型
    # 文件中包含预处理器、SMOTE配置和随机森林模型
    joblib.dump(
        grid_search.best_estimator_,
        MODEL_OUTPUT_PATH,
    )

    print(
        "\n最佳模型已保存至："
        f"{MODEL_OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n模型运行失败：")
        print(f"{type(error).__name__}: {error}")
        raise