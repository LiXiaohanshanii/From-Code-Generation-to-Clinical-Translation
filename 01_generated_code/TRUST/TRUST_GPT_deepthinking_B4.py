from __future__ import annotations

import math
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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline


# ============================================================
# 1. 全局配置
# ============================================================

DATA_FILE = Path("train_data.csv")
MODEL_FILE = Path("random_forest_smote_model.joblib")

TARGET_COLUMN = "TRUST"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_CV_FOLDS = 5

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


# ============================================================
# 2. 工具函数
# ============================================================

def create_one_hot_encoder() -> OneHotEncoder:
    """
    创建独热编码器。

    sparse_output=False 适用于较新的 scikit-learn。
    兼容部分仍使用 sparse=False 参数的旧版本。
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float64,
        )


def load_and_validate_data(file_path: Path) -> pd.DataFrame:
    """
    读取CSV文件并检查必要列。
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}\n"
            "请确认 train_data.csv 位于当前项目工作目录中。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    # 清除列名前后的空格
    data.columns = data.columns.astype(str).str.strip()

    # 用户描述中同时出现了 DIAGNOSIS 和 DIAGONSIS。
    # 如果CSV使用了拼写 DIAGONSIS，则自动改为 DIAGNOSIS。
    if "DIAGNOSIS" not in data.columns and "DIAGONSIS" in data.columns:
        data = data.rename(columns={"DIAGONSIS": "DIAGNOSIS"})
        warnings.warn(
            "检测到列名 DIAGONSIS，已自动重命名为 DIAGNOSIS。",
            stacklevel=2,
        )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要列：\n"
            f"{missing_columns}\n\n"
            f"当前数据列为：\n{data.columns.tolist()}"
        )

    if data.empty:
        raise ValueError("数据集为空，无法构建模型。")

    duplicated_columns = data.columns[data.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(
            f"数据集中存在重复列名：{duplicated_columns}"
        )

    return data


def prepare_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    准备特征和二分类目标变量。

    原始TRUST：
        TRUST < 16  -> 0
        TRUST >= 16 -> 1
    """
    features = data[FEATURE_COLUMNS].copy()

    # 分类变量按题目描述应不存在缺失值
    categorical_columns = ONE_HOT_COLUMNS + ORDINAL_COLUMNS
    categorical_missing = features[categorical_columns].isna().sum()
    categorical_missing = categorical_missing[categorical_missing > 0]

    if not categorical_missing.empty:
        raise ValueError(
            "以下分类变量存在缺失值，但当前方案未对分类变量进行填补：\n"
            f"{categorical_missing.to_string()}"
        )

    # 将连续变量强制转换为数值。
    # 非法字符串会变成NaN，随后由中位数填补。
    for column in CONTINUOUS_COLUMNS:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    # TRUST不能进行缺失值填补，因此非法值直接报错
    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    invalid_target_mask = trust_numeric.isna()
    if invalid_target_mask.any():
        invalid_examples = (
            data.loc[invalid_target_mask, TARGET_COLUMN]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"目标列 {TARGET_COLUMN} 中存在缺失值或非数值内容。\n"
            f"示例：{invalid_examples}"
        )

    target = (trust_numeric >= 16).astype(int)
    target.name = "TRUST_GE_16"

    class_counts = target.value_counts().sort_index()

    if target.nunique() != 2:
        raise ValueError(
            "目标变量转换后不是二分类数据。\n"
            f"当前类别分布：{class_counts.to_dict()}\n"
            "请确认TRUST列中同时存在小于16和大于等于16的样本。"
        )

    if class_counts.min() < 2:
        raise ValueError(
            "少数类别样本数小于2，无法进行分层训练集/测试集划分。"
        )

    return features, target


def determine_cv_and_smote_parameters(
    y_train: pd.Series,
) -> tuple[int, int]:
    """
    根据训练集少数类别样本数，确定：
    1. 分层交叉验证折数
    2. SMOTE的k_neighbors

    默认最多使用5折交叉验证和k_neighbors=5。
    当样本量较小时自动降低参数，避免SMOTE报错。
    """
    minimum_class_count = int(y_train.value_counts().min())

    cv_folds = min(MAX_CV_FOLDS, minimum_class_count)

    if cv_folds < 2:
        raise ValueError(
            "训练集中少数类别样本数不足，无法进行交叉验证。"
        )

    # 在某个交叉验证训练折中，少数类别可能出现的最小样本数
    maximum_validation_minority = math.ceil(
        minimum_class_count / cv_folds
    )
    minimum_training_fold_minority = (
        minimum_class_count - maximum_validation_minority
    )

    # SMOTE要求 k_neighbors 小于当前少数类别样本数
    smote_k_neighbors = min(
        5,
        minimum_training_fold_minority - 1,
    )

    if smote_k_neighbors < 1:
        raise ValueError(
            "少数类别样本数过少，无法在交叉验证过程中使用SMOTE。\n"
            f"训练集少数类别样本数：{minimum_class_count}"
        )

    return cv_folds, smote_k_neighbors


def build_preprocessor() -> ColumnTransformer:
    """
    构建数据预处理器。
    """
    one_hot_encoder = create_one_hot_encoder()

    # 未明确提供TPPA各类别的临床顺序，因此由OrdinalEncoder
    # 根据训练数据中的类别自动确定编码。
    #
    # 如果TPPA有明确顺序，例如：
    # 阴性 < 弱阳性 < 阳性 < 强阳性
    #
    # 可修改为：
    # OrdinalEncoder(
    #     categories=[["阴性", "弱阳性", "阳性", "强阳性"]],
    #     handle_unknown="use_encoded_value",
    #     unknown_value=-1,
    # )
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )

    continuous_pipeline = SklearnPipeline(
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
                continuous_pipeline,
                CONTINUOUS_COLUMNS,
            ),
        ],
        remainder="drop",

        # 强制输出稠密矩阵，方便后续SMOTE处理
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )

    return preprocessor


def build_model_pipeline(
    smote_k_neighbors: int,
) -> ImbalancedPipeline:
    """
    构建预处理、SMOTE和随机森林的完整管道。
    """
    preprocessor = build_preprocessor()

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=smote_k_neighbors,
    )

    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,

        # 明确禁止随机森林内部使用多进程
        n_jobs=1,
    )

    model_pipeline = ImbalancedPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("classifier", classifier),
        ]
    )

    return model_pipeline


def evaluate_model(
    model: ImbalancedPipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    在独立测试集上计算模型评估指标。
    """
    y_pred = model.predict(x_test)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("当前模型不支持predict_proba，无法计算AUC。")

    class_labels = np.asarray(model.classes_)
    positive_class_positions = np.where(class_labels == 1)[0]

    if len(positive_class_positions) != 1:
        raise ValueError(
            f"无法确定阳性类别1的位置，模型类别为：{class_labels.tolist()}"
        )

    positive_class_index = int(positive_class_positions[0])
    y_probability = model.predict_proba(x_test)[:, positive_class_index]

    if y_test.nunique() == 2:
        auc_value = roc_auc_score(y_test, y_probability)
    else:
        auc_value = np.nan
        warnings.warn(
            "测试集中只包含一个类别，无法计算ROC AUC。",
            stacklevel=2,
        )

    evaluation_results = pd.DataFrame(
        {
            "评估指标": [
                "Accuracy",
                "Recall",
                "Precision",
                "F1-score",
                "ROC AUC",
            ],
            "测试集结果": [
                accuracy_score(y_test, y_pred),
                recall_score(
                    y_test,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                precision_score(
                    y_test,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                f1_score(
                    y_test,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                auc_value,
            ],
        }
    )

    return evaluation_results


# ============================================================
# 3. 主程序
# ============================================================

def main() -> None:
    print("=" * 70)
    print("随机森林二分类模型：预测 TRUST 是否大于等于16")
    print("=" * 70)

    # --------------------------------------------------------
    # 读取并准备数据
    # --------------------------------------------------------
    data = load_and_validate_data(DATA_FILE)
    x, y = prepare_features_and_target(data)

    print(f"\n数据总行数：{len(data)}")
    print(f"模型特征数：{len(FEATURE_COLUMNS)}")

    print("\n目标变量定义：")
    print("0：TRUST < 16")
    print("1：TRUST >= 16")

    print("\n整体目标类别分布：")
    print(y.value_counts().sort_index().to_string())

    print("\n整体目标类别比例：")
    print(
        y.value_counts(normalize=True)
        .sort_index()
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # 划分训练集和测试集
    # 必须在SMOTE之前划分，测试集不进行SMOTE
    # --------------------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\n训练集样本数：{len(x_train)}")
    print(f"测试集样本数：{len(x_test)}")

    print("\n训练集类别分布：")
    print(y_train.value_counts().sort_index().to_string())

    print("\n测试集类别分布：")
    print(y_test.value_counts().sort_index().to_string())

    # --------------------------------------------------------
    # 根据少数类别样本数确定交叉验证和SMOTE参数
    # --------------------------------------------------------
    cv_folds, smote_k_neighbors = determine_cv_and_smote_parameters(
        y_train
    )

    print(f"\n分层交叉验证折数：{cv_folds}")
    print(f"SMOTE k_neighbors：{smote_k_neighbors}")

    # --------------------------------------------------------
    # 构建完整管道
    # --------------------------------------------------------
    model_pipeline = build_model_pipeline(
        smote_k_neighbors=smote_k_neighbors
    )

    # 用户指定的随机森林参数
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 同时记录多个交叉验证指标，以ROC AUC选择最佳模型
    scoring_metrics = {
        "accuracy": "accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring=scoring_metrics,

        # 根据平均ROC AUC选择并重新训练最佳模型
        refit="roc_auc",

        cv=cross_validation,

        # 不使用多进程
        n_jobs=1,

        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    # --------------------------------------------------------
    # 超参数搜索
    # SMOTE只会对每个交叉验证训练折执行
    # --------------------------------------------------------
    print("\n开始进行超参数调优……")
    grid_search.fit(x_train, y_train)

    print("\n超参数调优完成。")
    print("\n最佳参数：")
    for parameter_name, parameter_value in grid_search.best_params_.items():
        clean_name = parameter_name.replace("classifier__", "")
        print(f"{clean_name}: {parameter_value}")

    print(
        "\n最佳交叉验证平均ROC AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # --------------------------------------------------------
    # 输出所有参数组合的交叉验证结果
    # --------------------------------------------------------
    cv_results = pd.DataFrame(grid_search.cv_results_)

    cv_summary = cv_results[
        [
            "param_classifier__n_estimators",
            "mean_test_accuracy",
            "mean_test_recall",
            "mean_test_precision",
            "mean_test_f1",
            "mean_test_roc_auc",
            "rank_test_roc_auc",
        ]
    ].copy()

    cv_summary = cv_summary.rename(
        columns={
            "param_classifier__n_estimators": "n_estimators",
            "mean_test_accuracy": "CV_Accuracy",
            "mean_test_recall": "CV_Recall",
            "mean_test_precision": "CV_Precision",
            "mean_test_f1": "CV_F1",
            "mean_test_roc_auc": "CV_ROC_AUC",
            "rank_test_roc_auc": "AUC排名",
        }
    )

    cv_summary = cv_summary.sort_values("AUC排名")

    print("\n各参数组合的交叉验证结果：")
    print(
        cv_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # 独立测试集评估
    # --------------------------------------------------------
    best_model = grid_search.best_estimator_

    evaluation_results = evaluate_model(
        model=best_model,
        x_test=x_test,
        y_test=y_test,
    )

    print("\n独立测试集评估结果：")
    print(
        evaluation_results.to_string(
            index=False,
            formatters={
                "测试集结果": lambda value: (
                    "无法计算"
                    if pd.isna(value)
                    else f"{value:.4f}"
                )
            },
        )
    )

    # --------------------------------------------------------
    # 输出TPPA实际学习到的类别顺序
    # --------------------------------------------------------
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    fitted_ordinal_encoder = (
        fitted_preprocessor.named_transformers_["ordinal"]
    )

    tppa_categories = fitted_ordinal_encoder.categories_[0].tolist()

    print("\nTPPA序数编码所使用的类别顺序：")
    for index, category in enumerate(tppa_categories):
        print(f"{category!r} -> {index}")

    print("测试集中未见过的新TPPA类别将编码为：-1")

    # --------------------------------------------------------
    # 保存包含全部预处理步骤的最佳模型
    # --------------------------------------------------------
    joblib.dump(best_model, MODEL_FILE)

    print(
        "\n最佳模型已保存至："
        f"{MODEL_FILE.resolve()}"
    )
    print(
        "保存文件包含：独热编码、序数编码、中位数填补、"
        "SMOTE配置和随机森林模型。"
    )


if __name__ == "__main__":
    main()