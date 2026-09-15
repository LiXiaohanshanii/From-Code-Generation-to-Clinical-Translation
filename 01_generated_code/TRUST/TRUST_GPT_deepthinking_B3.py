from pathlib import Path
import sys
import warnings

import joblib
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
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# =========================
# 基本参数
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_FILE = Path("train_data.csv")
MODEL_FILE = Path("random_forest_trust_model.joblib")

TARGET_COLUMN = "TRUST"

# 独热编码变量
ONEHOT_COLUMNS = [
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

FEATURE_COLUMNS = ONEHOT_COLUMNS + ORDINAL_COLUMNS + CONTINUOUS_COLUMNS


def create_one_hot_encoder() -> OneHotEncoder:
    """
    创建OneHotEncoder。

    sparse_output=False适用于较新版本的scikit-learn。
    sparse=False用于兼容较旧版本。
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
    读取并检查数据。
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}\n"
            "请将train_data.csv放在当前项目运行目录下，"
            "或修改DATA_FILE为正确路径。"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    if data.empty:
        raise ValueError("train_data.csv中没有数据。")

    # 去除列名前后的空格，避免因隐藏空格导致找不到列
    data.columns = data.columns.str.strip()

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "数据集中缺少以下必要列："
            f"{missing_columns}\n"
            f"当前数据列为：{data.columns.tolist()}"
        )

    if data.columns[-1] != TARGET_COLUMN:
        warnings.warn(
            f"目标列{TARGET_COLUMN}不是数据集最后一列，"
            "程序仍将按列名读取该目标列。",
            UserWarning,
        )

    # 仅保留建模需要的列
    data = data[required_columns].copy()

    # 将连续变量转换为数值。
    # 非法字符会转换为NaN，随后由中位数填充。
    for column in CONTINUOUS_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # 分类变量无缺失值；统一转为字符串，避免同一列中混合数字和文本
    categorical_columns = ONEHOT_COLUMNS + ORDINAL_COLUMNS
    for column in categorical_columns:
        if data[column].isna().any():
            raise ValueError(
                f"分类变量{column}存在缺失值，"
                "但当前需求设定分类变量应无缺失值。"
            )

        data[column] = data[column].astype(str).str.strip()

    # TRUST必须可以转换为数值
    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    missing_target_count = data[TARGET_COLUMN].isna().sum()

    if missing_target_count > 0:
        warnings.warn(
            f"目标列{TARGET_COLUMN}中有{missing_target_count}行"
            "无法转换为数值，这些行将被删除。",
            UserWarning,
        )
        data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    if data.empty:
        raise ValueError("删除目标列无效数据后，没有可用于建模的数据。")

    return data


def create_binary_target(trust_values: pd.Series) -> pd.Series:
    """
    将TRUST转换为二分类标签：

    TRUST >= 16：1
    TRUST < 16 ：0
    """
    return (trust_values >= 16).astype(int)


def determine_cv_and_smote_parameters(
    y_train: pd.Series,
) -> tuple[int, int]:
    """
    根据训练集中少数类别样本数，确定：

    1. 分层交叉验证折数
    2. SMOTE的k_neighbors

    这样可以降低少数类别样本较少时SMOTE报错的风险。
    """
    class_counts = y_train.value_counts()

    if len(class_counts) != 2:
        raise ValueError(
            "训练集没有同时包含两个类别。"
            "请检查TRUST数据或调整测试集比例。"
        )

    minority_count = int(class_counts.min())

    if minority_count < 2:
        raise ValueError(
            "训练集少数类别样本数少于2，无法执行SMOTE。"
        )

    # 最多使用5折，但每个类别至少需要有与折数相当的样本数
    cv_splits = min(5, minority_count)

    if cv_splits < 2:
        raise ValueError("数据量不足，无法进行分层交叉验证。")

    # 估计每个交叉验证训练折中的最少少数类样本数
    estimated_fold_minority = int(
        np.floor(minority_count * (cv_splits - 1) / cv_splits)
    )

    # SMOTE要求k_neighbors小于少数类别样本数
    smote_k_neighbors = max(
        1,
        min(5, estimated_fold_minority - 1),
    )

    return cv_splits, smote_k_neighbors


def build_pipeline(
    smote_k_neighbors: int,
) -> ImbPipeline:
    """
    构建预处理、SMOTE和随机森林模型Pipeline。
    """

    # 独热编码流程
    onehot_pipeline = SklearnPipeline(
        steps=[
            (
                "onehot",
                create_one_hot_encoder(),
            ),
        ]
    )

    # TPPA序数编码流程
    ordinal_pipeline = SklearnPipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories="auto",
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    # 连续变量中位数填充流程
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
                "onehot_features",
                onehot_pipeline,
                ONEHOT_COLUMNS,
            ),
            (
                "ordinal_features",
                ordinal_pipeline,
                ORDINAL_COLUMNS,
            ),
            (
                "continuous_features",
                continuous_pipeline,
                CONTINUOUS_COLUMNS,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    # 使用imblearn的Pipeline，保证SMOTE只在训练数据上执行
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
                random_forest,
            ),
        ]
    )

    return pipeline


def evaluate_model(
    model: GridSearchCV,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    在独立测试集上评估模型。
    """
    y_pred = model.predict(x_test)
    y_probability = model.predict_proba(x_test)[:, 1]

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

    result = pd.DataFrame(
        {
            "评价指标": list(metrics.keys()),
            "测试集结果": list(metrics.values()),
        }
    )

    return result


def main() -> None:
    print("=" * 70)
    print("随机森林二分类模型：预测TRUST是否大于或等于16")
    print("=" * 70)

    # 1. 读取数据
    data = load_and_validate_data(DATA_FILE)

    x = data[FEATURE_COLUMNS].copy()
    y = create_binary_target(data[TARGET_COLUMN])

    print(f"\n总样本数：{len(data)}")
    print("\n二分类标签定义：")
    print("0：TRUST < 16")
    print("1：TRUST >= 16")

    print("\n完整数据集类别分布：")
    class_distribution = pd.DataFrame(
        {
            "样本数": y.value_counts().sort_index(),
            "比例": y.value_counts(
                normalize=True
            ).sort_index(),
        }
    )
    print(class_distribution)

    if y.nunique() != 2:
        raise ValueError(
            "转换后的目标变量不是二分类数据。"
            "请确认TRUST列中同时存在小于16和大于等于16的样本。"
        )

    # 2. 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\n训练集样本数：{len(x_train)}")
    print(f"测试集样本数：{len(x_test)}")

    print("\n训练集类别分布（SMOTE前）：")
    print(y_train.value_counts().sort_index())

    print("\n测试集类别分布：")
    print(y_test.value_counts().sort_index())

    if y_test.nunique() != 2:
        raise ValueError(
            "测试集中没有同时包含两个类别，无法计算AUC。"
            "请增加样本量或调整TEST_SIZE。"
        )

    # 3. 根据少数类样本数量确定CV折数和SMOTE参数
    cv_splits, smote_k_neighbors = determine_cv_and_smote_parameters(
        y_train
    )

    print(f"\n分层交叉验证折数：{cv_splits}")
    print(f"SMOTE k_neighbors：{smote_k_neighbors}")

    # 4. 构建Pipeline
    pipeline = build_pipeline(smote_k_neighbors)

    # 5. 设置超参数搜索空间
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    stratified_cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 使用AUC选择最佳模型
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=stratified_cv,
        refit=True,
        n_jobs=1,             # 明确禁用多进程
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    # 6. 模型训练和超参数搜索
    print("\n开始执行网格搜索和模型训练……")
    grid_search.fit(x_train, y_train)

    print("\n模型训练完成。")

    readable_best_params = {
        key.replace("classifier__", ""): value
        for key, value in grid_search.best_params_.items()
    }

    print("\n最佳超参数：")
    for parameter, value in readable_best_params.items():
        print(f"{parameter}: {value}")

    print(
        "\n最佳交叉验证平均AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # 7. 测试集评估
    evaluation_result = evaluate_model(
        grid_search,
        x_test,
        y_test,
    )

    print("\n独立测试集评价结果：")
    print(
        evaluation_result.to_string(
            index=False,
            formatters={
                "测试集结果": lambda value: f"{value:.4f}"
            },
        )
    )

    # 8. 输出分类报告
    y_pred = grid_search.predict(x_test)

    print("\n测试集分类报告：")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "TRUST < 16",
                "TRUST >= 16",
            ],
            digits=4,
            zero_division=0,
        )
    )

    # 9. 保存最佳模型
    joblib.dump(
        grid_search.best_estimator_,
        MODEL_FILE,
    )

    print(
        f"\n最佳模型已保存至：{MODEL_FILE.resolve()}"
    )

    # 10. 保存评价指标，便于后续查看
    evaluation_file = Path("random_forest_evaluation.csv")
    evaluation_result.to_csv(
        evaluation_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"评价指标已保存至：{evaluation_file.resolve()}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n程序运行失败：{exc}", file=sys.stderr)
        sys.exit(1)