from pathlib import Path

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
from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline


# =========================
# 基本配置
# =========================
DATA_PATH = Path("train_data.csv")
TARGET_COLUMN = "TRUST"
POSITIVE_THRESHOLD = 16

TEST_SIZE = 0.20
RANDOM_STATE = 42


# 独热编码变量
ONE_HOT_FEATURES = [
    "SEX",
    "DEPT",
    "DIAGNOSIS",
]

# 序数编码变量
ORDINAL_FEATURES = [
    "TPPA",
]

# 连续变量
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

FEATURE_COLUMNS = (
    ONE_HOT_FEATURES
    + ORDINAL_FEATURES
    + CONTINUOUS_FEATURES
)


def create_one_hot_encoder() -> OneHotEncoder:
    """
    创建独热编码器。

    sparse_output=False 适用于较新的 scikit-learn 版本；
    如果运行环境使用旧版本，则自动改用 sparse=False。
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def load_and_prepare_data(
    file_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    读取并初步整理数据。

    TRUST >= 16 转换为正类1；
    TRUST < 16 转换为负类0。
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件：{file_path.resolve()}"
        )

    data = pd.read_csv(file_path, encoding="utf-8")

    if data.empty:
        raise ValueError("数据文件为空。")

    # 检查目标列是否为最后一列
    if data.columns[-1] != TARGET_COLUMN:
        raise ValueError(
            f"目标列必须是最后一列且名称为 {TARGET_COLUMN}。"
            f"当前最后一列为：{data.columns[-1]}"
        )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"数据中缺少以下必要列：{missing_columns}"
        )

    # 仅保留指定特征
    x = data[FEATURE_COLUMNS].copy()

    # 将独热编码变量统一转换为字符串，避免混合数据类型
    for column in ONE_HOT_FEATURES:
        x[column] = x[column].astype(str).str.strip()

    # TPPA保留原始类别含义，但统一转换为字符串
    x["TPPA"] = x["TPPA"].astype(str).str.strip()

    # 连续变量转换为数值。
    # 无法转换的内容会变为NaN，之后由中位数填充。
    for column in CONTINUOUS_FEATURES:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    # 检查并转换目标列
    trust_numeric = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    if trust_numeric.isna().any():
        invalid_values = (
            data.loc[trust_numeric.isna(), TARGET_COLUMN]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "TRUST列包含无法转换为数值的内容。"
            f"示例异常值：{invalid_values[:10]}"
        )

    # 二分类标签：TRUST >= 16 为1，否则为0
    y = (trust_numeric >= POSITIVE_THRESHOLD).astype(int)
    y.name = "TRUST_GE_16"

    if y.nunique() != 2:
        raise ValueError(
            "目标变量转换后没有同时包含0和1两个类别。"
            "请检查TRUST列中是否同时存在小于16和大于等于16的数据。"
        )

    return x, y


def build_model_pipeline() -> ImbalancedPipeline:
    """
    创建预处理、SMOTE和随机森林组成的完整管道。
    """

    # 连续变量：中位数填充
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
                create_one_hot_encoder(),
                ONE_HOT_FEATURES,
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL_FEATURES,
            ),
            (
                "continuous",
                continuous_pipeline,
                CONTINUOUS_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,  # 不使用多进程
    )

    # 必须使用imblearn的Pipeline，才能在管道内使用SMOTE
    model_pipeline = ImbalancedPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(random_state=RANDOM_STATE),
            ),
            ("classifier", random_forest),
        ]
    )

    return model_pipeline


def main() -> None:
    # 1. 读取和整理数据
    x, y = load_and_prepare_data(DATA_PATH)

    print("数据集基本信息")
    print(f"样本数量：{len(x)}")
    print(f"特征数量：{x.shape[1]}")
    print("\n二分类目标分布：")
    print(y.value_counts().sort_index())
    print("\n目标类别比例：")
    print(y.value_counts(normalize=True).sort_index())

    # 分层划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n训练集样本数：", len(x_train))
    print("测试集样本数：", len(x_test))

    # SMOTE默认k_neighbors=5，因此训练数据中的少数类
    # 至少需要具有一定数量的样本
    minority_count = int(y_train.value_counts().min())

    if minority_count < 6:
        raise ValueError(
            "训练集中少数类别样本少于6个，"
            "无法安全使用默认参数k_neighbors=5的SMOTE。"
            "可以增加样本量，或根据实际情况降低SMOTE的k_neighbors。"
        )

    # 2. 构建模型管道
    model_pipeline = build_model_pipeline()

    # 3. 指定超参数搜索范围
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    # 分层5折交叉验证
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 以ROC-AUC作为超参数选择指标
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cross_validation,
        refit=True,
        n_jobs=1,  # 不使用多进程
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    # 4. 训练和调参
    print("\n开始进行网格搜索和模型训练……")
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    print("\n最优超参数：")
    for parameter_name, parameter_value in (
        grid_search.best_params_.items()
    ):
        print(f"{parameter_name}: {parameter_value}")

    print(
        "\n交叉验证最优平均AUC："
        f"{grid_search.best_score_:.4f}"
    )

    # 5. 在测试集上预测
    y_pred = best_model.predict(x_test)
    y_probability = best_model.predict_proba(x_test)[:, 1]

    # 6. 计算评价指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0,
    )
    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0,
    )
    auc = roc_auc_score(y_test, y_probability)

    # 7. 输出测试集结果
    print("\n测试集模型评估结果")
    print("-" * 35)
    print(f"Accuracy ：{accuracy:.4f}")
    print(f"Recall   ：{recall:.4f}")
    print(f"Precision：{precision:.4f}")
    print(f"F1-score ：{f1:.4f}")
    print(f"ROC-AUC  ：{auc:.4f}")


if __name__ == "__main__":
    main()