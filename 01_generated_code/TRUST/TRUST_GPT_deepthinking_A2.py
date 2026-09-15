from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# =========================
# 1. 基本配置
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20

# CSV 文件与当前 Python 脚本位于同一目录
DATA_PATH = Path(__file__).resolve().parent / "train_data.csv"

CATEGORICAL_ONEHOT_COLUMNS = [
    "SEX",
    "DEPT",
    "DIAGNOSIS",
]

CATEGORICAL_ORDINAL_COLUMNS = [
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
    CATEGORICAL_ONEHOT_COLUMNS
    + CATEGORICAL_ORDINAL_COLUMNS
    + CONTINUOUS_COLUMNS
)

TARGET_COLUMN = "TRUST"


# =========================
# 2. 读取并检查数据
# =========================
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"未找到数据文件：{DATA_PATH}\n"
        "请将 train_data.csv 放在当前 Python 脚本所在目录。"
    )

data = pd.read_csv(DATA_PATH, encoding="utf-8")

# 清除列名前后的空格
data.columns = data.columns.str.strip()

# 兼容可能出现的拼写 DIAGONSIS
if "DIAGONSIS" in data.columns and "DIAGNOSIS" not in data.columns:
    data = data.rename(columns={"DIAGONSIS": "DIAGNOSIS"})

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

# 仅保留本次建模需要的列
data = data[required_columns].copy()


# =========================
# 3. 构建二分类目标
#    TRUST >= 16 记为 1
#    TRUST < 16 记为 0
# =========================
trust_numeric = pd.to_numeric(
    data[TARGET_COLUMN],
    errors="coerce",
)

if trust_numeric.isna().any():
    invalid_values = (
        data.loc[trust_numeric.isna(), TARGET_COLUMN]
        .astype(str)
        .unique()
        .tolist()
    )
    raise ValueError(
        "TRUST 列包含无法转换为数值的内容："
        f"{invalid_values}"
    )

X = data[FEATURE_COLUMNS].copy()
y = (trust_numeric >= 16).astype(int)

if y.nunique() != 2:
    raise ValueError(
        "根据 TRUST >= 16 转换后，目标变量没有同时包含两个类别。"
        f"当前类别分布为：{y.value_counts().to_dict()}"
    )

print("完整数据集目标类别分布：")
print(y.value_counts().rename(index={0: "TRUST < 16", 1: "TRUST >= 16"}))
print()


# =========================
# 4. 划分训练集和测试集
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("训练集目标类别分布：")
print(y_train.value_counts().rename(index={0: "TRUST < 16", 1: "TRUST >= 16"}))
print()

print("测试集目标类别分布：")
print(y_test.value_counts().rename(index={0: "TRUST < 16", 1: "TRUST >= 16"}))
print()


# =========================
# 5. 数据预处理
# =========================

# SEX、DEPT、DIAGNOSIS：独热编码
onehot_pipeline = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        )
    ]
)

# TPPA：序数编码
# 未明确指定 TPPA 各类别的业务顺序，因此由编码器自动确定类别顺序。
# 测试集中出现训练集没有见过的类别时编码为 -1。
ordinal_pipeline = Pipeline(
    steps=[
        (
            "ordinal",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        )
    ]
)

# 连续变量：中位数填补缺失值
continuous_pipeline = Pipeline(
    steps=[
        (
            "median_imputer",
            SimpleImputer(strategy="median"),
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "onehot_features",
            onehot_pipeline,
            CATEGORICAL_ONEHOT_COLUMNS,
        ),
        (
            "ordinal_features",
            ordinal_pipeline,
            CATEGORICAL_ORDINAL_COLUMNS,
        ),
        (
            "continuous_features",
            continuous_pipeline,
            CONTINUOUS_COLUMNS,
        ),
    ],
    remainder="drop",
)


# =========================
# 6. 根据少数类样本数设置交叉验证
#    和 SMOTE 的 k_neighbors
# =========================
minority_count = int(y_train.value_counts().min())

if minority_count < 3:
    raise ValueError(
        "训练集中少数类样本数少于 3，无法可靠地同时执行"
        "分层交叉验证和 SMOTE。"
    )

# 最多使用 5 折；少数类较少时自动减少折数
cv_splits = min(5, minority_count)

# 估算每个交叉验证训练折中最少的少数类样本数
maximum_validation_minority = (
    minority_count + cv_splits - 1
) // cv_splits

minimum_training_minority = (
    minority_count - maximum_validation_minority
)

if minimum_training_minority < 2:
    raise ValueError(
        "少数类样本数不足，无法在交叉验证训练折内执行 SMOTE。"
    )

# SMOTE 要求 k_neighbors 小于训练折中的少数类样本数
smote_k_neighbors = min(
    5,
    minimum_training_minority - 1,
)

print(f"交叉验证折数：{cv_splits}")
print(f"SMOTE k_neighbors：{smote_k_neighbors}")
print()


# =========================
# 7. 构建完整建模流水线
# =========================
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
        (
            "classifier",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,  # 随机森林不使用多进程
            ),
        ),
    ]
)


# =========================
# 8. 超参数网格
# =========================
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

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=parameter_grid,
    scoring="roc_auc",
    cv=cross_validation,
    refit=True,
    n_jobs=1,       # 网格搜索不使用多进程
    verbose=1,
    return_train_score=False,
)


# =========================
# 9. 模型训练与超参数搜索
# =========================
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n最优超参数：")
for parameter_name, parameter_value in grid_search.best_params_.items():
    print(f"{parameter_name}: {parameter_value}")

print(
    "\n交叉验证最优平均 AUC："
    f"{grid_search.best_score_:.4f}"
)


# =========================
# 10. 在独立测试集上进行评估
# =========================
y_pred = best_model.predict(X_test)

positive_class_index = list(best_model.classes_).index(1)
y_pred_probability = best_model.predict_proba(X_test)[
    :,
    positive_class_index,
]

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
auc = roc_auc_score(
    y_test,
    y_pred_probability,
)

evaluation_results = pd.DataFrame(
    {
        "评估指标": [
            "Accuracy",
            "Recall",
            "Precision",
            "F1-score",
            "AUC",
        ],
        "测试集结果": [
            accuracy,
            recall,
            precision,
            f1,
            auc,
        ],
    }
)

print("\n独立测试集评估结果：")
print(
    evaluation_results.to_string(
        index=False,
        formatters={
            "测试集结果": lambda value: f"{value:.4f}"
        },
    )
)