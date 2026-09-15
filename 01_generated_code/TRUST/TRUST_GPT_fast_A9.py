import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ==============================
# 1. 数据读取
# ==============================

data_path = "train_data.csv"

df = pd.read_csv(
    data_path,
    encoding="utf-8"
)

print("数据维度:")
print(df.shape)

print("\n数据前5行:")
print(df.head())


# ==============================
# 2. 构建二分类目标变量
# TRUST >=16 为1，否则为0
# ==============================

df["TARGET"] = (df["TRUST"] >= 16).astype(int)


# 删除原始TRUST
X = df.drop(columns=["TRUST", "TARGET"])
y = df["TARGET"]


# ==============================
# 3. 定义变量类型
# ==============================

# One-Hot变量
onehot_features = [
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]


# Ordinal变量
ordinal_features = [
    "TPPA"
]


# 连续变量
numeric_features = [
    "AGE",
    "TP",
    "HIV",
    "WBC",
    "RBC",
    "PLT",
    "NC",
    "LY",
    "NLR"
]


# ==============================
# 4. 数据预处理
# ==============================

# 连续变量：
# 中位数填充
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# One-Hot编码
onehot_transformer = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# TPPA序数编码
ordinal_transformer = Pipeline(
    steps=[
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


# 汇总预处理
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "onehot",
            onehot_transformer,
            onehot_features
        ),
        (
            "ordinal",
            ordinal_transformer,
            ordinal_features
        )
    ]
)


# ==============================
# 5. 划分训练集和测试集
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n训练集类别分布:")
print(y_train.value_counts())

print("\n测试集类别分布:")
print(y_test.value_counts())


# ==============================
# 6. 构建随机森林模型
# ==============================

rf = RandomForestClassifier(
    random_state=42
)


# 使用SMOTE
# 注意：
# SMOTE必须放在编码之后，因此放在Pipeline中
model_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "classifier",
            rf
        )
    ]
)


# ==============================
# 7. 超参数调优
# ==============================

param_grid = {

    "classifier__n_estimators": [
        100,
        200
    ],

    "classifier__max_depth": [
        10
    ],

    "classifier__min_samples_split": [
        2
    ],

    "classifier__min_samples_leaf": [
        1
    ],

    "classifier__class_weight": [
        "balanced"
    ]
}


grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=1,       # 不使用多进程
    verbose=1
)


# ==============================
# 8. 模型训练
# ==============================

print("\n开始模型训练...")

grid_search.fit(
    X_train,
    y_train
)


print("\n最佳参数:")
print(grid_search.best_params_)


print("\n最佳交叉验证AUC:")
print(grid_search.best_score_)


# 最优模型
best_model = grid_search.best_estimator_


# ==============================
# 9. 测试集预测
# ==============================

y_pred = best_model.predict(
    X_test
)


# AUC需要概率
y_prob = best_model.predict_proba(
    X_test
)[:, 1]


# ==============================
# 10. 模型评价
# ==============================

accuracy = accuracy_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n========== 模型评价结果 ==========")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)

print(
    f"AUC      : {auc:.4f}"
)

print(
    "=================================="
)