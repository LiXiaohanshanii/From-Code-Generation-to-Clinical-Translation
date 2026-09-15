# -*- coding: utf-8 -*-

"""
随机森林模型预测 TRUST >=16
适用于 PyCharm 2025.2.3

"""

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
    roc_auc_score,
    classification_report
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================
# 1. 读取数据
# ============================

data_path = "train_data.csv"

df = pd.read_csv(
    data_path,
    encoding="utf-8"
)


print("数据维度：")
print(df.shape)

print("\n数据前5行：")
print(df.head())


# ============================
# 2. 构建二分类目标变量
# TRUST >=16 为1，否则为0
# ============================

df["TRUST_binary"] = np.where(
    df["TRUST"] >= 16,
    1,
    0
)


# ============================
# 3. 定义特征和标签
# ============================

feature_columns = [
    "SEX",
    "AGE",
    "DEPT",
    "DIAGNOSIS",
    "TPPA",
    "TP",
    "HIV",
    "WBC",
    "RBC",
    "PLT",
    "NC",
    "LY",
    "NLR"
]


X = df[feature_columns]

y = df["TRUST_binary"]


# ============================
# 4. 划分训练集和测试集
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n训练集类别分布：")
print(y_train.value_counts())

print("\n测试集类别分布：")
print(y_test.value_counts())


# ============================
# 5. 定义变量类型
# ============================

# One-Hot编码变量
onehot_features = [
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]


# Ordinal编码变量
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


# ============================
# 6. 数据预处理
# ============================


# 连续变量：
# 中位数填充
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# One-Hot编码
onehot_transformer = Pipeline(
    steps=[
        (
            "onehot",
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
            "ordinal",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)



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



# ============================
# 7. 构建随机森林模型
# ============================


rf_model = RandomForestClassifier(
    random_state=42
)



# ============================
# 8. 建立Pipeline
# SMOTE必须位于预处理之后
# ============================


pipeline = ImbPipeline(
    steps=[

        (
            "preprocess",
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
            rf_model
        )
    ]
)



# ============================
# 9. 超参数搜索
# 不使用多进程
# ============================


param_grid = {

    "classifier__n_estimators":
        [100, 200],

    "classifier__max_depth":
        [10],

    "classifier__min_samples_split":
        [2],

    "classifier__min_samples_leaf":
        [1],

    "classifier__class_weight":
        ["balanced"]

}



grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=1,
    verbose=1
)



# ============================
# 10. 模型训练
# ============================

print("\n开始模型训练...")

grid_search.fit(
    X_train,
    y_train
)



print("\n最佳参数：")
print(grid_search.best_params_)



print("\n最佳交叉验证AUC：")
print(grid_search.best_score_)



# ============================
# 11. 测试集预测
# ============================


best_model = grid_search.best_estimator_


y_pred = best_model.predict(
    X_test
)


# 获取预测概率用于AUC
y_prob = best_model.predict_proba(
    X_test
)[:, 1]



# ============================
# 12. 模型评价
# ============================


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



print("\n==============================")
print("随机森林模型性能评价")
print("==============================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print(
    f"AUC       : {auc:.4f}"
)



print("\n分类报告：")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================
# 13. 保存预测结果
# ============================

result = X_test.copy()

result["True_TRUST_binary"] = y_test.values

result["Predicted_TRUST_binary"] = y_pred

result["Predicted_probability"] = y_prob


result.to_csv(
    "RF_prediction_results.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n预测结果已保存：RF_prediction_results.csv")