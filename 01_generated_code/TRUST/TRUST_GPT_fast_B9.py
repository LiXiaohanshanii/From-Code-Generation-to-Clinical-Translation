# -*- coding: utf-8 -*-

"""
随机森林模型预测 TRUST >=16
适用于 PyCharm 2025.2.3

功能：
1. 数据读取
2. 数据预处理
3. SMOTE类别平衡
4. 随机森林建模
5. GridSearch超参数优化
6. 模型评价
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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



# ===============================
# 1. 读取数据
# ===============================

data_path = "train_data.csv"

df = pd.read_csv(
    data_path,
    encoding="utf-8"
)


print("数据维度：")
print(df.shape)


# ===============================
# 2. 构建目标变量
# ===============================

# TRUST为最后一列
df["TRUST_binary"] = (
    df["TRUST"] >= 16
).astype(int)


print("\n目标变量分布：")
print(df["TRUST_binary"].value_counts())



# ===============================
# 3. 定义特征
# ===============================


features = [
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


X = df[features]

y = df["TRUST_binary"]



# ===============================
# 4. 划分训练集和测试集
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# ===============================
# 5. 定义变量类型
# ===============================


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
continuous_features = [
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



# ===============================
# 6. 数据预处理
# ===============================


preprocessor = ColumnTransformer(

    transformers=[

        # 分类变量One-Hot
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            onehot_features
        ),


        # TPPA序数编码
        (
            "ordinal",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ordinal_features
        ),


        # 连续变量中位数填充
        (
            "continuous",
            Pipeline(
                steps=[

                    (
                        "median_imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    )

                ]
            ),
            continuous_features
        )
    ]

)



# ===============================
# 7. 建立随机森林模型
# ===============================


rf = RandomForestClassifier(
    random_state=42
)



# ===============================
# 8. 构建Pipeline
# ===============================

# 注意：
# SMOTE必须放在编码之后、模型之前

pipeline = ImbPipeline(

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



# ===============================
# 9. 超参数设置
# ===============================


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



# ===============================
# 10. GridSearchCV
# 不使用多进程
# ===============================


grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="roc_auc",

    cv=5,

    n_jobs=1,

    verbose=1

)



print("\n开始模型训练...")

grid_search.fit(
    X_train,
    y_train
)



print("\n最佳参数：")
print(grid_search.best_params_)



print("\n最佳交叉验证AUC：")
print(grid_search.best_score_)



# ===============================
# 11. 模型预测
# ===============================


best_model = grid_search.best_estimator_


y_pred = best_model.predict(
    X_test
)


# ROC-AUC需要概率
y_prob = best_model.predict_proba(
    X_test
)[:, 1]



# ===============================
# 12. 模型评价
# ===============================


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
print("随机森林模型评价结果")
print("==============================")

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


print("\n模型训练完成！")