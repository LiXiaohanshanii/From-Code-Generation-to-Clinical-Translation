import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================
# 1. 读取数据
# ==========================

data = pd.read_csv("CrAg_train.csv", encoding="utf-8")

# ==========================
# 2. 构建二分类标签
#    CSF-T >=20 为1
#    CSF-T <20 为0
# ==========================

data["CSF_binary"] = (data["CSF-T"] >= 20).astype(int)

# ==========================
# 3. 特征与标签
# ==========================

X = data.drop(columns=["CSF-T", "CSF_binary"])
y = data["CSF_binary"]

# ==========================
# 4. 特征分类
# ==========================

numeric_features = [
    "CL",
    "GLU",
    "Protein",
    "RBC",
    "AGE"
]

onehot_features = [
    "Color",
    "Transparency",
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]

ordinal_features = [
    "SER-T",
    "Ink staining"
]

# ==========================
# 5. 各类型预处理
# ==========================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

onehot_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

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

# ==========================
# 6. ColumnTransformer
#    顺序：数值→独热→序数
# ==========================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("onehot", onehot_transformer, onehot_features),
        ("ordinal", ordinal_transformer, ordinal_features)
    ]
)

# ==========================
# 7. 划分训练集与验证集
# ==========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================
# 8. 构建Pipeline
#    预处理→SMOTE→随机森林
#    （避免数据泄露）
# ==========================

pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)

# ==========================
# 9. 超参数
# ==========================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1],
    "classifier__class_weight": ["balanced"]
}

# ==========================
# 10. 五折交叉验证
# ==========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=1,          # 不使用多进程
    verbose=1,
    refit=True
)

# ==========================
# 11. 模型训练
# ==========================

grid_search.fit(X_train, y_train)

print("=" * 60)
print("Best Parameters:")
print(grid_search.best_params_)

print("=" * 60)
print("Best CV F1:")
print(grid_search.best_score_)

# ==========================
# 12. 验证集预测
# ==========================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_valid)

y_prob = best_model.predict_proba(X_valid)[:, 1]

# ==========================
# 13. 模型评估
# ==========================

accuracy = accuracy_score(y_valid, y_pred)

precision = precision_score(y_valid, y_pred)

recall = recall_score(y_valid, y_pred)

f1 = f1_score(y_valid, y_pred)

auc = roc_auc_score(y_valid, y_prob)

print("=" * 60)
print("Validation Performance")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")