import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

# ==========================
# 1. 读取数据
# ==========================
df = pd.read_csv("CrAg_train.csv", encoding="utf-8")

# ==========================
# 2. 构建二分类标签
#    CSF-T >=20 为1
#    CSF-T <20 为0
# ==========================
target_col = "CSF-T"

df[target_col] = (df[target_col] >= 20).astype(int)

# ==========================
# 3. 特征和标签
# ==========================
X = df.drop(columns=[target_col])
y = df[target_col]

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
# 5. 数值预处理
# ==========================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# ==========================
# 6. One-Hot编码
# ==========================
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

# ==========================
# 7. Ordinal编码
# 未指定类别顺序，因此自动学习
# ==========================
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
# 8. ColumnTransformer
# 顺序：
# 数值 -> OneHot -> Ordinal
# ==========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("onehot", onehot_transformer, onehot_features),
        ("ordinal", ordinal_transformer, ordinal_features)
    ],
    remainder="drop"
)

# ==========================
# 9. 划分训练集和验证集
# ==========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# 10. 建立Pipeline
#     注意：
#     SMOTE放在CV内部，避免数据泄露
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
# 11. 参数网格
# ==========================
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1],
    "classifier__class_weight": ["balanced"]
}

# ==========================
# 12. 五折交叉验证
# ==========================
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ==========================
# 13. GridSearchCV
# ==========================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=1,
    verbose=1,
    refit=True
)

# ==========================
# 14. 训练模型
# ==========================
grid_search.fit(X_train, y_train)

print("=" * 60)
print("Best Parameters:")
print(grid_search.best_params_)

print("=" * 60)
print("Best CV F1:")
print(grid_search.best_score_)

# ==========================
# 15. 在验证集预测
# ==========================
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_valid)

y_prob = best_model.predict_proba(X_valid)[:, 1]

# ==========================
# 16. 计算指标
# ==========================
accuracy = accuracy_score(y_valid, y_pred)

recall = recall_score(y_valid, y_pred)

precision = precision_score(y_valid, y_pred)

f1 = f1_score(y_valid, y_pred)

auc = roc_auc_score(y_valid, y_prob)

print("=" * 60)
print("Validation Performance")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")