import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================
# 1. 读取数据
# ==========================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# ==========================
# 2. 特征与目标变量定义
# ==========================
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA',
                'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
target_col = 'TRUST'

X = df[feature_cols].copy()
y = df[target_col].copy()

# ==========================
# 3. 目标变量转换为二分类（≥16 为1，否则为0）
# ==========================
y = (y >= 16).astype(int)

# ==========================
# 4. 定义分类变量和连续变量
# ==========================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal_cols = ['TPPA']
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ==========================
# 5. 构建预处理管道
# ==========================
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量（独热编码）
categorical_onehot_transformer = OneHotEncoder(
    drop='first', sparse_output=False, handle_unknown='ignore'
)

# TPPA序数编码
categorical_ordinal_transformer = OrdinalEncoder(
    handle_unknown='use_encoded_value', unknown_value=-1
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_cols),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot_cols),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'
)

# ==========================
# 6. 划分训练集和测试集
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# 7. 构建完整管道（预处理 + SMOTE + 随机森林）
# ==========================
rf = RandomForestClassifier(random_state=42, n_jobs=1)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])

# ==========================
# 8. 超参数调优（GridSearchCV）
# ==========================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ==========================
# 9. 最佳模型与参数
# ==========================
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\n" + "=" * 50)
print("最佳超参数：")
print("=" * 50)
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ==========================
# 10. 模型预测与评估
# ==========================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("模型评估结果（测试集）")
print("=" * 50)
print(f"  准确率 (Accuracy)    : {accuracy:.4f}")
print(f"  召回率 (Recall)       : {recall:.4f}")
print(f"  精确率 (Precision)    : {precision:.4f}")
print(f"  F1分数 (F1-score)     : {f1:.4f}")
print(f"  AUC曲线下面积 (AUC)   : {auc:.4f}")
print("=" * 50)
