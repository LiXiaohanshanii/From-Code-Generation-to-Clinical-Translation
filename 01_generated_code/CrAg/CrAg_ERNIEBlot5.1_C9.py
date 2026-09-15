# -*- coding: utf-8 -*-
"""
随机森林分类模型 - CrAg_train 数据集
预测 CSF-T 是否 ≥ 20（二分类）
防止数据泄露：使用 imblearn.pipeline.Pipeline 将 SMOTE 嵌入交叉验证内部
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 特征列和目标列
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols]
y = df[target_col]

# ============================================================
# 2. 二分类目标：CSF-T >= 20 为 1，否则为 0
# ============================================================
y = (y >= 20).astype(int)

# ============================================================
# 3. 划分训练集和验证集（4:1）
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 4. 定义变量类型
# ============================================================
# 连续变量
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 独热编码变量
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

# 序数编码变量
ordinal_cols = ['SER-T', 'Ink staining']

# ============================================================
# 5. 构建预处理管道（顺序：数值→独热→序数）
# ============================================================
# (1) 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# (2) 独热编码变量
onehot_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# (3) 序数编码变量
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', continuous_transformer, continuous_cols),
        ('onehot', onehot_transformer, onehot_cols),
        ('ordinal', ordinal_transformer, ordinal_cols)
    ],
    remainder='drop'  # 其他列（如果有）丢弃
)

# ============================================================
# 6. 构建完整的 ImbPipeline（防止数据泄露）
#    顺序：预处理 → SMOTE → 随机森林
# ============================================================
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
])

# ============================================================
# 7. 超参数调优（GridSearchCV，仅在训练集上交叉验证）
# ============================================================
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
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 最佳模型
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\n" + "=" * 60)
print("最佳超参数：")
print(best_params)
print("=" * 60)

# ============================================================
# 8. 在验证集上评估
# ============================================================
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
pre = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print("\n" + "=" * 60)
print("验证集评估结果：")
print(f"  准确率 (Accuracy):  {acc:.4f}")
print(f"  召回率 (Recall):    {rec:.4f}")
print(f"  精确率 (Precision): {pre:.4f}")
print(f"  F1 分数 (F1-score): {f1:.4f}")
print(f"  AUC:               {auc:.4f}")
print("=" * 60)

# ============================================================
# 9. 输出样本数量信息（可选参考）
# ============================================================
print(f"\n训练集样本数: {len(y_train)}, 验证集样本数: {len(y_val)}")
print(f"训练集正样本比例: {y_train.mean():.2%}, 验证集正样本比例: {y_val.mean():.2%}")