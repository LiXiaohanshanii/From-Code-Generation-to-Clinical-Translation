# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 隐球菌荚膜抗原(CrAg)滴度预测
目标：预测CSF-T是否≥20（二分类）
环境：PyCharm 2025.2.3 / Python 3.x
依赖：pandas, numpy, scikit-learn, imbalanced-learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("步骤 1: 加载数据")
print("=" * 70)

df = pd.read_csv('CrAg_train.csv', encoding='utf-8')
print(f"  数据集形状: {df.shape}")
print(f"  列名: {df.columns.tolist()}")

# ============================================================
# 2. 特征与目标分离
# ============================================================
# 目标列 CSF-T 是最后一列
X = df.iloc[:, :-1]          # 特征矩阵
y_raw = df.iloc[:, -1]       # 原始目标 (滴度值)

# 二分类转换: ≥20 → 1, <20 → 0
y = (y_raw >= 20).astype(int)
print(f"\n  目标列(CSF-T≥20) 分布:")
print(f"    类别 0 (<20): {(y == 0).sum()}")
print(f"    类别 1 (≥20): {(y == 1).sum()}")

# ============================================================
# 3. 定义变量类型
# ============================================================
# 连续变量
continuous_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 独热编码变量
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

# 序数编码变量
ordinal_features = ['SER-T', 'Ink staining']

print(f"\n  连续变量:     {continuous_features}")
print(f"  独热编码变量: {onehot_features}")
print(f"  序数编码变量: {ordinal_features}")

# ============================================================
# 4. 划分训练集 / 验证集 (4:1)
# ============================================================
print("\n" + "=" * 70)
print("步骤 2: 划分训练集与验证集 (4:1)")
print("=" * 70)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # 保持类别比例
)
print(f"  训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本")

# ============================================================
# 5. 构建预处理流水线
#    顺序：数值(中位数填充) → 独热 → 序数
# ============================================================
print("\n" + "=" * 70)
print("步骤 3: 构建预处理流水线")
print("=" * 70)

# 5.1 连续变量: 中位数填充
cont_transformer = SimpleImputer(strategy='median')

# 5.2 独热编码
ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 5.3 序数编码
ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 5.4 组合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', cont_transformer, continuous_features),   # ① 数值
        ('ohe', ohe_transformer, onehot_features),       # ② 独热
        ('ord', ord_transformer, ordinal_features),      # ③ 序数
    ],
    remainder='drop'
)

print("  流水线组件:")
print("    [数值] → SimpleImputer(median)")
print("    [独热] → OneHotEncoder(ignore)")
print("    [序数] → OrdinalEncoder(unknown=-1)")

# ============================================================
# 6. 构建完整 Pipeline (SMOTE 嵌入交叉验证内部，防泄露)
# ============================================================
print("\n" + "=" * 70)
print("步骤 4: 构建带 SMOTE 的完整 Pipeline")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])
print("  Pipeline: preprocessor → SMOTE → RandomForest")

# ============================================================
# 7. 超参数调优 (GridSearchCV + 5折分层交叉验证)
# ============================================================
print("\n" + "=" * 70)
print("步骤 5: 超参数调优 (GridSearchCV)")
print("=" * 70)

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=1,               # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

print(f"\n  最佳参数: {grid_search.best_params_}")
print(f"  最佳交叉验证 F1: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ============================================================
# 8. 在验证集上评估
# ============================================================
print("\n" + "=" * 70)
print("步骤 6: 验证集评估")
print("=" * 70)

y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
pre = precision_score(y_val, y_pred)
f1  = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print(f"\n  准确率 (Accuracy):  {acc:.4f}")
print(f"  召回率 (Recall):    {rec:.4f}")
print(f"  精确率 (Precision): {pre:.4f}")
print(f"  F1 分数 (F1-score): {f1:.4f}")
print(f"  AUC:                {auc:.4f}")

print("\n" + "=" * 70)
print("模型构建与评估完成")
print("=" * 70)