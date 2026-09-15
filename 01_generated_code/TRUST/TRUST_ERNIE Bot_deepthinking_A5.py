# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 完整版
环境：PyCharm 2025.2.3 / Python 3.x
目标：预测 TRUST 是否 >= 16（二分类）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ============================================================
# 1. 加载数据
# ============================================================
print("正在加载数据...")
df = pd.read_csv('train_data.csv', encoding='utf-8')

print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# ============================================================
# 2. 构建目标变量（二分类：TRUST >= 16）
# ============================================================
df['TARGET'] = (df['TRUST'] >= 16).astype(int)

print(f"\n目标变量分布:\n{df['TARGET'].value_counts()}")
print(f"  0 (<16): {(df['TARGET'] == 0).sum()}")
print(f"  1 (>=16): {(df['TARGET'] == 1).sum()}")

# ============================================================
# 3. 分离特征和目标
# ============================================================
# 特征列（不包含 TRUST 和 TARGET）
feature_columns = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA',
                   'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[feature_columns].copy()
y = df['TARGET'].copy()

# ============================================================
# 4. 定义特征类型
# ============================================================
categorical_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ordinal = ['TPPA']                    # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# ============================================================
# 5. 划分训练集和测试集（分层抽样）
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ============================================================
# 6. 构建预处理管道
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        # 独热编码：SEX, DEPT, DIAGNOSIS
        ('ohe',
         OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
         categorical_ohe),

        # 序数编码：TPPA
        ('ordinal',
         OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         categorical_ordinal),

        # 中位数填充连续变量
        ('continuous',
         SimpleImputer(strategy='median'),
         continuous)
    ],
    remainder='drop'
)

# ============================================================
# 7. 构建完整 Pipeline（预处理 + SMOTE + 随机森林）
# ============================================================
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ============================================================
# 8. 定义超参数搜索空间
# ============================================================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

print("\n开始超参数调优（GridSearchCV，n_jobs=1）...")

# ============================================================
# 9. 网格搜索（不使用多进程）
# ============================================================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,          # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

# ============================================================
# 10. 获取最优模型
# ============================================================
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\n" + "=" * 60)
print("最优超参数组合:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print("=" * 60)

# ============================================================
# 11. 在测试集上预测
# ============================================================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# ============================================================
# 12. 模型评估
# ============================================================
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("随机森林分类模型 - 测试集评估结果")
print("=" * 60)
print(f"  准确率 (Accuracy):  {accuracy:.4f}")
print(f"  召回率 (Recall):    {recall:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  F1分数 (F1-score):  {f1:.4f}")
print(f"  AUC:                {auc:.4f}")
print("=" * 60)

# 额外输出分类报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=['<16', '>=16']))

print("\n程序执行完毕。")
