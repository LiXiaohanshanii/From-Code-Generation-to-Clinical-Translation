# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 预测TRUST滴度是否≥16
适用环境：PyCharm 2025.2.3
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ======================== 1. 数据加载 ========================
print("=" * 60)
print("步骤1：加载数据")
print("=" * 60)

df = pd.read_csv('train_data.csv', encoding='utf-8')

print(f"数据集形状: {df.shape}")
print(f"列名列表: {df.columns.tolist()}")

# 分离特征和目标变量（目标列为最后一列TRUST）
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"\n特征列数: {X.shape[1]}")
print(f"样本总数: {X.shape[0]}")
print(f"\n目标变量TRUST原始分布:")
print(y.value_counts().sort_index())

# ======================== 2. 目标变量二分类转换 ========================
print("\n" + "=" * 60)
print("步骤2：目标变量转换为二分类（≥16 为1，<16 为0）")
print("=" * 60)

y_binary = (y >= 16).astype(int)
print(f"二分类目标分布:")
print(f"  0（<16）: {(y_binary == 0).sum()} 例")
print(f"  1（≥16）: {(y_binary == 1).sum()} 例")

# ======================== 3. 定义特征列类型 ========================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal_cols = ['TPPA']                       # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

print(f"\n独热编码特征: {categorical_onehot_cols}")
print(f"序数编码特征: {categorical_ordinal_cols}")
print(f"连续变量特征: {continuous_cols}")

# ======================== 4. 构建预处理管道 ========================
print("\n" + "=" * 60)
print("步骤3：构建预处理管道（ColumnTransformer）")
print("=" * 60)

# 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量（独热编码）：无缺失值，直接编码
categorical_onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 分类变量（序数编码）：无缺失值，直接编码
categorical_ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot_cols),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'
)

print("预处理管道构建完成。")

# ======================== 5. 构建完整Pipeline ========================
print("\n" + "=" * 60)
print("步骤4：构建完整建模管道（预处理 + SMOTE + 随机森林）")
print("=" * 60)

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("完整管道构建完成。")

# ======================== 6. 划分训练集与测试集 ========================
print("\n" + "=" * 60)
print("步骤5：划分训练集和测试集（8:2分层抽样）")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"训练集正类比例: {(y_train == 1).mean():.2%}")
print(f"测试集正类比例: {(y_test == 1).mean():.2%}")

# ======================== 7. 超参数调优 ========================
print("\n" + "=" * 60)
print("步骤6：GridSearchCV 超参数调优（n_jobs=1，不使用多进程）")
print("=" * 60)

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1,
    refit=True
)

print("开始网格搜索...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数组合:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"最佳交叉验证 AUC: {grid_search.best_score_:.4f}")

# ======================== 8. 模型评估 ========================
print("\n" + "=" * 60)
print("步骤7：在测试集上评估最佳模型")
print("=" * 60)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'指标':<20} {'结果':>10}")
print("-" * 32)
print(f"{'准确率 (Accuracy)':<20} {accuracy:>10.4f}")
print(f"{'召回率 (Recall)':<20} {recall:>10.4f}")
print(f"{'精确率 (Precision)':<20} {precision:>10.4f}")
print(f"{'F1分数 (F1-score)':<20} {f1:>10.4f}")
print(f"{'AUC':<20} {auc:>10.4f}")

# ======================== 9. 结果汇总 ========================
print("\n" + "=" * 60)
print("模型构建与评估完成！")
print("=" * 60)
