# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 预测TRUST是否≥16
适用于PyCharm 2025.2.3环境
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# ====================== 1. 数据加载 ======================
print("=" * 60)
print("步骤1: 加载数据")
print("=" * 60)

df = pd.read_csv('train_data.csv', encoding='utf-8')
print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# ====================== 2. 数据预处理 ======================
print("\n" + "=" * 60)
print("步骤2: 数据预处理")
print("=" * 60)

# 分离特征和目标变量
X = df.iloc[:, :-1]  # 所有特征列
y = df.iloc[:, -1]   # 最后一列TRUST为目标变量

# 将目标变量转换为二分类：TRUST >= 16 为1，否则为0
y_binary = (y >= 16).astype(int)
print(f"原始目标变量分布:\n{y.value_counts().sort_index()}")
print(f"二分类目标变量分布:\n{y_binary.value_counts()}")

# 定义分类变量和连续变量
categorical_features_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_feature_ordinal = ['TPPA']                    # 序数编码
continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 验证特征列
all_features = categorical_features_ohe + categorical_feature_ordinal + continuous_features
print(f"\n特征列验证: {all_features}")
print(f"特征列数量: {len(all_features)}")

# ====================== 3. 构建预处理管道 ======================
print("\n" + "=" * 60)
print("步骤3: 构建预处理管道")
print("=" * 60)

# 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量(独热编码)
categorical_ohe_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# 分类变量(序数编码)
categorical_ordinal_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_features),
        ('cat_ohe', categorical_ohe_transformer, categorical_features_ohe),
        ('cat_ord', categorical_ordinal_transformer, categorical_feature_ordinal)
    ],
    remainder='drop'  # 丢弃其他未指定的列
)

print("预处理管道构建完成")

# ====================== 4. 划分训练集和测试集 ======================
print("\n" + "=" * 60)
print("步骤4: 划分训练集和测试集")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"训练集正类比例: {y_train.mean():.4f}")
print(f"测试集正类比例: {y_test.mean():.4f}")

# ====================== 5. 构建完整管道（含SMOTE和模型） ======================
print("\n" + "=" * 60)
print("步骤5: 构建完整管道（预处理 + SMOTE + 随机森林）")
print("=" * 60)

# 定义随机森林分类器
rf_classifier = RandomForestClassifier(
    random_state=42,
    n_jobs=1  # 不使用多进程
)

# 使用imblearn的Pipeline，将SMOTE放在模型训练之前
full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_classifier)
])

print("完整管道构建完成")

# ====================== 6. 超参数调优 ======================
print("\n" + "=" * 60)
print("步骤6: 网格搜索超参数调优")
print("=" * 60)

# 定义参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# GridSearchCV，n_jobs=1不使用多进程
grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,
    verbose=1,
    refit=True
)

# 拟合模型
print("\n开始网格搜索...")
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("\n" + "=" * 60)
print("最佳超参数:")
print("=" * 60)
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\n最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ====================== 7. 模型评估 ======================
print("\n" + "=" * 60)
print("步骤7: 模型评估")
print("=" * 60)

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("模型评估结果:")
print("=" * 60)
print(f"  准确率 (Accuracy):  {accuracy:.4f}")
print(f"  召回率 (Recall):    {recall:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  F1分数 (F1-score):  {f1:.4f}")
print(f"  AUC:                {auc:.4f}")
print("=" * 60)

# ====================== 8. 额外信息 ======================
print("\n" + "=" * 60)
print("附加信息:")
print("=" * 60)
print(f"SMOTE后训练集正类样本数: {sum(y_train)} -> SMOTE后: "
      f"{sum(grid_search.best_estimator_.named_steps['smote'].fit_resample("
      f"grid_search.best_estimator_.named_steps['preprocessor'].transform(X_train), y_train))}")

# 查看预处理后的特征数量
X_train_transformed = grid_search.best_estimator_.named_steps['preprocessor'].transform(X_train)
print(f"预处理后特征维度: {X_train_transformed.shape[1]}")

# SMOTE后的训练集大小
X_resampled, y_resampled = grid_search.best_estimator_.named_steps['smote'].fit_resample(
    X_train_transformed, y_train
)
print(f"SMOTE后训练集大小: {X_resampled.shape[0]}")
print(f"SMOTE后正类样本数: {sum(y_resampled)}")
print(f"SMOTE后负类样本数: {sum(1 - y_resampled)}")

print("\n" + "=" * 60)
print("模型构建与评估完成！")
print("=" * 60)
