# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 预测TRUST是否≥16
适用环境：PyCharm 2025.2.3
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ============================================================
# 1. 数据加载
# ============================================================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 提取特征列和目标列
feature_columns = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                   'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
target_column = 'TRUST'

X = data[feature_columns]
y = data[target_column]

# ============================================================
# 2. 目标变量转换为二分类（≥16 为 1，<16 为 0）
# ============================================================
y = (y >= 16).astype(int)

print(f"目标变量分布：\n{y.value_counts()}")
print(f"正类(≥16)比例：{y.mean():.4f}")

# ============================================================
# 3. 特征类型划分
# ============================================================
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                       # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ============================================================
# 4. 数据分割（先分割再处理，避免数据泄露）
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 5. 分类变量编码（在训练集上fit，在训练集和测试集上transform）
# ============================================================
# 5.1 独热编码
oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_oh = oh_encoder.fit_transform(X_train[categorical_onehot])
X_test_oh = oh_encoder.transform(X_test[categorical_onehot])

# 5.2 序数编码
ord_encoder = OrdinalEncoder()
X_train_ord = ord_encoder.fit_transform(X_train[categorical_ordinal])
X_test_ord = ord_encoder.transform(X_test[categorical_ordinal])

# ============================================================
# 6. 连续变量缺失值处理（中位数填充）
# ============================================================
med_imputer = SimpleImputer(strategy='median')
X_train_cont = med_imputer.fit_transform(X_train[continuous])
X_test_cont = med_imputer.transform(X_test[continuous])

# ============================================================
# 7. 合并所有特征
# ============================================================
X_train_processed = np.hstack([
    X_train_oh,
    X_train_ord,
    X_train_cont
])
X_test_processed = np.hstack([
    X_test_oh,
    X_test_ord,
    X_test_cont
])

# 获取独热编码后的特征名称（便于查看）
oh_feature_names = oh_encoder.get_feature_names_out(categorical_onehot)
all_feature_names = np.concatenate([oh_feature_names, categorical_ordinal, continuous])
print(f"\n处理后特征维度：{X_train_processed.shape}")
print(f"特征数量：{X_train_processed.shape[1]}")

# ============================================================
# 8. SMOTE处理分类不平衡（仅在训练集上进行）
# ============================================================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"\nSMOTE后训练集类别分布：\n{pd.Series(y_train_smote).value_counts()}")

# ============================================================
# 9. 随机森林模型构建与超参数调优
# ============================================================
rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=1  # 不使用多进程
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,  # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train_smote, y_train_smote)

print(f"\n最佳参数：{grid_search.best_params_}")
print(f"最佳交叉验证AUC：{grid_search.best_score_:.4f}")

# ============================================================
# 10. 使用最佳模型在测试集上评估
# ============================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print("模型评估结果（测试集）")
print("="*50)
print(f"准确率 (Accuracy)：  {accuracy:.4f}")
print(f"召回率 (Recall)：    {recall:.4f}")
print(f"精确率 (Precision)： {precision:.4f}")
print(f"F1分数 (F1-score)：  {f1:.4f}")
print(f"AUC：               {auc:.4f}")
print("="*50)
