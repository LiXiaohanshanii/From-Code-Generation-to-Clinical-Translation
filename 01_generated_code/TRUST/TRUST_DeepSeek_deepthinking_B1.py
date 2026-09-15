# -*- coding: utf-8 -*-
"""
随机森林分类模型：预测TRUST滴度是否≥16
数据文件：train_data.csv (UTF-8编码)
环境：PyCharm 2025.2.3, Python 3.9+
依赖库：pandas, numpy, scikit-learn, imbalanced-learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 设置随机种子，保证结果可复现
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ========================== 1. 读取数据 ==========================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# ========================== 2. 分离特征与目标 ==========================
# 目标列为最后一列 'TRUST'
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 创建二分类标签：TRUST >= 16 为正类(1)，否则为负类(0)
y_binary = (y >= 16).astype(int)

# ========================== 3. 定义特征列名 ==========================
# 分类变量（独热编码）
categorical_cols = ['SEX', 'DEPT', 'DIAGNOSIS']   # 注意列名：DIAGNOSIS
# 分类变量（序数编码）
ordinal_col = 'TPPA'
# 连续变量（中位数填充）
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ========================== 4. 预处理流水线 ==========================
# 4.1 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 4.2 分类变量（独热）：OneHotEncoder，处理未知类别
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 4.3 分类变量（序数）：OrdinalEncoder，按字母顺序编码
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('ord', ordinal_transformer, [ordinal_col])
    ],
    remainder='drop'  # 确保只处理指定列
)

# ========================== 5. 划分训练集和测试集 ==========================
# 按目标分层划分，保证正负样本比例在训练/测试中一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, stratify=y_binary, random_state=RANDOM_STATE
)

# ========================== 6. 对训练集进行预处理 ==========================
# 先拟合预处理器并转换训练集
X_train_processed = preprocessor.fit_transform(X_train)
# 转换测试集（使用训练集拟合的变换器）
X_test_processed = preprocessor.transform(X_test)

# 获取特征名称（可选，用于调试）
# feature_names = (preprocessor.named_transformers_['cont'].get_feature_names_out(continuous_cols).tolist() +
#                  preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist() +
#                  [ordinal_col])
# print(f"预处理后特征数量: {X_train_processed.shape[1]}")

# ========================== 7. SMOTE处理类别不平衡 ==========================
# 仅在训练集上应用SMOTE，生成合成样本
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# ========================== 8. 定义随机森林模型及超参数网格 ==========================
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)  # 单进程，不允许多线程

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

# ========================== 9. 网格搜索超参数调优 ==========================
# 使用5折交叉验证，评估指标为roc_auc（可改为'accuracy'等，但AUC对不平衡更稳健）
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',      # 以AUC作为优化目标
    n_jobs=1,               # 单进程，符合要求
    verbose=1               # 输出进度信息
)

grid_search.fit(X_train_resampled, y_train_resampled)

# 输出最佳参数组合
print("最佳超参数组合:", grid_search.best_params_)

# 使用最佳模型
best_rf = grid_search.best_estimator_

# ========================== 10. 在测试集上评估模型 ==========================
y_pred = best_rf.predict(X_test_processed)
y_pred_proba = best_rf.predict_proba(X_test_processed)[:, 1]  # 正类概率

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n=== 模型评估结果 ===")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:               {auc:.4f}")