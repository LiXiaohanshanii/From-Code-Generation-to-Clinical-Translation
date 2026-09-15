# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------- 1. 读取数据 --------------------
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列是最后一列（TRUST）
X = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]

# -------------------- 2. 划分训练集和测试集（分层） --------------------
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# -------------------- 3. 目标值二分类（是否 >= 16） --------------------
y_train = (y_train_raw >= 16).astype(int)
y_test = (y_test_raw >= 16).astype(int)

# -------------------- 4. 定义特征列类型 --------------------
categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_cols_ordinal = ['TPPA']                    # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 验证列名存在性（防止拼写错误）
for col in categorical_cols_onehot + categorical_cols_ordinal + continuous_cols:
    if col not in X_train.columns:
        raise ValueError(f"列 '{col}' 不存在于数据中，请检查原始列名。")

# -------------------- 5. 构建预处理流水线 --------------------
# 5.1 连续变量：中位数填充（使用训练集拟合）
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 5.2 分类变量独热编码（处理未知类别）
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 5.3 分类变量序数编码（TPPA）
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_cols),
        ('cat_onehot', onehot_transformer, categorical_cols_onehot),
        ('cat_ordinal', ordinal_transformer, categorical_cols_ordinal)
    ],
    remainder='drop'  # 仅保留指定列
)

# -------------------- 6. 在训练集上拟合预处理器，并转换训练/测试集 --------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 获取特征名称（便于理解，非必须）
# onehot 特征名
onehot_feature_names = []
for col in categorical_cols_onehot:
    cats = preprocessor.named_transformers_['cat_onehot'].named_steps['onehot'].categories_[0]
    onehot_feature_names.extend([f"{col}_{cat}" for cat in cats])
feature_names = continuous_cols + onehot_feature_names + ['TPPA_encoded']
# 将处理后的数组转为 DataFrame（便于查看，但不影响后续）
X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

# -------------------- 7. 应用 SMOTE 解决类别不平衡（仅训练集） --------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# 输出重采样后的类别分布（可选）
print("SMOTE 重采样后训练集类别分布:")
print(pd.Series(y_train_resampled).value_counts())

# -------------------- 8. 随机森林超参数调优（GridSearchCV，单进程） --------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc',          # 以 AUC 作为择优指标
    cv=5,
    n_jobs=1,                   # 禁用多进程
    verbose=1
)
grid_search.fit(X_train_resampled, y_train_resampled)

best_rf = grid_search.best_estimator_
print("\n最佳超参数组合:", grid_search.best_params_)

# -------------------- 9. 在测试集上评估模型 --------------------
y_pred = best_rf.predict(X_test_processed)
y_pred_proba = best_rf.predict_proba(X_test_processed)[:, 1]  # 正类概率

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 打印结果
print("\n===== 测试集评估结果 =====")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1 分数 (F1-score): {f1:.4f}")
print(f"AUC 分数 (ROC-AUC): {auc:.4f}")

# （可选）输出分类报告
from sklearn.metrics import classification_report
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['<16', '>=16']))