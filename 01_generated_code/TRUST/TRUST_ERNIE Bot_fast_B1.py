# -*- coding: utf-8 -*-
"""
随机森林分类模型构建与评估
环境：PyCharm 2025.2.3，Python 3.x
数据集：train_data.csv，编码utf-8
任务：预测TRUST是否>=16（二分类）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score,
                             roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# ========================== 1. 数据加载 ==========================
print("=" * 60)
print("步骤1：加载数据")
df = pd.read_csv('train_data.csv', encoding='utf-8')
print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print("=" * 60)

# ========================== 2. 特征与目标分离 ==========================
# 特征列（除最后一列TRUST外）
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[feature_cols].copy()
y = df.iloc[:, -1].copy()  # 最后一列 TRUST

# 目标变量二值化：>=16 为 1，否则为 0
y_binary = (y >= 16).astype(int)
print(f"\n目标变量分布:\n{y_binary.value_counts()}")
print(f"原始目标变量示例: {y.unique()[:10]}")

# ========================== 3. 定义列类型 ==========================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal_cols = ['TPPA']                     # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT',
                    'NC', 'LY', 'NLR']                  # 连续变量

# ========================== 4. 构建预处理Pipeline ==========================
# 4.1 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 4.2 分类变量（独热编码）：无缺失值，直接编码
categorical_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 4.3 分类变量（序数编码）：无缺失值，直接编码
categorical_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value',
                                                  unknown_value=-1)

# 4.4 ColumnTransformer 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot_cols),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'  # 未列出的列丢弃
)

# ========================== 5. 构建完整Pipeline（含SMOTE和模型） ==========================
# 注意：SMOTE不支持在Pipeline内部使用（需要单独处理），
# 因此先用preprocessor处理数据，再做SMOTE，最后训练模型。

print("\n步骤2：数据预处理（填充 + 编码）...")
X_processed = preprocessor.fit_transform(X)

# 获取编码后的特征名（用于查看）
onehot_feature_names = preprocessor.named_transformers_['cat_onehot'].get_feature_names_out(
    categorical_onehot_cols)
all_feature_names = np.concatenate(
    [continuous_cols, onehot_feature_names, categorical_ordinal_cols]
)
print(f"预处理后特征维度: {X_processed.shape}")

# ========================== 6. 数据集划分 ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ========================== 7. SMOTE处理分类不平衡 ==========================
print("\n步骤3：使用SMOTE处理训练集不平衡...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE前训练集类别分布:\n{pd.Series(y_train).value_counts()}")
print(f"SMOTE后训练集类别分布:\n{pd.Series(y_train_smote).value_counts()}")

# ========================== 8. 随机森林模型 + 超参数调优 ==========================
print("\n步骤4：随机森林超参数调优（GridSearchCV）...")

rf_clf = RandomForestClassifier(
    n_jobs=1,           # 不使用多进程
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth':[10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,           # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train_smote, y_train_smote)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ========================== 9. 模型评估 ==========================
print("\n步骤5：在测试集上评估模型...")
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 60)
print("模型评估结果（测试集）")
print("=" * 60)
print(f"  准确率 (Accuracy)  : {accuracy:.4f}")
print(f"  召回率 (Recall)     : {recall:.4f}")
print(f"  精确率 (Precision)  : {precision:.4f}")
print(f"  F1分数 (F1-score)   : {f1:.4f}")
print(f"  AUC曲线下面积       : {auc:.4f}")
print("=" * 60)

# ========================== 10. 特征重要性输出 ==========================
feature_importance = best_model.feature_importances_
importance_df = pd.DataFrame({
    '特征': all_feature_names,
    '重要性': feature_importance
}).sort_values(by='重要性', ascending=False)

print("\n特征重要性排名（前20）:")
print(importance_df.head(20).to_string(index=False))
print("=" * 60)

# ========================== 11. 保存模型（可选） ==========================
import joblib
joblib.dump(best_model, 'rf_model_best.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("\n模型已保存为 rf_model_best.pkl")
print("预处理器已保存为 preprocessor.pkl")
print("=" * 60)
print("程序执行完毕！")
