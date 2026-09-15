# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 检验科数据分析
适用环境：PyCharm 2025.2.3
任务：预测TRUST滴度是否≥16（二分类）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 60)
print("【步骤1】加载数据集 train_data.csv")
print("=" * 60)

df = pd.read_csv('train_data.csv', encoding='utf-8')
print(f"数据集维度: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"特征列: {df.columns[:-1].tolist()}")
print(f"目标列: {df.columns[-1]}")
print("-" * 60)

# ============================================================
# 2. 特征与目标分离
# ============================================================
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[feature_cols].copy()
y = df.iloc[:, -1].copy()          # TRUST列（最后一列）

# 二分类目标：TRUST ≥ 16 → 1，否则 → 0
y_binary = (y >= 16).astype(int)

print(f"\n目标变量原始值示例: {y.unique()[:10]}")
print(f"二分类目标分布:\n{y_binary.value_counts().to_string()}")
print("-" * 60)

# ============================================================
# 3. 定义列类型
# ============================================================
# 独热编码的分类变量
cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
# 序数编码的分类变量
cat_ordinal_cols = ['TPPA']
# 连续变量
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT',
                   'NC', 'LY', 'NLR']

# ============================================================
# 4. 构建预处理Pipeline
# ============================================================
# 连续变量：中位数填充
cont_imputer = SimpleImputer(strategy='median')

# 独热编码（分类变量无缺失值）
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 序数编码（分类变量无缺失值）
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', cont_imputer, continuous_cols),
        ('ohe', ohe, cat_onehot_cols),
        ('ord', ord_enc, cat_ordinal_cols)
    ],
    remainder='drop'
)

print("\n【步骤2】执行数据预处理（缺失值填充 + 编码）...")
X_processed = preprocessor.fit_transform(X)

# 获取编码后的特征名称
ohe_names = preprocessor.named_transformers_['ohe'].get_feature_names_out(cat_onehot_cols)
all_feature_names = np.concatenate([continuous_cols, ohe_names.tolist(), cat_ordinal_cols])

print(f"预处理后特征矩阵维度: {X_processed.shape}")
print("-" * 60)

# ============================================================
# 5. 划分训练集与测试集
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

print(f"\n【步骤3】数据集划分:")
print(f"  训练集: {X_train.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")
print(f"  训练集类别分布: {pd.Series(y_train).value_counts().to_dict()}")
print("-" * 60)

# ============================================================
# 6. SMOTE处理分类不平衡
# ============================================================
print("\n【步骤4】使用SMOTE处理训练集不平衡...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"  SMOTE前: {pd.Series(y_train).value_counts().to_dict()}")
print(f"  SMOTE后: {pd.Series(y_train_smote).value_counts().to_dict()}")
print("-" * 60)

# ============================================================
# 7. 随机森林 + GridSearchCV 超参数调优
# ============================================================
print("\n【步骤5】随机森林超参数调优（GridSearchCV）...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    n_jobs=1,              # 不使用多进程
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,              # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train_smote, y_train_smote)

best_model = grid_search.best_estimator_
print(f"\n最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
print("-" * 60)

# ============================================================
# 8. 模型评估（测试集）
# ============================================================
print("\n【步骤6】模型评估（测试集）...")

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 60)
print("           模 型 评 估 结 果（测试集）")
print("=" * 60)
print(f"  准确率 (Accuracy)  : {acc:.4f}")
print(f"  召回率 (Recall)     : {rec:.4f}")
print(f"  精确率 (Precision)  : {pre:.4f}")
print(f"  F1分数 (F1-score)   : {f1:.4f}")
print(f"  AUC               : {auc:.4f}")
print("=" * 60)

# ============================================================
# 9. 特征重要性
# ============================================================
importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    '特征': all_feature_names,
    '重要性': importances
}).sort_values('重要性', ascending=False)

print("\n特征重要性排名（Top 20）:")
print(feat_imp_df.head(20).to_string(index=False))
print("-" * 60)

# ============================================================
# 10. 保存模型
# ============================================================
joblib.dump(best_model, 'random_forest_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("\n模型已保存: random_forest_model.pkl")
print("预处理器已保存: preprocessor.pkl")
print("=" * 60)
print("程序执行完毕！")
