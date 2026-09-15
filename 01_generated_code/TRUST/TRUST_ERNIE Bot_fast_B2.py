# -*- coding: utf-8 -*-
"""
随机森林分类模型构建与评估
环境：PyCharm 2025.2.3
数据集：train_data.csv（utf-8编码）
目标：预测TRUST是否>=16（二分类）
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
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import warnings
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 70)
print("步骤1：加载数据")
print("=" * 70)

df = pd.read_csv('train_data.csv', encoding='utf-8')
print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"目标列 TRUST 唯一值: {sorted(df['TRUST'].unique())}")
print(f"目标列 TRUST 分布:\n{df['TRUST'].value_counts().sort_index()}")

# ============================================================
# 2. 特征与目标分离
# ============================================================
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[feature_cols].copy()
y_raw = df.iloc[:, -1].copy()          # TRUST列
y = (y_raw >= 16).astype(int)          # 二值化：>=16为1，否则为0

print(f"\n目标变量二分类分布:\n{y.value_counts()}")

# ============================================================
# 3. 定义列类型
# ============================================================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal_cols = ['TPPA']
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ============================================================
# 4. 构建预处理Pipeline
# ============================================================
print("\n" + "=" * 70)
print("步骤2：构建预处理Pipeline")
print("=" * 70)

# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量（独热编码）：无缺失值
categorical_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 分类变量（序数编码）：无缺失值
categorical_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value',
                                                   unknown_value=-1)

# 组合预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_cols),
        ('cat_ohe', categorical_onehot_transformer, categorical_onehot_cols),
        ('cat_ord', categorical_ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'
)

# 拟合并转换
X_processed = preprocessor.fit_transform(X)

# 获取特征名（便于后续分析）
ohe_names = preprocessor.named_transformers_['cat_ohe'].get_feature_names_out(categorical_onehot_cols)
all_feature_names = np.concatenate([continuous_cols, ohe_names, categorical_ordinal_cols])
print(f"预处理后特征维度: {X_processed.shape}")
print(f"特征总数量: {len(all_feature_names)}")

# ============================================================
# 5. 划分训练集与测试集
# ============================================================
print("\n" + "=" * 70)
print("步骤3：划分训练集与测试集")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"训练集类别分布:\n{pd.Series(y_train).value_counts()}")
print(f"测试集类别分布:\n{pd.Series(y_test).value_counts()}")

# ============================================================
# 6. SMOTE处理分类不平衡
# ============================================================
print("\n" + "=" * 70)
print("步骤4：使用SMOTE处理训练集不平衡")
print("=" * 70)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE前训练集类别分布:\n{pd.Series(y_train).value_counts()}")
print(f"SMOTE后训练集类别分布:\n{pd.Series(y_train_smote).value_counts()}")

# ============================================================
# 7. 随机森林超参数调优
# ============================================================
print("\n" + "=" * 70)
print("步骤5：随机森林超参数调优（GridSearchCV）")
print("=" * 70)

rf_clf = RandomForestClassifier(
    n_jobs=1,
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10] ,
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,
    verbose=1,
    refit=True
)

grid_search.fit(X_train_smote, y_train_smote)

print(f"\n最佳参数组合:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ============================================================
# 8. 模型评估
# ============================================================
print("\n" + "=" * 70)
print("步骤6：在测试集上评估模型")
print("=" * 70)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "-" * 50)
print("模型评估结果（测试集）")
print("-" * 50)
print(f"  准确率 (Accuracy)      : {accuracy:.4f}")
print(f"  召回率 (Recall)         : {recall:.4f}")
print(f"  精确率 (Precision)      : {precision:.4f}")
print(f"  F1分数 (F1-score)       : {f1:.4f}")
print(f"  AUC曲线下面积 (AUC)     : {auc:.4f}")
print("-" * 50)

# ============================================================
# 9. 特征重要性
# ============================================================
print("\n" + "=" * 70)
print("步骤7：特征重要性排名（Top 15）")
print("=" * 70)

feature_importance = best_model.feature_importances_
importance_df = pd.DataFrame({
    '特征': all_feature_names,
    '重要性': feature_importance
}).sort_values(by='重要性', ascending=False)

print(importance_df.head(15).to_string(index=False))

# ============================================================
# 10. 保存模型与预处理器
# ============================================================
print("\n" + "=" * 70)
print("步骤8：保存模型与预处理器")
print("=" * 70)

joblib.dump(best_model, 'rf_model_best.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("模型已保存: rf_model_best.pkl")
print("预处理器已保存: preprocessor.pkl")

print("\n" + "=" * 70)
print("程序执行完毕！")
print("=" * 70)
