# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 检验科数据分析
适用环境：PyCharm 2025.2.3
任务：预测TRUST滴度是否>=16（二分类）
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
print(f"所有列名: {df.columns.tolist()}")
print(f"\n目标列 TRUST 原始值分布:")
print(df.iloc[:, -1].value_counts().sort_index())

# ============================================================
# 2. 特征与目标分离 + 二值化
# ============================================================
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[feature_cols].copy()
y = df.iloc[:, -1].copy()  # TRUST（最后一列）

# 二分类目标：>=16 为 1，<16 为 0
y_binary = (y >= 16).astype(int)
print(f"\n【步骤2】目标二值化结果分布:")
print(f"  TRUST >= 16 (1): {y_binary.sum()} 例")
print(f"  TRUST <  16 (0): {(y_binary == 0).sum()} 例")
print(f"  不平衡比例: {y_binary.mean():.2%}")

# ============================================================
# 3. 定义各列类型
# ============================================================
cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
cat_ordinal_cols = ['TPPA']                      # 序数编码
num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT',
            'NC', 'LY', 'NLR']                    # 连续变量

# ============================================================
# 4. 构建预处理管道（ColumnTransformer）
# ============================================================
print("\n【步骤3】构建预处理管道...")

# 4.1 连续变量：中位数填充（分类变量无缺失，不需要填充器）
num_transformer = SimpleImputer(strategy='median')

# 4.2 独热编码
cat_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 4.3 序数编码
cat_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat_ohe', cat_onehot_transformer, cat_onehot_cols),
        ('cat_ord', cat_ordinal_transformer, cat_ordinal_cols)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# ============================================================
# 5. 划分训练集/测试集
# ============================================================
print("\n【步骤4】划分训练集与测试集 (80%/20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

# 先对训练集做预处理
X_train_processed = preprocessor.fit_transform(X_train)
# 测试集只做 transform（不 fit）
X_test_processed = preprocessor.transform(X_test)

# 生成编码后的特征名称
ohe_names = preprocessor.named_transformers_['cat_ohe'].get_feature_names_out(cat_onehot_cols)
all_feature_names = list(num_cols) + list(ohe_names) + cat_ordinal_cols
print(f"  预处理后特征数: {X_train_processed.shape[1]}")
print(f"  训练集: {X_train_processed.shape[0]} 样本")
print(f"  测试集: {X_test_processed.shape[0]} 样本")

# ============================================================
# 6. SMOTE 处理训练集不平衡
# ============================================================
print("\n【步骤5】SMOTE 过采样处理不平衡...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"  SMOTE前 训练集类别分布: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
print(f"  SMOTE后 训练集类别分布: 0={sum(y_train_smote==0)}, 1={sum(y_train_smote==1)}")

# ============================================================
# 7. 随机森林 + GridSearchCV 超参数调优
# ============================================================
print("\n【步骤6】随机森林超参数调优 (GridSearchCV, n_jobs=1)...")

rf_clf = RandomForestClassifier(random_state=42, n_jobs=1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
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

best_model = grid_search.best_estimator_
print(f"\n  最优参数: {grid_search.best_params_}")
print(f"  最优交叉验证 F1: {grid_search.best_score_:.4f}")

# ============================================================
# 8. 模型评估（测试集）
# ============================================================
print("\n【步骤7】模型评估（测试集）")
print("=" * 60)

y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"  准确率 (Accuracy)  : {acc:.4f}")
print(f"  召回率 (Recall)     : {rec:.4f}")
print(f"  精确率 (Precision)  : {pre:.4f}")
print(f"  F1分数 (F1-score)   : {f1:.4f}")
print(f"  AUC                : {auc:.4f}")
print("=" * 60)

# ============================================================
# 9. 特征重要性
# ============================================================
print("\n【步骤8】特征重要性（Top 15）:")
importance = best_model.feature_importances_
imp_df = pd.DataFrame({'特征': all_feature_names, '重要性': importance})
imp_df = imp_df.sort_values('重要性', ascending=False).head(15).reset_index(drop=True)
print(imp_df.to_string(index=False))

# ============================================================
# 10. 保存模型
# ============================================================
joblib.dump(best_model, 'rf_model_best.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("\n模型已保存: rf_model_best.pkl")
print("预处理器已保存: preprocessor.pkl")
print("\n程序执行完毕！")
