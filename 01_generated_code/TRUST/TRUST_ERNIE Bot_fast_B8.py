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

# ==================== 1. 数据读取 ====================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# ==================== 2. 特征与目标分离 ====================
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
X = df[feature_cols].copy()
y = df.iloc[:, -1].copy()  # 目标列 TRUST（最后一列）

# 将目标转换为二分类：≥16 为 1，<16 为 0
y = (y >= 16).astype(int)

# ==================== 3. 分类变量与连续变量区分 ====================
categorical_cols_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_col_ord = ['TPPA']                        # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ==================== 4. 构建预处理Pipeline ====================
# 4.1 连续变量：中位数填充
continuous_imputer = SimpleImputer(strategy='median')

# 4.2 分类变量（独热编码）
ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# 4.3 分类变量（序数编码）
ord_encoder = OrdinalEncoder()

# 4.4 合并预处理步骤
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_imputer, continuous_cols),
        ('ohe', ohe_encoder, categorical_cols_ohe),
        ('ord', ord_encoder, categorical_col_ord)
    ],
    remainder='drop'
)

# ==================== 5. 划分训练集和测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== 6. 完整Pipeline（含SMOTE + 随机森林） ====================
rf_model = RandomForestClassifier(random_state=42, n_jobs=1)  # 不使用多进程

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# ==================== 7. 超参数网格 ====================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# ==================== 8. GridSearchCV超参数调优 ====================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

# ==================== 9. 最佳模型与预测 ====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# ==================== 10. 模型评估 ====================
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 60)
print("随机森林分类模型评估结果（二分类：TRUST ≥ 16）")
print("=" * 60)
print(f"最佳参数: {grid_search.best_params_}")
print("-" * 60)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):   {f1:.4f}")
print(f"AUC:                 {auc:.4f}")
print("=" * 60)

# ==================== 11. 额外输出 ====================
print(f"\n测试集样本数: {len(y_test)}")
print(f"测试集中正类(≥16)数量: {y_test.sum()}")
print(f"测试集中负类(<16)数量: {(y_test == 0).sum()}")
