import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ============================================================
# 2. 特征与目标定义
# ============================================================
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color',
                'Protein', 'Transparency', 'RBC', 'SEX', 'AGE',
                'DEPT', 'DIAGNOSIS']
X = df[feature_cols].copy()

# 目标列：最后一列 CSF-T，二分类 >=20
y = (df.iloc[:, -1] >= 20).astype(int)

# ============================================================
# 3. 划分训练集 / 验证集（4:1）
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 4. 定义变量类型
# ============================================================
continuous_vars = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
onehot_vars = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_vars = ['SER-T', 'Ink staining']

# ============================================================
# 5. 构建预处理器（按顺序：数值→独热→序数）
# ============================================================

# 5.1 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 5.2 分类变量：独热编码（handle_unknown='ignore' 防止验证集新类别报错）
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 5.3 序数编码
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value',
                                     unknown_value=-1)

# 5.4 使用 ColumnTransformer 组合
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_vars),
        ('onehot', onehot_transformer, onehot_vars),
        ('ordinal', ordinal_transformer, ordinal_vars)
    ],
    remainder='drop'  # 不在列表中的列直接丢弃
)

# ============================================================
# 6. 构建完整 Pipeline（SMOTE 嵌入交叉验证内部，防止数据泄露）
# ============================================================
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=1  # 不使用多进程
)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_clf)
])

# ============================================================
# 7. 超参数调优（GridSearchCV + StratifiedKFold）
# ============================================================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    refit=True,
    verbose=1,
    n_jobs=1  # 不使用多进程
)

grid_search.fit(X_train, y_train)

# ============================================================
# 8. 输出最佳参数
# ============================================================
print("=" * 60)
print("最佳超参数：")
print(grid_search.best_params_)
print(f"最佳交叉验证 F1 分数：{grid_search.best_score_:.4f}")
print("=" * 60)

# ============================================================
# 9. 在验证集上评估
# ============================================================
best_model = grid_search.best_estimator_

# 预测类别与概率
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

# 计算评估指标
acc = accuracy_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
pre = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print("\n验证集评估结果：")
print(f"  准确率 (Accuracy) : {acc:.4f}")
print(f"  召回率 (Recall)   : {rec:.4f}")
print(f"  精确率 (Precision): {pre:.4f}")
print(f"  F1 分数 (F1-score): {f1:.4f}")
print(f"  AUC             : {auc:.4f}")
print("=" * 60)