import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
# 关键：必须使用 imblearn 的 Pipeline 以防止 SMOTE 数据泄露
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# =============================================================================
# 1. 数据加载与目标变量转换
# =============================================================================
# 读取数据
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 定义特征列和目标列
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols]
y = (df[target_col] >= 20).astype(int)  # 二分类: ≥20为1, <20为0

# =============================================================================
# 2. 划分训练集和验证集 (4:1)
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 3. 定义预处理组件 (严格按照: 数值 → 独热 → 序数 的顺序)
# =============================================================================
numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_features = ['SER-T', 'Ink staining']

# 数值型: 中位数填充
numeric_transformer = SimpleImputer(strategy='median')

# 独热编码 (分类变量无缺失值，handle_unknown='ignore' 增强鲁棒性)
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 序数编码 (分类变量无缺失值)
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value',
                                      unknown_value=-1)

# 组合预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('onehot', onehot_transformer, onehot_features),
        ('ordinal', ordinal_transformer, ordinal_features)
    ],
    remainder='drop'  # 丢弃未指定的列，防止意外泄露
)

# =============================================================================
# 4. 构建防泄露 Pipeline (SMOTE 嵌入 CV 内部)
# =============================================================================
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),       # ✅ 仅在训练折内执行
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# 5. 超参数调优 (GridSearchCV, F1评分, 不使用多进程)
# =============================================================================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,          # ❌ 不使用多进程
    refit=True,
    verbose=1
)

print("=" * 60)
print("开始超参数调优 (SMOTE安全地嵌入在CV内部)...")
print("=" * 60)
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳CV F1分数: {grid_search.best_score_:.4f}")

# =============================================================================
# 6. 在独立验证集上评估最终模型
# =============================================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print("\n" + "=" * 60)
print("独立验证集评估结果")
print("=" * 60)
print(f"Accuracy : {acc:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")
print("=" * 60)