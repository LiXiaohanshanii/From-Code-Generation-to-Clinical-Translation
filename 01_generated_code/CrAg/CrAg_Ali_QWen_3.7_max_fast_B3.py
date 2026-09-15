import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)

# =============================================================================
# 1. 数据加载与目标变量转换
# =============================================================================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols].copy()
y = (df[target_col] >= 20).astype(int)

# =============================================================================
# 2. 划分训练集与测试集 (7:3 分层抽样)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]}, 阳性比例: {y_train.mean():.3f}")
print(f"测试集大小: {X_test.shape[0]}, 阳性比例: {y_test.mean():.3f}")

# =============================================================================
# 3. 定义预处理策略
# =============================================================================
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
ordinal_cols = ['SER-T', 'Ink staining']
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), continuous_cols),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols)
    ],
    remainder='drop'
)

# =============================================================================
# 4. 构建Pipeline (预处理 + SMOTE + 随机森林)
# =============================================================================
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# 5. 超参数调优 (仅在训练集上进行)
# =============================================================================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=cv_strategy,
    n_jobs=1,
    verbose=1,
    refit=True
)

print("\n开始在训练集上进行超参数搜索...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"训练集交叉验证最佳F1: {grid_search.best_score_:.4f}")

# =============================================================================
# 6. 在独立测试集上评估最终模型
# =============================================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n========== 独立测试集评估结果 ==========")
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("========================================")