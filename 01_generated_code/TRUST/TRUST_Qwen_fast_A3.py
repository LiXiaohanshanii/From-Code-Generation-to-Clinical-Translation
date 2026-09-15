import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ===================== 1. 数据加载与目标变量构建 =====================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 定义特征列
categorical_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
ordinal_col = ['TPPA']
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
feature_cols = categorical_cols + ordinal_col + continuous_cols

X = df[feature_cols].copy()
# 二分类目标：TRUST >= 16 为 1，否则为 0
y = (df['TRUST'] >= 16).astype(int)

# ===================== 2. 构建预处理流水线 =====================
# 分类变量独热编码
ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# TPPA序数编码（按字符串排序作为默认顺序，若需自定义顺序请修改categories参数）
ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 连续变量中位数填充
median_imputer = SimpleImputer(strategy='median')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', ohe_transformer, categorical_cols),
        ('ord', ord_transformer, ordinal_col),
        ('num', median_imputer, continuous_cols)
    ],
    remainder='drop'
)

# ===================== 3. 构建含SMOTE的完整流水线 =====================
# 使用imblearn的Pipeline以确保SMOTE仅在训练折内执行，避免数据泄露
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ===================== 4. 超参数调优（不使用多进程） =====================
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
    n_jobs=1,          # 不使用多进程
    refit=True,
    verbose=0
)

grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print(f"最佳超参数: {grid_search.best_params_}")
print(f"交叉验证最佳F1: {grid_search.best_score_:.4f}")

# ===================== 5. 模型评估（在全部数据上评估最终模型） =====================
y_pred = best_model.predict(X)
y_prob = best_model.predict_proba(X)[:, 1]

acc = accuracy_score(y, y_pred)
rec = recall_score(y, y_pred)
prec = precision_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)

print("\n===== 模型评估结果 =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
