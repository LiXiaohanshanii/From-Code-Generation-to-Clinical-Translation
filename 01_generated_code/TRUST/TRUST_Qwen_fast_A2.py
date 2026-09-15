import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
categorical_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ord_cols = ['TPPA']
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

feature_cols = categorical_ohe_cols + categorical_ord_cols + continuous_cols
X = df[feature_cols].copy()

# 构建二分类目标：TRUST >= 16 为阳性(1)，否则为阴性(0)
y = (df['TRUST'] >= 16).astype(int)

# ===================== 2. 划分训练集与测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 3. 构建预处理 + SMOTE + 模型管道 =====================
# 预处理器：对不同类型特征分别处理
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_ohe_cols),
        ('ord', OrdinalEncoder(), categorical_ord_cols),
        ('num', SimpleImputer(strategy='median'), continuous_cols),
    ],
    remainder='drop'
)

# 使用 imblearn 的 Pipeline 以支持 SMOTE 步骤
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
    'classifier__class_weight': ['balanced'],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,          # 不使用多进程
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"交叉验证最佳F1: {grid_search.best_score_:.4f}")

# ===================== 5. 模型评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n===== 测试集评估结果 =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
