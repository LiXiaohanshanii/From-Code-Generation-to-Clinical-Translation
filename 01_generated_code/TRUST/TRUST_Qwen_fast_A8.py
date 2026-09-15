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
cat_cols_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']
cat_cols_ord = ['TPPA']
num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
feature_cols = cat_cols_ohe + cat_cols_ord + num_cols

X = df[feature_cols].copy()
y = (df['TRUST'] >= 16).astype(int)

# ===================== 2. 划分训练集与测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 3. 构建预处理流水线 =====================
# 独热编码（分类变量无缺失值，handle_unknown='ignore'防止测试集出现新类别）
ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 序数编码（TPPA为有序分类变量）
ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 连续变量中位数填充
num_imputer = SimpleImputer(strategy='median')

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', ohe_transformer, cat_cols_ohe),
        ('ord', ord_transformer, cat_cols_ord),
        ('num', num_imputer, num_cols)
    ],
    remainder='drop'
)

# ===================== 4. 构建含SMOTE的完整流水线 =====================
# 使用imblearn的Pipeline以确保SMOTE仅在训练折内执行，避免数据泄露
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ===================== 5. 超参数调优（GridSearchCV，不使用多进程） =====================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,          # 不使用多进程
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ===================== 6. 模型评估 =====================
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
