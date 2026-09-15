import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# ===================== 1. 读取数据 =====================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# ===================== 2. 构建二分类目标变量 =====================
# TRUST为滴度类型，判断是否 >= 16
y = (df['TRUST'] >= 16).astype(int)

# ===================== 3. 定义特征列 =====================
categorical_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ord_cols = ['TPPA']
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

feature_cols = categorical_ohe_cols + categorical_ord_cols + numerical_cols
X = df[feature_cols].copy()

# ===================== 4. 划分训练集与测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 5. 构建预处理Pipeline =====================
# 分类变量独热编码
ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 分类变量序数编码（TPPA按自然顺序编码）
ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 连续变量中位数填充
num_imputer = SimpleImputer(strategy='median')

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', ohe_transformer, categorical_ohe_cols),
        ('ord', ord_transformer, categorical_ord_cols),
        ('num', num_imputer, numerical_cols)
    ],
    remainder='drop'
)

# ===================== 6. 构建含SMOTE的完整Pipeline =====================
# 使用imblearn的Pipeline以确保SMOTE仅在训练折内执行，避免数据泄露
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ===================== 7. 超参数调优（GridSearchCV，不使用多进程） =====================
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
    cv=5,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ===================== 8. 在测试集上评估最优模型 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# ===================== 9. 输出评估指标 =====================
print("\n===== 测试集评估结果 =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
