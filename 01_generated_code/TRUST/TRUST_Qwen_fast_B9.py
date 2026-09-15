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

# ===================== 1. 数据加载与目标变量构建 =====================
# 读取数据集，指定utf-8编码
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 定义特征列
categorical_cols = ['SEX', 'DEPT', 'DIAGNOSIS', 'TPPA']
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
feature_cols = categorical_cols + numerical_cols

# 构建二分类目标变量：TRUST >= 16 为阳性(1)，否则为阴性(0)
y = (df['TRUST'] >= 16).astype(int)
X = df[feature_cols].copy()

# ===================== 2. 划分训练集与测试集 =====================
# 先划分数据集，确保SMOTE只在训练集上执行，避免数据泄露
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 3. 构建预处理与模型Pipeline =====================
# 注意：DIAGNOSIS 在需求描述中存在拼写差异(DIAGONSIS)，此处以特征列定义中的 DIAGNOSIS 为准

# 分类变量预处理：
# SEX, DEPT, DIAGNOSIS -> OneHotEncoding
# TPPA -> OrdinalEncoding (滴度相关，具有自然顺序)
onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['TPPA']

preprocessor = ColumnTransformer(
    transformers=[
        ('num_imputer', SimpleImputer(strategy='median'), numerical_cols),
        ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
        ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
    ],
    remainder='drop'
)

# 定义随机森林超参数搜索空间（不使用多进程，n_jobs=1）
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [1],
    'clf__class_weight': ['balanced']
}

# 使用 imblearn 的 Pipeline 以支持 SMOTE 步骤
# SMOTE 放在预处理器之后、分类器之前，仅对训练集进行过采样
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=1))
])

# ===================== 4. 超参数调优（GridSearchCV） =====================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,       # 明确不使用多进程
    verbose=1
)

print("开始超参数调优...")
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证 F1-score: {grid_search.best_score_:.4f}")

# ===================== 5. 模型评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n===== 测试集评估结果 =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
