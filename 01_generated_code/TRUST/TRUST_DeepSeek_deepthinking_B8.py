import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 读取数据 =====================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征和目标列（目标列是最后一列 TRUST）
X = data.drop('TRUST', axis=1)
y = data['TRUST']

# 将目标转为二分类：≥16 记为 1，否则为 0
y_binary = (y >= 16).astype(int)

# ===================== 2. 定义特征类型 =====================
categorical_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ord = ['TPPA']                       # 序数编码
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ===================== 3. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# ===================== 4. 构建预处理流水线 =====================
# 数值变量：中位数填充
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码变量：众数填充（实际无缺失，以防万一），并设置 sparse_output=False 输出稠密矩阵
ohe_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数编码变量：众数填充，OrdinalEncoder 默认按字母顺序排序
ord_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('ohe', ohe_transformer, categorical_ohe),
    ('ord', ord_transformer, categorical_ord)
])

# ===================== 5. 构建完整模型流水线（含 SMOTE） =====================
rf = RandomForestClassifier(random_state=42, n_jobs=1)

# 超参数网格（仅 n_estimators 有两个候选，其他固定）
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用 imblearn 的 Pipeline 确保 SMOTE 仅在交叉验证的训练折内进行
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])

# ===================== 6. 超参数调优（GridSearchCV） =====================
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,          # 禁用多进程
    verbose=1
)
grid_search.fit(X_train, y_train)

# ===================== 7. 评估最佳模型 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n========== 最佳超参数 ==========")
print(grid_search.best_params_)
print("\n========== 测试集评估指标 ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")