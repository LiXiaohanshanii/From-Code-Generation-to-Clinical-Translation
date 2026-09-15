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

# ============================================================
# 1. 数据加载与目标变量构建
# ============================================================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 定义特征列
cat_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
cat_cols_ordinal = ['TPPA']
num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

feature_cols = cat_cols_onehot + cat_cols_ordinal + num_cols
X = df[feature_cols].copy()

# 构建二分类目标: TRUST >= 16 为阳性(1), 否则为阴性(0)
y = (df['TRUST'] >= 16).astype(int)

# ============================================================
# 2. 划分训练集与测试集
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3. 构建预处理 + SMOTE + 模型的完整Pipeline
# ============================================================
# 注意: SMOTE 必须在 fit 阶段仅对训练集生效，防止数据泄露
# 使用 imblearn.pipeline.Pipeline 以支持在交叉验证中正确应用SMOTE

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols_onehot),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols_ordinal),
        ('num_impute', SimpleImputer(strategy='median'), num_cols)
    ],
    remainder='drop'
)

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ============================================================
# 4. 超参数调优 (不使用多进程 n_jobs=1)
# ============================================================
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
    scoring='f1',       # 不平衡数据推荐以F1作为优化指标
    n_jobs=1,           # 明确禁用多进程
    verbose=1
)

print("开始超参数调优...")
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}\n")

# ============================================================
# 5. 模型评估
# ============================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 50)
print("测试集模型评估结果")
print("=" * 50)
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC曲线下面积:      {auc:.4f}")
print("=" * 50)
