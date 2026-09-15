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
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列TRUST为滴度数值，转换为二分类: >=16 为阳性(1), <16 为阴性(0)
y = (df['TRUST'] >= 16).astype(int)
X = df.drop(columns=['TRUST'])

# ===================== 2. 划分训练集与测试集 =====================
# 先划分再处理，防止数据泄露；SMOTE仅在训练集上执行
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 3. 定义特征分组 =====================
categorical_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ord_cols = ['TPPA']
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ===================== 4. 构建预处理Pipeline =====================
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_ohe_cols),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ord_cols),
        ('num', SimpleImputer(strategy='median'), continuous_cols)
    ],
    remainder='drop'  # 丢弃未指定的列，确保安全
)

# ===================== 5. 构建含SMOTE的完整Pipeline =====================
# 使用imblearn的Pipeline确保SMOTE只在训练折中执行，不会泄露到验证/测试集
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ===================== 6. 超参数调优（不使用多进程） =====================
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
    scoring='f1',          # 不平衡数据以F1为主要优化指标
    cv=5,
    n_jobs=1,              # 明确禁用多进程
    refit=True,
    verbose=1
)

print("开始超参数调优...")
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}\n")

# ===================== 7. 模型评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'F1-score': f1_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_prob)
}

print("=" * 40)
print("测试集模型评估结果:")
print("=" * 40)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name:>12s}: {metric_value:.4f}")
print("=" * 40)
