import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
# 关键：使用imblearn的Pipeline和SMOTE以防止数据泄露
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# =============================================================================
# 1. 数据加载与目标变量构建
# =============================================================================
# 读取数据，utf-8编码
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 定义特征列
cat_ohe_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
cat_ord_cols = ['SER-T', 'Ink staining']
num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 构建二分类目标：CSF-T >= 20 为阳性(1)，否则为阴性(0)
y = (df['CSF-T'] >= 20).astype(int)
X = df[cat_ohe_cols + cat_ord_cols + num_cols]

print(f"数据集形状: {X.shape}")
print(f"目标变量分布:\n{y.value_counts(normalize=True)}")

# =============================================================================
# 2. 划分训练集与验证集 (4:1)
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 3. 构建防泄露的预处理 + 模型 Pipeline
# =============================================================================
# 注意：预处理顺序严格按照 数值 → 独热 → 序数 执行
# ColumnTransformer 默认并行执行各转换器，但输出拼接顺序由transformers列表顺序决定
# 这里我们按要求的逻辑顺序排列，并确保SMOTE在ColumnTransformer之后执行

preprocessor = ColumnTransformer(
    transformers=[
        # Step 1: 连续变量中位数填充
        ('num', SimpleImputer(strategy='median'), num_cols),
        # Step 2: 无序分类变量独热编码 (handle_unknown='ignore' 防止验证集出现新类别报错)
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_ohe_cols),
        # Step 3: 有序分类变量序数编码
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_ord_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# 使用 imblearn Pipeline 将 SMOTE 嵌入，确保只在训练折内过采样
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# 4. 超参数调优 (GridSearchCV, 不使用多进程)
# =============================================================================
# 参数名需加前缀以匹配Pipeline中的步骤名
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
    scoring='f1',           # 以F1为优化指标
    cv=5,                   # 5折交叉验证
    n_jobs=1,               # 不使用多进程
    refit=True,             # 用最佳参数在全部训练集上重新训练
    verbose=1
)

print("\n开始超参数搜索...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证 F1: {grid_search.best_score_:.4f}")

# =============================================================================
# 5. 验证集评估
# =============================================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print("\n" + "="*50)
print("验证集评估结果")
print("="*50)
print(f"Accuracy : {acc:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")
print("="*50)