import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 数据加载与目标变量转换
# =============================================================================
# 读取数据
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 定义特征列和目标列
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols]
y = df[target_col]

# 二分类任务转换: CSF-T >= 20 为阳性(1), 否则为阴性(0)
y_binary = (y >= 20).astype(int)

# =============================================================================
# 2. 划分训练集和验证集 (4:1)
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# =============================================================================
# 3. 定义预处理组件
# =============================================================================
# 根据数据类型分组
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['SER-T', 'Ink staining']

# 注意：预处理顺序严格按照 数值 -> 独热 -> 序数 执行
# 使用 ColumnTransformer 并行处理不同类型的特征，保证顺序正确且无泄露
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), continuous_cols),
        ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
        ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
    ],
    remainder='drop'  # 丢弃未指定的列以防万一
)

# =============================================================================
# 4. 构建防泄露Pipeline (SMOTE嵌入内部)
# =============================================================================
# 关键点：使用 imblearn.pipeline.Pipeline 而非 sklearn.pipeline.Pipeline
# 这确保了 SMOTE 仅在 GridSearchCV 的每个 fold 的训练集上执行
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# 5. 超参数调优 (GridSearchCV)
# =============================================================================
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
    scoring='f1',           # 以F1作为优化指标
    cv=5,                   # 5折交叉验证
    n_jobs=1,               # 不使用多进程，避免Windows下fork问题
    verbose=1
)

print("开始模型训练与超参数调优...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证 F1 分数: {grid_search.best_score_:.4f}")

# =============================================================================
# 6. 模型评估 (在独立验证集上)
# =============================================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print("\n===== 独立验证集评估结果 =====")
print(f"Accuracy : {acc:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")
print("==============================")