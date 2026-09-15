import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ============================================================
# 2. 定义特征列和目标列
# ============================================================
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color',
                'Protein', 'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'  # 最后一列

X = df[feature_cols]
y_raw = df[target_col]

# ============================================================
# 3. 构造二分类目标：是否 >= 20
# ============================================================
y = (y_raw >= 20).astype(int)

# ============================================================
# 4. 划分训练集和验证集 (4:1)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 5. 定义变量类型
# ============================================================
# 分类变量（独热编码）
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

# 分类变量（序数编码）
ordinal_cols = ['SER-T', 'Ink staining']

# 连续变量
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# ============================================================
# 6. 构建预处理管道
# ============================================================

# 连续变量：中位数填充
continuous_transformer = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码
onehot_transformer = SkPipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数编码
ordinal_transformer = SkPipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合所有预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('onehot', onehot_transformer, onehot_cols),
        ('ordinal', ordinal_transformer, ordinal_cols)
    ],
    remainder='drop'  # 确保没有遗漏列
)

# ============================================================
# 7. 使用 imblearn.pipeline.Pipeline 嵌入 SMOTE，防止数据泄露
# ============================================================
model = RandomForestClassifier(random_state=42, n_jobs=1)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', model)
])

# ============================================================
# 8. 超参数调优（GridSearchCV 内部使用交叉验证）
# ============================================================
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
    refit=True,
    n_jobs=1  # 不使用多进程
)

grid_search.fit(X_train, y_train)

# ============================================================
# 9. 输出最佳参数
# ============================================================
print("=" * 60)
print("最佳超参数：")
print(grid_search.best_params_)
print(f"最佳交叉验证 F1 分数：{grid_search.best_score_:.4f}")
print("=" * 60)

# ============================================================
# 10. 在验证集上评估
# ============================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("验证集评估结果：")
print(f"  准确率 (Accuracy) : {accuracy:.4f}")
print(f"  召回率 (Recall)   : {recall:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  F1 分数 (F1-score): {f1:.4f}")
print(f"  AUC               : {auc:.4f}")
print("=" * 60)

# ============================================================
# 11. 额外信息：训练集 vs 验证集类别分布
# ============================================================
print("\n训练集目标分布：")
print(y_train.value_counts().sort_index())
print("\n验证集目标分布：")
print(y_test.value_counts().sort_index())