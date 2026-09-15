import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
# 关键：必须使用imblearn的Pipeline以支持SMOTE步骤
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# =============================================================================
# 1. 数据加载与目标变量构建
# =============================================================================
# 读取数据
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 定义特征列和目标列
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols].copy()
y_raw = df[target_col].copy()

# 构建二分类目标：>=20 为阳性(1)，<20 为阴性(0)
y = (y_raw >= 20).astype(int)

# =============================================================================
# 2. 划分训练集与验证集 (4:1)
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集样本量: {X_train.shape[0]}, 阳性比例: {y_train.mean():.3f}")
print(f"验证集样本量: {X_val.shape[0]}, 阳性比例: {y_val.mean():.3f}")

# =============================================================================
# 3. 定义预处理组件与Pipeline
# =============================================================================
# 根据数据类型分组
numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_features = ['SER-T', 'Ink staining']

# 构建Pipeline：严格遵循 数值→独热→序数→SMOTE→模型 的顺序
# 使用imblearn.pipeline.Pipeline确保SMOTE不会泄露到验证集
pipe = Pipeline([
    # Step 1: 连续变量缺失值中位数填充
    ('num_imputer', SimpleImputer(strategy='median')),

    # Step 2: 分类变量独热编码
    ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),

    # Step 3: 序数变量编码 (假设类别已按自然顺序排列，如阴性/弱阳/阳性)
    # 若实际类别顺序非字母序，需在此处传入categories参数显式指定
    ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value',
                                       unknown_value=-1)),

    # Step 4: SMOTE过采样 (仅在CV训练折内执行)
    ('smote', SMOTE(random_state=42)),

    # Step 5: 随机森林分类器
    ('classifier', RandomForestClassifier(random_state=42))
], memory=None)

# 设置各步骤对应的特征列
# 注意：Pipeline中的transformer默认对所有输入特征操作
# 这里我们需要通过ColumnTransformer或手动调整
# 为简化代码并保持预处理顺序，我们采用自定义方式处理混合类型
# 重新构建更精确的Pipeline：

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# 重建完整Pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# 4. 超参数调优 (GridSearchCV, F1评分, 不使用多进程)
# =============================================================================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=final_pipeline,
    param_grid=param_grid,
    scoring='f1',  # 以F1为优化指标
    cv=5,  # 5折交叉验证
    n_jobs=1,  # 不使用多进程
    verbose=1,
    refit=True  # 用最佳参数在全部训练集上重训
)

print("\n开始超参数搜索...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1: {grid_search.best_score_:.4f}")

# =============================================================================
# 5. 验证集评估
# =============================================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

# 计算各项指标
acc = accuracy_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print("\n" + "=" * 50)
print("验证集评估结果:")
print("=" * 50)
print(f"Accuracy:  {acc:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("=" * 50)