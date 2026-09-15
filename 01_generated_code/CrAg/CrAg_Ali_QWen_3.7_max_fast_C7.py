import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
# 关键导入：使用imblearn的Pipeline防止SMOTE数据泄露
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# =============================================================================
# 1. 数据加载与目标变量转换
# =============================================================================
# 读取数据，指定utf-8编码
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 定义特征列和目标列
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols]
y_raw = df[target_col]

# 任务目标：预测CSF-T是否 >= 20 (二分类: 1=阳性/高滴度, 0=阴性/低滴度)
y = (y_raw >= 20).astype(int)

# =============================================================================
# 2. 划分训练集与验证集 (4:1)
# =============================================================================
# random_state保证结果可复现，stratify=y保证分层抽样维持类别比例
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}")
print(f"训练集正样本比例: {y_train.mean():.3f}")

# =============================================================================
# 3. 构建防泄露的预处理+模型Pipeline
# =============================================================================
# 按照要求定义特征分组
numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_features = ['SER-T', 'Ink staining']

# 注意：ColumnTransformer内部并行执行各分支，但整体嵌入Pipeline后
# SMOTE会在ColumnTransformer之后、RF之前仅对训练折生效
preprocessor = ColumnTransformer(
    transformers=[
        # 连续变量：中位数填充缺失值
        ('num', SimpleImputer(strategy='median'), numeric_features),
        # 分类变量(独热)：handle_unknown='ignore'防止验证集出现新类别报错
        ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
        # 分类变量(序数)：SER-T和Ink staining具有等级意义
        ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
    ],
    remainder='drop'  # 丢弃未指定的列，增强安全性
)

# 构建imblearn Pipeline：预处理 → SMOTE → 随机森林
# 这样GridSearchCV在每次CV迭代中，SMOTE只fit_transform训练折
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# 4. 超参数调优 (以F1为评分指标，不使用多进程)
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
    scoring='f1',           # 以F1作为优化目标
    cv=5,                   # 5折交叉验证
    n_jobs=1,               # 不使用多进程，避免Windows下fork问题
    verbose=1,
    refit=True              # 用最佳参数在全部训练集上重新拟合
)

print("\n开始超参数搜索...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# =============================================================================
# 5. 在独立验证集上评估最终模型
# =============================================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

# 计算各项评估指标
metrics = {
    'Accuracy': accuracy_score(y_val, y_pred),
    'Recall': recall_score(y_val, y_pred),
    'Precision': precision_score(y_val, y_pred),
    'F1-score': f1_score(y_val, y_pred),
    'AUC': roc_auc_score(y_val, y_prob)
}

print("\n" + "="*50)
print("独立验证集评估结果:")
print("="*50)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name:<12}: {metric_value:.4f}")
print("="*50)