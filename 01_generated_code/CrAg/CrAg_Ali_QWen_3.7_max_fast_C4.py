import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 数据加载与目标变量二值化
# =============================================================================
# 读取数据，指定utf-8编码
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 定义特征列和目标列
target_col = 'CSF-T'
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']

X = df[feature_cols]
y = (df[target_col] >= 20).astype(int)  # 二分类：>=20为1，<20为0

# =============================================================================
# 2. 划分训练集与验证集 (4:1)
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 3. 构建防泄露的预处理+模型Pipeline
# =============================================================================
# 分类变量分组
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['SER-T', 'Ink staining']
numeric_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 按照要求顺序：数值 → 独热 → 序数 构建ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
    ],
    remainder='drop'
)

# 使用 imblearn.pipeline.Pipeline 将 SMOTE 嵌入管道内部
# 这是防止数据泄露的核心：SMOTE只在CV的训练折上拟合
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# 4. 超参数调优 (GridSearchCV, 不使用多进程)
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
    scoring='f1',
    cv=5,
    n_jobs=1,          # 不使用多进程
    refit=True,
    verbose=0
)

print("开始超参数调优...")
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳CV F1分数: {grid_search.best_score_:.4f}\n")

# =============================================================================
# 5. 在独立验证集上评估模型
# =============================================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print("=" * 50)
print("       独立验证集评估结果")
print("=" * 50)
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 50)