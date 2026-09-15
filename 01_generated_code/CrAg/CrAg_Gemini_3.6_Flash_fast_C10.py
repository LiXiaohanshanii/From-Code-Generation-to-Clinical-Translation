import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 1. 加载数据集
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 2. 构建二分类目标变量 (CSF-T >= 20 为 1，否则为 0)
# 注意：CSF-T 为最后一列滴度变量
y = (df['CSF-T'] >= 20).astype(int)
X = df.drop(columns=['CSF-T'])

# 3. 明确定义三类特征列
num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['SER-T', 'Ink staining']

# 4. 划分训练集与验证集 (比例 4:1，采用分层抽样以保持正负样本比例一致)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. 构建数据预处理流水线 (顺序：数值中位数填充 -> 独热编码 -> 序数编码)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
    ]
)

# 6. 构建包含 SMOTE 的 imblearn 管道 (防止交叉验证中的数据泄露)
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 7. 超参数网格配置 (严格对应需求)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 8. 网格搜索与超参数调优 (以 F1 分数为评价指标，单线程运行)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1
)

# 9. 模型训练
grid_search.fit(X_train, y_train)

# 10. 验证集评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)[:, 1]

# 11. 计算指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)

# 12. 打印评估结果与最优参数
print("=== 最优超参数组合 ===")
print(grid_search.best_params_)
print("\n=== 验证集模型评估指标 ===")
print(f"准确率 (Accuracy) : {accuracy:.4f}")
print(f"召回率 (Recall)   : {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1 分数 (F1-score): {f1:.4f}")
print(f"ROC-AUC 曲线面积  : {auc:.4f}")