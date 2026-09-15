import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ===================== 1. 加载数据 =====================
data = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 分离特征和目标
X = data.iloc[:, :-1]          # 所有特征列
y = data.iloc[:, -1]           # 最后一列 CSF-T

# ===================== 2. 目标变量转换（二分类：>=20 vs <20）=====================
y_binary = (y >= 20).astype(int)

# ===================== 3. 定义列类型 =====================
# 分类变量 - 独热编码
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
# 分类变量 - 序数编码
ordinal_cols = ['SER-T', 'Ink staining']
# 连续变量
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# ===================== 4. 构建预处理管道 =====================
# 连续变量缺失值处理：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 序数编码
ordinal_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 独热编码
onehot_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 组合所有预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('onehot', onehot_transformer, onehot_cols)
    ]
)

# ===================== 5. 构建完整模型管道（含SMOTE）=====================
# 使用 ImbPipeline 支持 SMOTE 在 pipeline 中
model = RandomForestClassifier(random_state=42, n_jobs=1)

# 完整管道：预处理 -> SMOTE -> 随机森林
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', model)
])

# ===================== 6. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ===================== 7. 超参数调优 =====================
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
    scoring='f1',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ===================== 8. 输出最佳参数 =====================
print("=" * 60)
print("最佳超参数：")
print(grid_search.best_params_)
print(f"最佳交叉验证F1分数：{grid_search.best_score_:.4f}")
print("=" * 60)

# ===================== 9. 模型评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n模型评估结果：")
print(f"准确率（Accuracy）：  {accuracy:.4f}")
print(f"召回率（Recall）：    {recall:.4f}")
print(f"精确率（Precision）： {precision:.4f}")
print(f"F1分数（F1-score）：  {f1:.4f}")
print(f"AUC：                {auc:.4f}")
print("=" * 60)