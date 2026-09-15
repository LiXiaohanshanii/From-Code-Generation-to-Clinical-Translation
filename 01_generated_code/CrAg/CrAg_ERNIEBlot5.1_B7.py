import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ===================== 1. 数据加载 =====================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ===================== 2. 分离特征与目标 =====================
# 特征列（排除目标列 CSF-T，目标列是最后一列）
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']

X = df[feature_cols]
y = df.iloc[:, -1]  # 最后一列 CSF-T

# ===================== 3. 目标变量转换为二分类 =====================
# 预测 CSF-T 是否 >= 20
y_binary = (y >= 20).astype(int)

# ===================== 4. 定义变量类型 =====================
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['SER-T', 'Ink staining']
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# ===================== 5. 构建预处理管道 =====================
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 序数编码
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 独热编码
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 使用 ColumnTransformer 组合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('ohe', onehot_transformer, onehot_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ===================== 6. 构建完整管道（含SMOTE和随机森林） =====================
# 注意：SMOTE 不能放在 sklearn 的 Pipeline 中，需用 imblearn 的 Pipeline
rf_classifier = RandomForestClassifier(random_state=42, n_jobs=1)

# 定义参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用 ImbPipeline 将预处理器、SMOTE、分类器串联
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_classifier)
])

# ===================== 7. 划分训练集与测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ===================== 8. 网格搜索超参数调优 =====================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ===================== 9. 输出最佳参数 =====================
print("=" * 60)
print("最佳参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证 F1 分数: {grid_search.best_score_:.4f}")
print("=" * 60)

# ===================== 10. 模型评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("模型评估结果（测试集）:")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1 分数 (F1-score): {f1:.4f}")
print(f"AUC:                 {auc:.4f}")
print("=" * 60)