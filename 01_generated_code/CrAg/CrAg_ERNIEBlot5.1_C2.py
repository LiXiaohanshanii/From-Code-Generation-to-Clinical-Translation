import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ======================== 1. 加载数据 ========================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ======================== 2. 定义特征和目标 ========================
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color',
                'Protein', 'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'  # 最后一列

X = df[feature_cols].copy()
y = df[target_col].copy()

# ======================== 3. 构建二分类目标 ========================
y_binary = (y >= 20).astype(int)

# ======================== 4. 划分训练集和验证集（4:1） ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ======================== 5. 定义列类型 ========================
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

ordinal_cols = ['SER-T', 'Ink staining']

# ======================== 6. 构建预处理管道（数值→独热→序数） ========================

# 6.1 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 6.2 独热编码（处理指定分类变量）
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# 6.3 序数编码（处理指定分类变量）
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 6.4 组合列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_cols),
        ('ohe', onehot_transformer, onehot_cols),
        ('ord', ordinal_transformer, ordinal_cols)
    ],
    remainder='drop'  # 其余列丢弃（理论上不应存在）
)

# ======================== 7. 构建完整模型管道（SMOTE嵌入交叉验证内部） ========================
model = RandomForestClassifier(random_state=42)

# 使用 imblearn 的 Pipeline，确保 SMOTE 在交叉验证内部执行，防止数据泄露
full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', model)
])

# ======================== 8. 超参数调优（GridSearchCV） ========================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ======================== 9. 输出最佳参数 ========================
print("=" * 60)
print("最佳超参数：")
print(grid_search.best_params_)
print(f"最佳交叉验证 F1 分数：{grid_search.best_score_:.4f}")
print("=" * 60)

# ======================== 10. 在验证集上评估 ========================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("验证集评估结果：")
print(f"  准确率 (Accuracy) : {acc:.4f}")
print(f"  召回率 (Recall)   : {rec:.4f}")
print(f"  精确率 (Precision): {pre:.4f}")
print(f"  F1 分数 (F1-score): {f1:.4f}")
print(f"  AUC               : {auc:.4f}")
print("=" * 60)