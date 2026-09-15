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

# ============ 1. 加载数据 ============
data = pd.read_csv('train_data.csv', encoding='utf-8')

# ============ 2. 分离特征与目标 ============
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
target_col = 'TRUST'

X = data[feature_cols]
y = data[target_col]

# ============ 3. 构建二分类目标：TRUST >= 16 → 1, else → 0 ============
y_binary = (y >= 16).astype(int)

# ============ 4. 定义分类变量与连续变量 ============
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal_cols = ['TPPA']
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ============ 5. 数据预处理：分类变量编码 + 连续变量缺失值填充 ============
categorical_onehot_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_ordinal_transformer = OrdinalEncoder()
continuous_transformer = SimpleImputer(strategy='median')

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', categorical_onehot_transformer, categorical_onehot_cols),
        ('ordinal', categorical_ordinal_transformer, categorical_ordinal_cols),
        ('continuous', continuous_transformer, continuous_cols)
    ],
    remainder='drop'
)

# ============ 6. 划分训练集与测试集 ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ============ 7. 构建含SMOTE的完整Pipeline ============
rf = RandomForestClassifier(random_state=42, n_jobs=1)

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])

# ============ 8. 网格搜索超参数调优 ============
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ============ 9. 输出最佳参数 ============
print("=" * 50)
print("最佳超参数：")
print(grid_search.best_params_)
print("=" * 50)

# ============ 10. 使用最佳模型预测 ============
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# ============ 11. 模型评估 ============
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("模型评估结果：")
print(f"  准确率 (Accuracy) : {accuracy:.4f}")
print(f"  召回率 (Recall)   : {recall:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  F1分数 (F1-score) : {f1:.4f}")
print(f"  AUC              : {auc:.4f}")
print("=" * 50)
