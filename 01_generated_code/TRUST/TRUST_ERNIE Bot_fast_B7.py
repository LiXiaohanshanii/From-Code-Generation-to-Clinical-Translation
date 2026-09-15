import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
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
X = data.iloc[:, :-1]          # 所有特征列
y = data.iloc[:, -1]           # 目标列 TRUST

# 二分类标签：≥16 为 1，否则为 0
y = (y >= 16).astype(int)

# ============ 3. 定义列类型 ============
cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
cat_ordinal_cols = ['TPPA']                       # 序数编码
cont_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# ============ 4. 构建预处理管道 ============
# 连续变量：中位数填充
cont_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量：独热编码
cat_onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# TPPA：序数编码（无需填充，因无缺失值）
cat_ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', cont_transformer, cont_cols),
        ('cat_onehot', cat_onehot_transformer, cat_onehot_cols),
        ('cat_ordinal', cat_ordinal_transformer, cat_ordinal_cols)
    ]
)

# ============ 5. 构建完整管道（含SMOTE + 随机森林） ============
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=1))
])

# ============ 6. 定义超参数网格 ============
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [1],
    'clf__class_weight': ['balanced']
}

# ============ 7. 划分训练集与测试集 ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============ 8. 网格搜索调优 ============
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

# ============ 9. 输出最佳参数 ============
print("=" * 60)
print("最佳超参数：")
print(grid_search.best_params_)
print("=" * 60)

# ============ 10. 在测试集上评估 ============
y_pred = grid_search.predict(X_test)
y_prob = grid_search.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n模型评估结果：")
print(f"  Accuracy  (准确率)  : {acc:.4f}")
print(f"  Recall    (召回率)  : {rec:.4f}")
print(f"  Precision (精确率)  : {pre:.4f}")
print(f"  F1-score  (F1分数)  : {f1:.4f}")
print(f"  AUC       (AUC面积) : {auc:.4f}")
print("=" * 60)
