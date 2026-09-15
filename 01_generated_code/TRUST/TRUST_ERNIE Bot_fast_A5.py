import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ========================== 1. 数据加载 ==========================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征与目标
X = df.iloc[:, :-1]  # 所有特征列（最后一列之前）
y = df.iloc[:, -1]   # 目标列 TRUST

# ========================== 2. 目标变量转换（二分类：TRUST >= 16） ==========================
y_binary = (y >= 16).astype(int)

# ========================== 3. 定义列类型 ==========================
cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
cat_ordinal_cols = ['TPPA']                      # 序数编码
num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# ========================== 4. 构建预处理管道 ==========================
# 连续变量：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码
categorical_onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数编码
categorical_ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat_onehot', categorical_onehot_transformer, cat_onehot_cols),
        ('cat_ordinal', categorical_ordinal_transformer, cat_ordinal_cols)
    ]
)

# ========================== 5. 划分训练集和测试集 ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ========================== 6. 构建包含 SMOTE 的完整管道 ==========================
smote = SMOTE(random_state=42)

rf_clf = RandomForestClassifier(random_state=42, n_jobs=1)

# 注意：SMOTE 放在预处理之后、模型之前
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', rf_clf)
])

# ========================== 7. 超参数调优（GridSearchCV） ==========================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth':[10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

grid_search.fit(X_train, y_train)

# ========================== 8. 最佳模型预测 ==========================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# ========================== 9. 模型评估 ==========================
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 50)
print("随机森林分类模型评估结果（SMOTE + 二分类 TRUST>=16）")
print("=" * 50)
print(f"最佳超参数: {grid_search.best_params_}")
print("-" * 50)
print(f"准确率  (Accuracy):  {accuracy:.4f}")
print(f"召回率  (Recall):    {recall:.4f}")
print(f"精确率  (Precision): {precision:.4f}")
print(f"F1分数  (F1-score):  {f1:.4f}")
print(f"AUC:                 {auc:.4f}")
print("=" * 50)
