import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ------------------------------
# 1. 数据加载
# ------------------------------
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 特征与目标分离
X = data.iloc[:, :-1]  # 除最后一列外为特征
y = data.iloc[:, -1]   # 最后一列为目标列 TRUST

# 二分类目标：是否 >= 16
y_binary = (y >= 16).astype(int)

# 定义列名
categorical_nominal = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ordinal = ['TPPA']                        # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ------------------------------
# 2. 数据预处理
# ------------------------------
# 分类变量无缺失值，但连续变量可能有缺失值 -> 中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_onehot = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # 避免多重共线性
])

categorical_ordinal_trans = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous),
        ('cat_onehot', categorical_onehot, categorical_nominal),
        ('cat_ordinal', categorical_ordinal_trans, categorical_ordinal)
    ])

# ------------------------------
# 3. 划分训练集和测试集
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ------------------------------
# 4. 构建带SMOTE的Pipeline（先预处理，再SMOTE，最后分类器）
# ------------------------------
# 注意：SMOTE必须在预处理之后应用，因为需要数值特征
rf_base = RandomForestClassifier(random_state=42, n_jobs=1)  # 单进程

# 使用imblearn的Pipeline以正确处理SMOTE
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_base)
])

# ------------------------------
# 5. 超参数调优
# ------------------------------
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# GridSearchCV，单进程（n_jobs=1）
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc',
    n_jobs=1, verbose=1
)

grid_search.fit(X_train, y_train)

# ------------------------------
# 6. 评估最佳模型
# ------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("=== 随机森林分类模型评估结果 ===")
print(f"最佳参数: {grid_search.best_params_}")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:               {auc:.4f}")