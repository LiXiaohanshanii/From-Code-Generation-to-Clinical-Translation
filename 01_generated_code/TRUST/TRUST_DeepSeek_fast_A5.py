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

# 1. 加载数据
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列是最后一列 'TRUST'
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. 将目标列转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# 3. 定义特征列
categorical_nominal = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ordinal = ['TPPA']                      # 序数编码
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 4. 预处理：分类变量无缺失值，连续变量用中位数填充
# 注意：TPPA虽然是分类，但按题目要求使用OrdinalEncoder，数值本身可能已有序数含义，无需处理缺失（题目说无缺失）
# 为确保鲁棒，对数值列仍然加中位数填充

# 创建预处理器
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # 避免多重共线性
])

categorical_ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('nom', categorical_nominal_transformer, categorical_nominal),
        ('ord', categorical_ordinal_transformer, categorical_ordinal)
    ])

# 5. 构建包含SMOTE和随机森林的Pipeline
rf_model = RandomForestClassifier(random_state=42, n_jobs=1)  # 不使用多进程

# 注意：SMOTE需要在数据标准化/编码之后，且在模型之前，但必须在训练集上应用，不能泄漏到验证集
# 使用imblearn的Pipeline确保SMOTE只应用于训练折叠
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 7. 超参数调优（不使用多进程）
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc',
    n_jobs=1, refit=True, verbose=1
)

# 8. 训练模型
grid_search.fit(X_train, y_train)

# 9. 最佳模型评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 10. 打印结果
print("最佳参数组合:", grid_search.best_params_)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC: {auc:.4f}")