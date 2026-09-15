import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. 加载数据
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征与目标（最后一列为TRUST）
X = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]

# 3. 将目标转换为二分类：是否 >=16
y = (y_raw >= 16).astype(int)

# 4. 定义特征列名（根据题目描述）
categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_cols_ordinal = ['TPPA']                     # 序数编码
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 5. 划分训练集和测试集（先划分，避免数据泄露）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. 定义预处理步骤（仅对连续变量填充中位数，分类变量不做填充）
# 连续变量中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码（处理未知类别，丢弃第一个避免共线性可选）
onehot_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数编码（假设类别按字母顺序或出现顺序）
ordinal_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 合并预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat_onehot', onehot_transformer, categorical_cols_onehot),
        ('cat_ordinal', ordinal_transformer, categorical_cols_ordinal)
    ]
)

# 7. 构建完整管道（先预处理，再SMOTE过采样，最后随机森林）
# 注意：SMOTE仅应用于训练集（管道内应用，保证交叉验证时每个fold正确过采样）
rf_model = RandomForestClassifier(random_state=42, n_jobs=1)  # 不使用多进程
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# 8. 超参数调优（使用GridSearchCV，n_jobs=1禁用多进程）
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=1, verbose=1
)
grid_search.fit(X_train, y_train)

# 9. 获取最佳模型
best_model = grid_search.best_estimator_
print("最佳参数组合:", grid_search.best_params_)

# 10. 在测试集上评估
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n模型在测试集上的表现：")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC: {auc:.4f}")