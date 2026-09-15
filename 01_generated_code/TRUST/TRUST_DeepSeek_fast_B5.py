import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. 读取数据
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和目标列（目标列为最后一列）
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. 将目标列转换为二分类：TRUST >= 16 为 1，否则为 0
y_binary = (y >= 16).astype(int)

# 4. 定义特征列（根据实际列名确认）
# 假设原始列顺序为：SEX, AGE, DEPT, DIAGNOSIS, TPPA, TP, HIV, WBC, RBC, PLT, NC, LY, NLR, TRUST
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                     # 序数编码
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 5. 预处理步骤
# 数值特征：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码分类变量
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # 避免多重共线性
])

# 序数编码分类变量（TPPA）
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('onehot', onehot_transformer, categorical_onehot),
        ('ordinal', ordinal_transformer, categorical_ordinal)
    ])

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 7. 构建包含SMOTE和随机森林的Pipeline（先SMOTE再训练分类器）
rf_model = RandomForestClassifier(random_state=42, n_jobs=1)  # 禁用多进程
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# 8. 超参数调优（GridSearchCV）
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc',
    n_jobs=1, verbose=1  # 禁用多进程
)

# 9. 训练模型
grid_search.fit(X_train, y_train)

# 10. 最佳模型
best_model = grid_search.best_estimator_
print("最佳参数组合:", grid_search.best_params_)

# 11. 预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 12. 评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n模型评估结果（测试集）:")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC曲线下面积 (AUC): {auc:.4f}")