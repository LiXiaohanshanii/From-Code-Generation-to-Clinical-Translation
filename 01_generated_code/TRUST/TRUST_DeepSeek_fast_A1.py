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

# 1. 读取数据
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和目标列（目标列TRUST是最后一列）
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. 将目标转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# 4. 定义特征列
categorical_nominal = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ordinal = ['TPPA']                       # 序数编码
continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 5. 构建预处理流程
# 5.1 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 5.2 名义分类变量：独热编码
nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False))  # 避免多重共线性
])

# 5.3 序数分类变量：序数编码 (注意：TPPA值通常有序，如1,2,4,8...)
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_features),
        ('nom', nominal_transformer, categorical_nominal),
        ('ord', ordinal_transformer, categorical_ordinal)
    ])

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 7. 构建包含SMOTE和随机森林的Pipeline
rf_model = RandomForestClassifier(random_state=42, n_jobs=1)  # 不使用多进程

# 注意：SMOTE应在预处理之后、模型之前，且只应用于训练集
# 使用imblearn的Pipeline确保SMOTE只作用于训练折叠
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# 8. 超参数调优
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

grid_search.fit(X_train, y_train)

# 9. 最佳模型评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 计算指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 10. 打印结果
print("="*50)
print("随机森林分类模型评估结果")
print("="*50)
print(f"最佳参数: {grid_search.best_params_}")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("="*50)