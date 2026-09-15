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

# 2. 分离特征和目标列
X = df.iloc[:, :-1]  # 除了最后一列都是特征
y = df.iloc[:, -1]   # 最后一列是TRUST

# 3. 将目标列转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# 4. 定义列名
categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_cols_ordinal = ['TPPA']                     # 序数编码
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# 5. 创建预处理管道
# 连续变量：中位数填充
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量-独热编码
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # 避免多重共线性
])

# 分类变量-序数编码 (TPPA假设为有序类别，如1,2,4,8等，按数值顺序编码)
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

# 组合预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat_onehot', onehot_transformer, categorical_cols_onehot),
        ('cat_ordinal', ordinal_transformer, categorical_cols_ordinal)
    ])

# 6. 分割数据集（先分割再应用SMOTE，避免数据泄露）
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 7. 创建包含预处理、SMOTE和随机森林的完整管道
# 注意：SMOTE需要在预处理之后应用，因为需要数值型特征
rf_model = RandomForestClassifier(random_state=42, n_jobs=1)  # 不使用多进程

# 构建管道（预处理 -> SMOTE -> 分类器）
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
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',  # 使用AUC作为调优指标
    n_jobs=1,           # 不使用多进程
    verbose=1
)

# 9. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)
print("训练完成。")

# 10. 最佳模型
best_model = grid_search.best_estimator_
print(f"最佳参数组合: {grid_search.best_params_}")

# 11. 在测试集上进行预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 正类的概率

# 12. 评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 13. 输出结果
print("\n========== 模型评估结果 ==========")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("===================================\n")