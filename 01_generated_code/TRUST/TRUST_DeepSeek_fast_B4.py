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

# 1. 加载数据
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和目标
X = df.iloc[:, :-1]   # 除最后一列外的所有列
y = df.iloc[:, -1]    # 最后一列 TRUST

# 3. 目标变量转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# 4. 定义列名
# 根据描述：SEX, DEPT, DIAGNOSIS, TPPA 是分类变量
# 注意：描述中写的是 DIAGNOSIS，而变量列表中是 DIAGNOSIS，保持一致
categorical_nominal_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal_cols = ['TPPA']
# 连续变量（含可能需要填充缺失值的）
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 5. 数据拆分（先拆分，后做SMOTE避免数据泄露）
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 6. 创建预处理步骤
# 6.1 数值特征：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 6.2 名义分类特征：独热编码
nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# 6.3 序数分类特征：序数编码（TPPA）
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 6.4 组合预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('nom', nominal_transformer, categorical_nominal_cols),
        ('ord', ordinal_transformer, categorical_ordinal_cols)
    ]
)

# 7. 构建包含SMOTE和随机森林的Pipeline
rf_model = RandomForestClassifier(random_state=42)

# 使用ImbPipeline以保证SMOTE只在训练集上拟合
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# 8. 定义超参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 9. 网格搜索（不启用多进程）
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',  # 使用AUC作为调优指标
    n_jobs=1,           # 不使用多进程
    verbose=1
)

# 10. 训练模型
print("开始训练和调优...")
grid_search.fit(X_train, y_train)

# 11. 最佳模型
best_model = grid_search.best_estimator_
print(f"最佳参数组合: {grid_search.best_params_}")

# 12. 在测试集上预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 13. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 14. 输出结果
print("\n========== 模型评估结果 ==========")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")