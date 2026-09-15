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

# 2. 分离特征和目标
X = df.iloc[:, :-1]  # 除最后一列外为特征
y = df.iloc[:, -1]   # 最后一列TRUST为目标

# 3. 将目标转换为二分类：是否 ≥ 16
y_binary = (y >= 16).astype(int)

# 4. 定义特征列
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal_cols = ['TPPA']                     # 序数编码
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# 5. 预处理步骤
# 连续变量：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量：独热编码（处理未知类别）
onehot_transformer = OneHotEncoder(handle_unknown='ignore')

# 分类变量：序数编码（TPPA，假设滴度值有自然顺序）
# 注意：如果TPPA有缺失值，需先填充，但题目说明无缺失
ordinal_transformer = OrdinalEncoder()

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ])

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 7. 构建包含SMOTE的完整Pipeline
# 注意：SMOTE应在预处理之后、分类器之前进行
smote = SMOTE(random_state=42)
rf = RandomForestClassifier(random_state=42, n_jobs=1)  # 不使用多进程

# 先预处理，再SMOTE，再分类器
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', rf)
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
    n_jobs=1, verbose=1  # 不使用多进程
)

# 9. 训练模型
grid_search.fit(X_train, y_train)

# 10. 最佳模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 11. 评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 12. 输出结果
print("=" * 50)
print("随机森林模型评估结果（二分类：TRUST ≥ 16）")
print("=" * 50)
print(f"最佳参数: {grid_search.best_params_}")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("=" * 50)