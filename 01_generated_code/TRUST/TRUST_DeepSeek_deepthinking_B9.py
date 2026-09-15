import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------- 1. 读取数据 --------------------
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征和目标（最后一列为TRUST）
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 将目标转换为二分类：>=16 记为 1，否则为 0
y_binary = (y >= 16).astype(int)

# -------------------- 2. 定义特征列 --------------------
# 分类变量（独热编码）
cat_features = ['SEX', 'DEPT', 'DIAGNOSIS']
# 有序分类变量（序数编码）
ord_features = ['TPPA']
# 连续变量（中位数填充）
num_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# -------------------- 3. 构建预处理流水线 --------------------
# 数值列：中位数填充
num_transformer = SimpleImputer(strategy='median')

# 分类列：独热编码（忽略未知类别）
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# 有序分类列：序数编码（未知值编码为 -1）
ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
        ('ord', ord_transformer, ord_features)
    ])

# -------------------- 4. 划分训练集和测试集（分层抽样） --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# -------------------- 5. 构建包含SMOTE和随机森林的流水线 --------------------
# 使用 imblearn 的 Pipeline 确保 SMOTE 仅在训练折内应用
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))  # 单进程
])

# -------------------- 6. 超参数网格 --------------------
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# -------------------- 7. GridSearchCV 调优（单进程，AUC为评分） --------------------
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,          # 不使用多进程
    verbose=1
)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters:", grid_search.best_params_)

# -------------------- 8. 在测试集上评估最佳模型 --------------------
best_model = grid_search.best_estimator_

# 预测类别和概率
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# 打印结果
print("\n===== 模型评估指标 =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {roc_auc:.4f}")