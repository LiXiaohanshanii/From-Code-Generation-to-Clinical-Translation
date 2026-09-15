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

# 2. 分离特征和目标列（目标列是最后一列TRUST）
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. 将目标列转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# 4. 定义特征列
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ordinal_cols = ['TPPA']                     # 序数编码
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# 5. 预处理流水线
# 连续变量：中位数填充
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量 - 独热编码（处理未知类别）
onehot_transformer = OneHotEncoder(handle_unknown='ignore')

# 分类变量 - 序数编码（TPPA）
# 注意：TPPA通常为滴度值，按顺序编码，假设可能的值从小到大排列
# 您可以根据实际数据调整categories参数，此处自动从数据中提取唯一值排序
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat_onehot', onehot_transformer, categorical_onehot_cols),
        ('cat_ordinal', ordinal_transformer, categorical_ordinal_cols)
    ]
)

# 6. 随机森林模型
rf_model = RandomForestClassifier(random_state=42, n_jobs=1)  # n_jobs=1 禁止多进程

# 7. 构建完整流水线（预处理 -> SMOTE -> 分类器）
# 注意：SMOTE需要在预处理之后，因为编码会改变特征空间
# 但由于preprocessor输出是numpy数组，我们先用preprocessor转换数据，再手动应用SMOTE
# 更规范的方法是在流水线中先预处理再SMOTE，但需要特殊处理。此处采用简单可靠的两阶段法

# 步骤A: 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# 步骤B: 仅对训练集拟合预处理器
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

# 步骤C: 应用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pre, y_train)

# 8. 超参数调优
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

# 不使用多进程（n_jobs=1）
rf_tuned = RandomForestClassifier(random_state=42, n_jobs=1)

grid_search = GridSearchCV(
    estimator=rf_tuned,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',  # 使用AUC作为调优指标
    n_jobs=1,           # 禁止多进程
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

# 最佳模型
best_rf = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")

# 9. 模型评估
y_pred = best_rf.predict(X_test_pre)
y_pred_proba = best_rf.predict_proba(X_test_pre)[:, 1]  # 正类的概率

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n模型评估结果:")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC: {auc:.4f}")