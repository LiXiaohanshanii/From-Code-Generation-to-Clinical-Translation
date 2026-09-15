import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------
# 1. 加载数据
# ------------------------------
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列名为 'TRUST'，假设为最后一列
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 将目标转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# ------------------------------
# 2. 定义特征列
# ------------------------------
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal_cols = ['TPPA']                     # 序数编码
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量（中位数填充）

# ------------------------------
# 3. 预处理流水线
# ------------------------------
# 3.1 连续变量：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 3.2 独热编码（处理未知类别）
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 3.3 序数编码（TPPA 假设为有序类别，如 '1', '2', '4', '8' 等）
# 注意：需根据实际数据确定类别顺序。这里提取唯一值并排序（转为数值再排序）
tppa_unique = sorted(df[categorical_ordinal_cols[0]].dropna().unique(),
                     key=lambda x: float(x) if str(x).replace('.','',1).isdigit() else str(x))
ordinal_categories = [tppa_unique]
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=ordinal_categories))
])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ])

# ------------------------------
# 4. 划分训练集和测试集
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ------------------------------
# 5. 定义包含预处理的最终模型流水线（仅用于后处理）
# ------------------------------
# 注意：SMOTE 需要在预处理之后、模型之前应用，因此不能直接放在同一个 Pipeline 中（因为 Pipeline 不支持中间 resample）。
# 我们将分别处理：先预处理训练集 -> SMOTE -> 训练模型，测试集只进行预处理。

# 5.1 对训练集进行预处理（fit_transform）
X_train_processed = preprocessor.fit_transform(X_train)
# 对测试集仅 transform
X_test_processed = preprocessor.transform(X_test)

# 5.2 应用 SMOTE 解决类别不平衡
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# ------------------------------
# 6. 超参数调优（不使用多进程）
# ------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc',          # 以 AUC 为调优指标
    cv=5,
    n_jobs=1,                   # 不使用多进程
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

best_rf = grid_search.best_estimator_

# ------------------------------
# 7. 模型评估
# ------------------------------
y_pred = best_rf.predict(X_test_processed)
y_pred_proba = best_rf.predict_proba(X_test_processed)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 输出结果
print("随机森林分类模型评估结果（目标：TRUST >= 16）")
print(f"最佳参数: {grid_search.best_params_}")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC (ROC曲线下面积): {auc:.4f}")