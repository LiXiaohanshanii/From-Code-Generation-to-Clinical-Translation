# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# 设置随机种子，保证结果可重复
RANDOM_STATE = 42

# 1. 读取数据
df = pd.read_csv('train_data.csv', encoding='utf-8')
print("数据形状:", df.shape)
print("目标列分布:\n", df['TRUST'].value_counts().sort_index())

# 2. 分离特征和目标，并将目标转换为二分类（≥16 为 1，否则 0）
X = df.drop('TRUST', axis=1)
y = (df['TRUST'] >= 16).astype(int)
print("\n二分类后的目标分布:\n", y.value_counts())

# 3. 划分训练集和测试集（保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)
print(f"\n训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 4. 定义特征列
numeric_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']   # 使用独热编码
ordinal_features = ['TPPA']                     # 使用序数编码

# 5. 构建预处理器
# 数值特征：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码特征（稠密输出，便于SMOTE处理）
onehot_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数编码特征（按训练数据中出现的顺序编码，未知类别编码为-1）
ordinal_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 合并所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('onehot', onehot_transformer, onehot_features),
        ('ordinal', ordinal_transformer, ordinal_features)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# 6. 在训练集上拟合预处理器并转换，测试集仅转换
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n预处理后训练集特征形状:", X_train_processed.shape)

# 7. 应用SMOTE解决类别不平衡（仅对训练集）
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
print("\nSMOTE后训练集分布:\n", pd.Series(y_train_resampled).value_counts())

# 8. 定义随机森林模型（禁用多进程）
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)

# 9. 超参数网格（仅调整n_estimators，其余固定）
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

# 使用GridSearchCV进行交叉验证调优（禁用多进程，评分用AUC）
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,               # 5折交叉验证
    scoring='roc_auc',  # 以AUC作为调优指标
    n_jobs=1,           # 禁用多进程
    verbose=1
)

# 在重采样后的训练集上训练
grid_search.fit(X_train_resampled, y_train_resampled)

print("\n最佳参数组合:", grid_search.best_params_)
print("最佳交叉验证AUC: {:.4f}".format(grid_search.best_score_))

# 10. 使用最佳模型在测试集上评估
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_processed)
y_pred_proba = best_rf.predict_proba(X_test_processed)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 打印结果
print("\n====== 测试集评估结果 ======")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC曲线下面积:     {auc:.4f}")