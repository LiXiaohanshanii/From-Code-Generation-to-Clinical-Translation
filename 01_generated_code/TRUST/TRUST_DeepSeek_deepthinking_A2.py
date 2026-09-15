# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 预测TRUST滴度是否≥16
适用于PyCharm 2025.2.3环境
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 设置随机种子，保证结果可复现
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==================== 1. 数据加载 ====================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征与目标（目标列是最后一列）
X = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]

# 将目标转换为二分类：TRUST >= 16 为正类(1)，否则为负类(0)
y = (y_raw >= 16).astype(int)

# ==================== 2. 特征列定义 ====================
# 分类变量（独热编码）
cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
# 分类变量（序数编码）
cat_ordinal_cols = ['TPPA']
# 连续变量（需填充中位数）
num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ==================== 3. 构建预处理管道 ====================
# 连续变量：中位数填充
num_transformer = SimpleImputer(strategy='median')

# 独热编码（handle_unknown='ignore' 避免测试集出现新类别）
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 序数编码（对TPPA，按出现顺序自动分配整数）
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat_onehot', onehot_transformer, cat_onehot_cols),
        ('cat_ordinal', ordinal_transformer, cat_ordinal_cols)
    ]
)

# ==================== 4. 数据拆分（分层抽样，保证类别比例） ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

# ==================== 5. 构建包含SMOTE和随机森林的管道 ====================
# 随机森林基础模型（固定部分参数）
rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=1,               # 不使用多进程
    max_depth=10,           # 参数网格中的固定值
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced'
)

# 完整管道：预处理 -> SMOTE -> 随机森林
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('rf', rf)
])

# ==================== 6. 超参数调优（网格搜索，不使用多进程） ====================
param_grid = {
    'rf__n_estimators': [100, 200]
    # 其他超参数已在rf中固定，无需在此重复
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,                # 5折交叉验证
    scoring='roc_auc',   # 使用AUC作为选择最优模型的指标
    n_jobs=1,            # 不使用多进程
    verbose=1
)

# 训练模型（自动进行预处理、SMOTE、交叉验证调参）
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)
print("最佳交叉验证AUC: {:.4f}".format(grid_search.best_score_))

# 获取最佳模型
best_model = grid_search.best_estimator_

# ==================== 7. 模型评估（测试集） ====================
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n========== 模型评估指标（测试集） ==========")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC (ROC曲线下面积): {auc:.4f}")