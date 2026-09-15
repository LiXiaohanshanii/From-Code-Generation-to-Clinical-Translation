# -*- coding: utf-8 -*-
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
import warnings

warnings.filterwarnings('ignore')

# ------------------------------
# 1. 读取数据
# ------------------------------
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列是最后一列，名称为'TRUST'
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 将目标转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# ------------------------------
# 2. 定义特征列
# ------------------------------
# 分类变量（独热编码）
cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']   # 注意：原始数据列为DIAGNOSIS
# 分类变量（序数编码）
cat_ordinal_cols = ['TPPA']
# 连续变量（需中位数填充）
cont_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ------------------------------
# 3. 构建预处理流水线
# ------------------------------
# 独热编码（处理未知类别）
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# 序数编码
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# 中位数填充
median_imputer = SimpleImputer(strategy='median')

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', onehot_encoder, cat_onehot_cols),
        ('ordinal', ordinal_encoder, cat_ordinal_cols),
        ('cont', median_imputer, cont_cols)
    ],
    remainder='drop'  # 丢弃未指定的列（如有）
)

# ------------------------------
# 4. 划分训练集和测试集
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ------------------------------
# 5. 构建包含SMOTE和随机森林的流水线
# ------------------------------
# 随机森林基础参数（固定部分）
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',   # 用户指定
    max_depth=10,              # 固定
    min_samples_split=2,       # 固定
    min_samples_leaf=1         # 固定
)

# 使用 imblearn 的 Pipeline 确保 SMOTE 在交叉验证的每一折内执行
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', rf)
])

# 超参数网格（仅 n_estimators 变化，其余固定）
param_grid = {
    'clf__n_estimators': [100, 200]
}

# GridSearchCV，使用 ROC-AUC 作为调优指标，单进程运行
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,                # 不使用多进程
    verbose=1
)

# ------------------------------
# 6. 训练模型（包含交叉验证调优）
# ------------------------------
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)
print("最佳交叉验证 ROC-AUC: {:.4f}".format(grid_search.best_score_))

# ------------------------------
# 7. 在测试集上评估最佳模型
# ------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# 打印评估结果
print("\n===== 测试集评估结果 =====")
print(f"准确率 (Accuracy)  : {accuracy:.4f}")
print(f"召回率 (Recall)    : {recall:.4f}")
print(f"精确率 (Precision) : {precision:.4f}")
print(f"F1 分数 (F1-score) : {f1:.4f}")
print(f"AUC 分数           : {auc:.4f}")