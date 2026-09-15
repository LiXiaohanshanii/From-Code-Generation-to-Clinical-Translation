import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ------------------------------
# 1. 读取数据
# ------------------------------
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列：最后一列 'TRUST'
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 将目标转换为二分类：>=16 为1，其余为0
y_binary = (y >= 16).astype(int)

# ------------------------------
# 2. 划分训练集和测试集（分层抽样）
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# ------------------------------
# 3. 定义特征列
# ------------------------------
# 分类变量（注意：DIAGNOSIS 与描述一致，预处理要求中拼写为 DIAGONSIS，但实际特征名为 DIAGNOSIS）
categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_cols_ordinal = ['TPPA']   # 使用序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ------------------------------
# 4. 构建预处理管道
# ------------------------------
# 4.1 连续变量：中位数填充
continuous_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# 4.2 独热编码变量
onehot_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 4.3 序数编码变量（TPPA）
ordinal_transformer = Pipeline([
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合到 ColumnTransformer
preprocessor = ColumnTransformer([
    ('cont', continuous_transformer, continuous_cols),
    ('cat_onehot', onehot_transformer, categorical_cols_onehot),
    ('cat_ordinal', ordinal_transformer, categorical_cols_ordinal)
])

# ------------------------------
# 5. 构建包含 SMOTE 和随机森林的完整管道（用于网格搜索）
# ------------------------------
# 注意：SMOTE 只应用于训练集，且必须在预处理之后
# 使用 imblearn 的 Pipeline 确保 SMOTE 在交叉验证中正确应用
rf = RandomForestClassifier(random_state=42, n_jobs=1)  # 禁用多进程

# 参数网格（按题目要求）
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 整体管道：预处理 -> SMOTE -> 分类器
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])

# 网格搜索（不使用多进程，n_jobs=1，评分使用 ROC AUC）
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,          # 禁用多进程
    verbose=1,
    refit=True
)

# ------------------------------
# 6. 训练模型（网格搜索自动进行交叉验证和SMOTE）
# ------------------------------
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)
print("最佳交叉验证 ROC AUC: {:.4f}".format(grid_search.best_score_))

# ------------------------------
# 7. 在测试集上评估
# ------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 计算指标
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n===== 测试集评估结果 =====")
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {pre:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC (ROC曲线下面积): {auc:.4f}")

# 可选：输出分类报告（略）