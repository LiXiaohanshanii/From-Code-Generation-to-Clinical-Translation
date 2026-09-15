import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ------------------------------
# 1. 读取数据
# ------------------------------
data = pd.read_csv('train_data.csv', encoding='utf-8')
print("数据集形状:", data.shape)

# 最后一列为目标变量 TRUST
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 二分类目标：是否 >= 16
y_binary = (y >= 16).astype(int)
print("正类样本数（TRUST >= 16）:", y_binary.sum())
print("负类样本数（TRUST < 16）:", (len(y_binary) - y_binary.sum()))

# ------------------------------
# 2. 划分训练集和测试集（分层抽样）
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ------------------------------
# 3. 定义特征列
# ------------------------------
# 分类变量：SEX, DEPT, DIAGNOSIS, TPPA
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ordinal_cols = ['TPPA']                     # 序数编码
# 连续变量（存在缺失值需中位数填充）
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ------------------------------
# 4. 构建预处理管道（填充 + 编码）
# ------------------------------
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量独热编码（忽略未知类别）
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 分类变量序数编码（默认按类别出现顺序赋值）
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ]
)

# ------------------------------
# 5. 构建完整管道（预处理 + SMOTE + 随机森林）
# ------------------------------
# 基础随机森林（参数将在网格搜索中覆盖）
rf = RandomForestClassifier(random_state=42, n_jobs=1)  # 禁用多进程

# 使用 imblearn 的 Pipeline 确保 SMOTE 只在训练折内执行
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])

# ------------------------------
# 6. 超参数网格（不使用多进程）
# ------------------------------
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# GridSearchCV，使用 AUC 作为调优指标，禁用多进程
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc',
    n_jobs=1, verbose=1, refit=True
)

# ------------------------------
# 7. 训练模型
# ------------------------------
print("\n开始网格搜索...")
grid_search.fit(X_train, y_train)
print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证 AUC: {:.4f}".format(grid_search.best_score_))

# ------------------------------
# 8. 测试集评估
# ------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n========== 模型评估结果 ==========")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1 分数 (F1-score): {f1:.4f}")
print(f"AUC 分数:           {auc:.4f}")