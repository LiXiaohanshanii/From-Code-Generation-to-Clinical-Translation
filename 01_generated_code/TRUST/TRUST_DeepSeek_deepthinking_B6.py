import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. 读取数据
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和目标
# 目标列是最后一列 'TRUST'
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 3. 目标二分类：>=16 为正类 (1)，<16 为负类 (0)
y_binary = (y >= 16).astype(int)

# 4. 定义特征列
# 分类变量（注意：数据集中列名为 DIAGNOSIS，用户描述中误写为 DIAGONSIS）
cat_cols = ['SEX', 'DEPT', 'DIAGNOSIS', 'TPPA']
# 连续变量
num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 5. 划分训练集和测试集（保持原始分布，后续 SMOTE 仅作用于训练集）
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 6. 预处理流水线（用于训练集和测试集转换）
# 分类变量编码：SEX, DEPT, DIAGNOSIS 用独热，TPPA 用序数编码
# 注意：OrdinalEncoder 默认按类别出现顺序编码，此处无特定顺序要求
preprocessor = ColumnTransformer(
    transformers=[
        ('cat_onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), ['SEX', 'DEPT', 'DIAGNOSIS']),
        ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['TPPA']),
        ('num_impute', SimpleImputer(strategy='median'), num_cols)
    ],
    remainder='passthrough'  # 其他列（如果有）保留，但这里所有列都已覆盖
)

# 7. 构建包含预处理、SMOTE 和随机森林的流水线
# 注意：SMOTE 必须在预处理之后进行，但为了在交叉验证中正确处理，使用 imblearn 的 Pipeline
# 先预处理，再 SMOTE，再分类器
model = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 8. 超参数网格（仅调优随机森林部分）
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 9. 网格搜索（不启用多进程）
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='roc_auc',  # 以 AUC 为优化目标
    n_jobs=1,           # 不使用多进程
    verbose=1
)

# 10. 训练模型
grid_search.fit(X_train, y_train)

# 11. 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)

# 12. 在测试集上预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 13. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# 14. 打印结果
print("\n===== 模型评估结果 =====")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1 分数 (F1-score): {f1:.4f}")
print(f"AUC 分数:           {auc:.4f}")