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

# ------------------------------
# 1. 读取数据
# ------------------------------
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列名为 'TRUST'，位于最后一列
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 将目标列转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# ------------------------------
# 2. 定义列名（根据题目描述）
# ------------------------------
# 分类变量（名义变量：独热编码）
categorical_nominal_cols = ['SEX', 'DEPT', 'DIAGNOSIS']

# 分类变量（有序变量：序数编码）
categorical_ordinal_cols = ['TPPA']

# 连续变量（使用中位数填充）
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 注意：题目中 DIAGNOSIS 拼写为 DIAGONSIS，此处按实际列名处理，请确保 CSV 中列名一致
# 若实际列名为 'DIAGNOSIS'，则无需修改；若为 'DIAGONSIS'，请改为下方：
# categorical_nominal_cols = ['SEX', 'DEPT', 'DIAGONSIS']

# ------------------------------
# 3. 预处理流水线（无缺失值填充）
# ------------------------------
# 名义变量独热编码（handle_unknown='ignore' 防止验证集出现未知类别）
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# 有序变量序数编码（需指定类别顺序，若无明确顺序则按出现顺序）
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# 数值变量中位数填充
numeric_imputer = SimpleImputer(strategy='median')

# 构建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_imputer, numeric_cols),
        ('nominal', onehot_encoder, categorical_nominal_cols),
        ('ordinal', ordinal_encoder, categorical_ordinal_cols)
    ],
    remainder='drop'  # 丢弃未指定的列（如有其他列）
)

# ------------------------------
# 4. 划分训练集和测试集
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ------------------------------
# 5. 构建带 SMOTE 的流水线
# ------------------------------
# 先填充/编码，再 SMOTE，最后随机森林
# 注意：SMOTE 应在预处理之后、分类器之前进行
base_classifier = RandomForestClassifier(random_state=42)

# 定义超参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用 imblearn 的 Pipeline 确保 SMOTE 在交叉验证中只作用于训练折
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', base_classifier)
])

# ------------------------------
# 6. 超参数调优（不使用多进程）
# ------------------------------
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',  # 以 AUC 作为调优指标
    n_jobs=1,           # 不使用多进程
    verbose=1
)

grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation AUC: {:.4f}".format(grid_search.best_score_))

# ------------------------------
# 7. 在测试集上评估模型
# ------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n===== Test Set Evaluation =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")