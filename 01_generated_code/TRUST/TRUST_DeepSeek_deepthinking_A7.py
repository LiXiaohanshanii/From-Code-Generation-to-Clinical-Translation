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

# ------------------------------
# 1. 加载数据
# ------------------------------
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列是最后一列，列名假设为 'TRUST'
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# 二分类目标：TRUST >= 16 为 1，否则 0
y_binary = (y >= 16).astype(int)

# 特征列（根据描述）
# 分类变量：SEX, DEPT, DIAGNOSIS, TPPA
# 连续变量：AGE, TP, HIV, WBC, RBC, PLT, NC, LY, NLR
categorical_cols = ['SEX', 'DEPT', 'DIAGNOSIS', 'TPPA']
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 确保数据中列名存在（检查）
assert all(col in X.columns for col in categorical_cols), "分类变量列名缺失"
assert all(col in X.columns for col in numeric_cols), "连续变量列名缺失"

# 分离分类变量中的独热编码变量和序数编码变量
onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
ordinal_cols = ['TPPA']                      # 序数编码

# ------------------------------
# 2. 数据预处理（定义转换器）
# ------------------------------
# 对于独热编码：处理分类变量（无缺失值）
onehot_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # drop='first'避免多重共线性
])

# 对于序数编码：TPPA（按类别顺序编码，默认按字母顺序，也可显式指定）
# 此处使用默认顺序，如果希望自定义顺序可设置categories参数
ordinal_transformer = Pipeline([
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 对于连续变量：中位数填充
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# 组合预处理步骤
preprocessor = ColumnTransformer([
    ('onehot', onehot_transformer, onehot_cols),
    ('ordinal', ordinal_transformer, ordinal_cols),
    ('numeric', numeric_transformer, numeric_cols)
])

# ------------------------------
# 3. 划分训练集和测试集（先拆分，再对训练集应用SMOTE）
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# ------------------------------
# 4. 构建含SMOTE和随机森林的Pipeline
# ------------------------------
# 注意：SMOTE只应用于训练集，因此我们将预处理+SMOTE+分类器封装在ImbPipeline中
# 这样在fit时，先预处理，然后SMOTE过采样，再训练分类器
rf_model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',   # 网格参数中已有，但这里先设置默认
    n_jobs=1                   # 单进程
)

# 构建完整的pipeline：预处理 -> SMOTE -> 随机森林
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# ------------------------------
# 5. 超参数调优（GridSearchCV）
# ------------------------------
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 注意：GridSearchCV默认使用多进程，设置n_jobs=1
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',          # 以AUC为优化指标
    n_jobs=1,                   # 单进程
    verbose=1
)

# 训练（自动应用SMOTE，因为SMOTE在pipeline中）
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)

# ------------------------------
# 6. 评估最佳模型
# ------------------------------
best_model = grid_search.best_estimator_

# 在测试集上预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]   # 正类概率

# 计算指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n========== 模型评估结果 ==========")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC (ROC曲线下面积): {auc:.4f}")