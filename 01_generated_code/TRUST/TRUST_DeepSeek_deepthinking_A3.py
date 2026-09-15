import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
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
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征和目标列（目标列在最后一列）
X = data.iloc[:, :-1]
y_raw = data.iloc[:, -1]          # TRUST列，滴度值（1,2,4,8,...）

# 创建二分类目标：是否 >= 16
y = (y_raw >= 16).astype(int)

# ------------------------------
# 2. 定义特征列
# ------------------------------
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码的分类变量
categorical_ordinal_cols = ['TPPA']                      # 序数编码的分类变量
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 确保所有列名存在于数据中
assert all(col in X.columns for col in categorical_onehot_cols), "独热编码列缺失"
assert all(col in X.columns for col in categorical_ordinal_cols), "序数编码列缺失"
assert all(col in X.columns for col in continuous_cols), "连续变量列缺失"

# ------------------------------
# 3. 预处理转换器
# ------------------------------
# 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量（独热编码）：无缺失，直接编码（drop='first'避免多重共线性可选）
onehot_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop=None, sparse_output=False))
])

# 分类变量（序数编码）：TPPA 按出现的顺序赋予整数（也可自定义顺序，此处让编码器自动推断）
ordinal_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 合并所有预处理步骤
preprocessor = ColumnTransformer(transformers=[
    ('cont', continuous_transformer, continuous_cols),
    ('cat_onehot', onehot_transformer, categorical_onehot_cols),
    ('cat_ordinal', ordinal_transformer, categorical_ordinal_cols)
])

# ------------------------------
# 4. 划分训练集和测试集（先划分，后做SMOTE）
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 在训练集上拟合预处理器，并转换训练集和测试集
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

# ------------------------------
# 5. SMOTE 过采样（仅在训练集上）
# ------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_pre, y_train)

# ------------------------------
# 6. 随机森林模型及超参数调优（不使用多进程）
# ------------------------------
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

# 使用分层K折交叉验证，n_jobs=1 禁用多进程
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',          # 以AUC作为调优指标
    n_jobs=1,                   # 不使用多进程
    verbose=1
)

# 在过采样后的训练集上执行网格搜索
grid_search.fit(X_train_res, y_train_res)

# 获取最佳模型
best_rf = grid_search.best_estimator_
print(f"最佳参数组合: {grid_search.best_params_}")

# ------------------------------
# 7. 在测试集上评估模型
# ------------------------------
y_pred = best_rf.predict(X_test_pre)
y_pred_proba = best_rf.predict_proba(X_test_pre)[:, 1]   # 正类的概率

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 输出评估指标
print("\n===== 模型评估结果 =====")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC (ROC曲线下面积): {auc:.4f}")