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

# ==================== 1. 加载数据 ====================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列（最后一列）
X = data.iloc[:, :-1]
y_raw = data.iloc[:, -1]   # TRUST原始值（1,2,4,8,...）

# 构造二分类目标：>=16 为正类 (1)，<16 为负类 (0)
y = (y_raw >= 16).astype(int)

# ==================== 2. 定义特征列 ====================
# 分类变量（OneHot编码）
onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']   # 注意原题拼写 DIAGNOSIS
# 分类变量（序数编码）
ordinal_cols = ['TPPA']
# 连续变量（需要中位数填充）
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 检查所有列是否存在于X中（防止拼写错误）
for col in onehot_cols + ordinal_cols + numeric_cols:
    if col not in X.columns:
        raise ValueError(f"列 '{col}' 不存在于数据中，请检查列名。")

# ==================== 3. 划分训练集和测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==================== 4. 定义预处理流程 ====================
# 数值特征：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码分类特征
onehot_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# 序数编码分类特征（TPPA）
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat_onehot', onehot_transformer, onehot_cols),
        ('cat_ordinal', ordinal_transformer, ordinal_cols)
    ]
)

# ==================== 5. 构建带SMOTE和随机森林的Pipeline ====================
# 注意：GridSearchCV 需要组合预处理 + SMOTE + 分类器
# 先对训练集做预处理，然后SMOTE，再训练随机森林
# 但SMOTE必须在数值特征上运行，预处理后全是数值，没问题。

# 定义随机森林参数网格（不使用多进程）
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 构建完整Pipeline：预处理 -> SMOTE -> 随机森林
# 注意：SMOTE 需要在预处理之后、分类器之前，并且只能应用于训练集
# 这里使用 imblearn 的 Pipeline 以确保 SMOTE 只作用于训练折叠
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ==================== 6. 网格搜索（不使用多进程） ====================
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,                # 禁用多进程
    verbose=1
)

grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)
print("最佳交叉验证 AUC: {:.4f}".format(grid_search.best_score_))

# ==================== 7. 在测试集上评估最佳模型 ====================
best_model = grid_search.best_estimator_

# 预测类别和概率
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]   # 正类概率

# 计算指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n========== 测试集评估结果 ==========")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC (ROC曲线下面积): {auc:.4f}")