import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# ===================== 1. 数据加载与目标变量构建 =====================
# 读取数据集，utf-8编码
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 定义特征列
categorical_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['TPPA']
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
feature_cols = categorical_cols + ordinal_cols + numerical_cols

# 构建二分类目标变量：TRUST >= 16 为阳性(1)，否则为阴性(0)
y = (df['TRUST'] >= 16).astype(int)
X = df[feature_cols].copy()

# 划分训练集和测试集（先划分再SMOTE，防止数据泄露）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 2. 数据预处理管道 =====================
# 分类变量：独热编码（One-Hot Encoding）
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数变量：TPPA序数编码（Ordinal Encoding）
# TPPA作为梅毒特异性抗体检测，通常有明确的滴度等级顺序
ord_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 连续变量：中位数填充缺失值
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 组合所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, categorical_cols),
        ('ord', ord_transformer, ordinal_cols),
        ('num', num_transformer, numerical_cols)
    ],
    remainder='drop'
)

# ===================== 3. 模型构建与超参数调优 =====================
# 定义随机森林及超参数网格（严格按需求设定）
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [1],
    'clf__class_weight': ['balanced']
}

# 使用imblearn的Pipeline将SMOTE集成到交叉验证中，避免测试集数据泄露
# n_jobs=1 明确禁用多进程
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='f1',          # 不平衡数据以F1为主要优化指标
    cv=5,
    n_jobs=1,              # 不使用多进程
    verbose=1
)

print("开始超参数调优...")
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ===================== 4. 模型评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n========== 测试集模型评估结果 ==========")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC曲线下面积:      {auc:.4f}")
print("=========================================")
