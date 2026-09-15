import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ============================================================
# 2. 定义特征列和目标列
# ============================================================
feature_cols = [
    'SER-T', 'CL', 'GLU', 'Ink staining', 'Color',
    'Protein', 'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS'
]
target_col = 'CSF-T'

X = df[feature_cols]
y = df[target_col]

# ============================================================
# 3. 目标变量二分类处理: >=20 为1，<20 为0
# ============================================================
y_binary = (y >= 20).astype(int)

# ============================================================
# 4. 定义变量类型
# ============================================================
# 分类变量 - 独热编码
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
# 分类变量 - 序数编码
ordinal_cols = ['SER-T', 'Ink staining']
# 连续变量
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# ============================================================
# 5. 缺失值处理 & 编码构建（使用 Pipeline + ColumnTransformer）
# ============================================================
# 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 序数编码变量
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('ordinal', OrdinalEncoder())
])

# 独热编码变量
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# ColumnTransformer 组合
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('onehot', onehot_transformer, onehot_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ============================================================
# 6. 划分训练集和测试集
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ============================================================
# 7. 构建完整 Pipeline（含 SMOTE + 随机森林）
# ============================================================
# 定义随机森林分类器
rf_clf = RandomForestClassifier(
    random_state=42,
    n_jobs=1  # 不使用多进程
)

# 使用 ImbPipeline 将 SMOTE 与模型串联
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_clf)
])

# ============================================================
# 8. 超参数调优（GridSearchCV）
# ============================================================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,       # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

# ============================================================
# 9. 输出最佳参数
# ============================================================
print("=" * 60)
print("最佳超参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证 F1 分数: {grid_search.best_score_:.4f}")
print("=" * 60)

# ============================================================
# 10. 在测试集上评估
# ============================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("模型评估结果（测试集）:")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1 分数 (F1-score): {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 60)