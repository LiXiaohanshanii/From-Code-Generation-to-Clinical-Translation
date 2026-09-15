import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. 读取数据
# ============================================================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# ============================================================
# 2. 分离特征与目标
# ============================================================
feature_cols = [
    'SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP',
    'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR'
]
target_col = 'TRUST'  # 最后一列

X = data[feature_cols]
y = data[target_col]

# ============================================================
# 3. 构造二分类目标：TRUST >= 16 → 1，否则 → 0
# ============================================================
y_binary = (y >= 16).astype(int)

# ============================================================
# 4. 划分训练集与测试集（分层抽样）
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

# ============================================================
# 5. 定义变量分组
# ============================================================
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                     # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC',
                   'PLT', 'NC', 'LY', 'NLR']      # 连续变量

# ============================================================
# 6. 构建预处理管道
# ============================================================
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 独热编码（drop='first'避免多重共线性，handle_unknown兼容未知类别）
onehot_transformer = OneHotEncoder(
    drop='first',
    sparse_output=False,
    handle_unknown='ignore'
)

# 序数编码
ordinal_transformer = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('onehot', onehot_transformer, categorical_onehot),
        ('ordinal', ordinal_transformer, categorical_ordinal)
    ],
    remainder='drop'
)

# ============================================================
# 7. 构建含 SMOTE + 随机森林的完整管道
# ============================================================
smote = SMOTE(random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=1           # 不使用多进程
)

model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', rf)
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
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,          # 不使用多进程
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n最佳超参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证 AUC: {grid_search.best_score_:.4f}")

# ============================================================
# 9. 在测试集上进行预测与评估
# ============================================================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# ============================================================
# 10. 输出评估结果
# ============================================================
print("\n" + "=" * 60)
print("        随机森林分类模型 — 测试集评估结果")
print("=" * 60)
print(f"  准确率 (Accuracy):  {accuracy:.4f}")
print(f"  召回率 (Recall):     {recall:.4f}")
print(f"  精确率 (Precision):  {precision:.4f}")
print(f"  F1分数 (F1-score):   {f1:.4f}")
print(f"  AUC:                 {auc:.4f}")
print("=" * 60)

# ============================================================
# 11. 输出测试集类别分布
# ============================================================
print(f"\n测试集实际分布: 0类={sum(y_test==0)}, 1类={sum(y_test==1)}")
print(f"测试集预测分布: 0类={sum(y_pred==0)}, 1类={sum(y_pred==1)}")
