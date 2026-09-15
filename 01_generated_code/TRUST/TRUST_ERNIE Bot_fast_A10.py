import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 读取数据
# ============================================================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# ============================================================
# 2. 分离特征与目标
# ============================================================
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA',
                'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[feature_cols].copy()
y = df.iloc[:, -1].copy()  # 最后一列 TRUST

# ============================================================
# 3. 目标变量二分类转换：TRUST >= 16 为 1，否则为 0
# ============================================================
y = (y >= 16).astype(int)

# ============================================================
# 4. 定义分类变量与连续变量
# ============================================================
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                      # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ============================================================
# 5. 构建预处理管道
# ============================================================
# 连续变量：中位数填充
cont_imputer = SimpleImputer(strategy='median')

# 分类变量（独热编码）——无缺失值，不需要 imputer
cat_ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

# 分类变量（序数编码）——无缺失值，不需要 imputer
cat_ord = OrdinalEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', cont_imputer, continuous),
        ('ohe', cat_ohe, categorical_onehot),
        ('ord', cat_ord, categorical_ordinal)
    ],
    remainder='drop'
)

# ============================================================
# 6. 划分训练集与测试集
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 7. 完整 Pipeline：预处理 + SMOTE + 随机森林
# ============================================================
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

# ============================================================
# 8. 超参数网格与 GridSearchCV 调优（不使用多进程）
# ============================================================
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [1],
    'clf__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,           # 不使用多进程
    verbose=1
)

grid_search.fit(X_train, y_train)

# ============================================================
# 9. 输出最优参数
# ============================================================
print("=" * 60)
print("最优超参数组合：")
print(grid_search.best_params_)
print(f"交叉验证最佳 AUC: {grid_search.best_score_:.4f}")
print("=" * 60)

# ============================================================
# 10. 使用最优模型在测试集上预测并评估
# ============================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n模型评估结果（测试集）：")
print("-" * 40)
print(f"准确率 (Accuracy)  : {acc:.4f}")
print(f"召回率 (Recall)    : {rec:.4f}")
print(f"精确率 (Precision) : {pre:.4f}")
print(f"F1 分数 (F1-score) : {f1:.4f}")
print(f"AUC               : {auc:.4f}")
print("-" * 40)
