import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ===================== 1. 数据加载 =====================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# ===================== 2. 特征与目标分离 =====================
# 目标列 TRUST 是最后一列
y_raw = data.iloc[:, -1].values
X = data.iloc[:, :-1]

# ===================== 3. 目标变量转换（二分类：≥16 为1，否则为0） =====================
y = (y_raw >= 16).astype(int)

# ===================== 4. 定义特征列类型 =====================
categorical_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ord_cols = ['TPPA']                        # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC',
                    'PLT', 'NC', 'LY', 'NLR']

# ===================== 5. 划分训练集与测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 6. 连续变量缺失值处理（中位数填充） =====================
continuous_imputer = SimpleImputer(strategy='median')

X_train_cont = pd.DataFrame(
    continuous_imputer.fit_transform(X_train[continuous_cols]),
    columns=continuous_cols,
    index=X_train.index
)
X_test_cont = pd.DataFrame(
    continuous_imputer.transform(X_test[continuous_cols]),
    columns=continuous_cols,
    index=X_test.index
)

# ===================== 7. 分类变量编码 =====================
# 7.1 独热编码（SEX、DEPT、DIAGNOSIS）
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X_train_ohe = pd.DataFrame(
    ohe.fit_transform(X_train[categorical_ohe_cols]),
    columns=ohe.get_feature_names_out(categorical_ohe_cols),
    index=X_train.index
)
X_test_ohe = pd.DataFrame(
    ohe.transform(X_test[categorical_ohe_cols]),
    columns=ohe.get_feature_names_out(categorical_ohe_cols),
    index=X_test.index
)

# 7.2 序数编码（TPPA）
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train_ord = pd.DataFrame(
    ord_enc.fit_transform(X_train[categorical_ord_cols]),
    columns=categorical_ord_cols,
    index=X_train.index
)
X_test_ord = pd.DataFrame(
    ord_enc.transform(X_test[categorical_ord_cols]),
    columns=categorical_ord_cols,
    index=X_test.index
)

# 7.3 合并所有特征
X_train_processed = pd.concat([X_train_cont, X_train_ohe, X_train_ord], axis=1)
X_test_processed = pd.concat([X_test_cont, X_test_ohe, X_test_ord], axis=1)

# ===================== 8. SMOTE 处理分类不平衡 =====================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_processed, y_train
)

print(f"SMOTE前训练集类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"SMOTE后训练集类别分布: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")

# ===================== 9. 随机森林超参数调优 =====================
rf = RandomForestClassifier(random_state=42, n_jobs=1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth':[10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ===================== 10. 模型预测与评估 =====================
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("模型评估结果（测试集）：")
print("=" * 50)
print(f"准确率  (Accuracy):  {accuracy:.4f}")
print(f"召回率  (Recall):    {recall:.4f}")
print(f"精确率  (Precision): {precision:.4f}")
print(f"F1分数  (F1-score):  {f1:.4f}")
print(f"AUC:                 {auc:.4f}")
print("=" * 50)
