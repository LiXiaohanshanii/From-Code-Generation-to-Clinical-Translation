import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ===================== 1. 加载数据 =====================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# ===================== 2. 分离特征与目标 =====================
X = data.iloc[:, :-1]  # 所有特征列（最后一列是目标）
y_raw = data.iloc[:, -1]  # TRUST 列

# ===================== 3. 目标变量转换（二分类：≥16 → 1，否则 → 0） =====================
y = (y_raw >= 16).astype(int)

# ===================== 4. 定义列类型 =====================
categorical_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ord_cols = ['TPPA']                       # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ===================== 5. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 6. 连续变量缺失值填充（中位数） =====================
cont_imputer = SimpleImputer(strategy='median')
X_train_cont = pd.DataFrame(
    cont_imputer.fit_transform(X_train[continuous_cols]),
    columns=continuous_cols,
    index=X_train.index
)
X_test_cont = pd.DataFrame(
    cont_imputer.transform(X_test[continuous_cols]),
    columns=continuous_cols,
    index=X_test.index
)

# ===================== 7. 独热编码（SEX、DEPT、DIAGNOSIS） =====================
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

# ===================== 8. 序数编码（TPPA） =====================
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

# ===================== 9. 合并所有特征 =====================
X_train_processed = pd.concat([X_train_cont, X_train_ohe, X_train_ord], axis=1)
X_test_processed = pd.concat([X_test_cont, X_test_ohe, X_test_ord], axis=1)

# ===================== 10. SMOTE 处理类别不平衡 =====================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"SMOTE 前训练集类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"SMOTE 后训练集类别分布: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")

# ===================== 11. 超参数调优（GridSearchCV） =====================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth':[10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=1)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

print(f"\n最佳超参数: {grid_search.best_params_}")
print(f"最佳交叉验证 F1 分数: {grid_search.best_score_:.4f}")

# ===================== 12. 使用最佳模型预测 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

# ===================== 13. 模型评估 =====================
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 55)
print("随机森林分类模型 - 测试集评估结果")
print("=" * 55)
print(f"  准确率 (Accuracy):   {accuracy:.4f}")
print(f"  召回率 (Recall):     {recall:.4f}")
print(f"  精确率 (Precision):  {precision:.4f}")
print(f"  F1 分数 (F1-score):  {f1:.4f}")
print(f"  AUC:                 {auc:.4f}")
print("=" * 55)
