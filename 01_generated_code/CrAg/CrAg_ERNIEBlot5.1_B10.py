import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ======================== 1. 数据加载 ========================
data = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 分离特征和目标
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
X = data[feature_cols].copy()
y_raw = data.iloc[:, -1].copy()  # CSF-T 为最后一列

# ======================== 2. 目标变量转换 ========================
# 二分类：CSF-T >= 20 为1，否则为0
y = (y_raw >= 20).astype(int)

# ======================== 3. 变量类型划分 ========================
categorical_ordinal = ['SER-T', 'Ink staining']  # 序数编码
categorical_onehot = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
continuous = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']  # 连续变量

# ======================== 4. 连续变量缺失值填充（中位数） ========================
imputer_median = SimpleImputer(strategy='median')
X_continuous = imputer_median.fit_transform(X[continuous])
X_continuous_df = pd.DataFrame(X_continuous, columns=continuous, index=X.index)

# ======================== 5. 分类变量处理 ========================
# 5.1 序数编码
ordinal_encoder = OrdinalEncoder()
X_ordinal = ordinal_encoder.fit_transform(X[categorical_ordinal])
X_ordinal_df = pd.DataFrame(X_ordinal, columns=categorical_ordinal, index=X.index)

# 5.2 独热编码
onehot_encoder = OneHotEncoder(sparse=False, drop='first')
X_onehot = onehot_encoder.fit_transform(X[categorical_onehot])
onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_onehot)
X_onehot_df = pd.DataFrame(X_onehot, columns=onehot_feature_names, index=X.index)

# ======================== 6. 合并所有特征 ========================
X_processed = pd.concat([X_continuous_df, X_ordinal_df, X_onehot_df], axis=1)

# ======================== 7. 训练集/测试集划分 ========================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# ======================== 8. SMOTE 处理分类不平衡 ========================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"SMOTE前训练集类别分布: {np.bincount(y_train)}")
print(f"SMOTE后训练集类别分布: {np.bincount(y_train_smote)}")

# ======================== 9. 随机森林超参数调优 ========================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42, n_jobs=1)  # 不使用多进程

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1
)

grid_search.fit(X_train_smote, y_train_smote)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ======================== 10. 模型评估 ========================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print("模型评估结果")
print("="*50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):     {recall:.4f}")
print(f"精确率 (Precision):  {precision:.4f}")
print(f"F1分数 (F1-score):   {f1:.4f}")
print(f"AUC曲线下面积:        {auc:.4f}")
print("="*50)