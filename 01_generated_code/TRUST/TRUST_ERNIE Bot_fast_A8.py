import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ===================== 1. 数据加载 =====================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# ===================== 2. 特征与目标分离 =====================
feature_columns = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                   'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = data[feature_columns].copy()
y = data.iloc[:, -1]  # TRUST 列（最后一列）

# ===================== 3. 目标变量二分类转换 =====================
# 预测 TRUST 是否 >= 16
y_binary = (y >= 16).astype(int)

print(f"原始目标分布:\n{y_binary.value_counts()}")
print(f"正样本比例: {y_binary.mean():.4f}")

# ===================== 4. 定义变量类型 =====================
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal = ['TPPA']
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ===================== 5. 构建预处理流水线 =====================
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 独热编码（SEX, DEPT, DIAGNOSIS）
onehot_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# 序数编码（TPPA）
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous),
        ('onehot', onehot_transformer, categorical_onehot),
        ('ordinal', ordinal_transformer, categorical_ordinal)
    ],
    remainder='drop'
)

# ===================== 6. 拆分训练集与测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ===================== 7. 构建完整Pipeline（含SMOTE） =====================
# 注意：SMOTE需要在预处理之后使用，因此分两步构建Pipeline

# 第一步：预处理
preprocess_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# 对训练集进行预处理
X_train_processed = preprocess_pipeline.fit_transform(X_train)
X_test_processed = preprocess_pipeline.transform(X_test)

# 第二步：SMOTE过采样
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"\nSMOTE后训练集分布:\n{pd.Series(y_train_smote).value_counts()}")

# ===================== 8. 随机森林 + 超参数调优 =====================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

grid_search.fit(X_train_smote, y_train_smote)

# 输出最佳参数
print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ===================== 9. 模型评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("模型评估结果（测试集）")
print("=" * 50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 50)
