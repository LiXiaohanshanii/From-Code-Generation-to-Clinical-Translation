import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE

# 1. 加载数据集
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 构建二分类目标变量（TRUST >= 16 为 1，否则为 0）
df['TARGET'] = (df['TRUST'] >= 16).astype(int)

# 3. 定义特征列表
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal = ['TPPA']
continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[categorical_onehot + categorical_ordinal + continuous_features]
y = df['TARGET']

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 构建预处理转换器 (ColumnTransformer)
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量独热编码与序数编码（已确定无缺失值）
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_features),
        ('onehot', onehot_transformer, categorical_onehot),
        ('ordinal', ordinal_transformer, categorical_ordinal)
    ]
)

# 6. 对特征进行特征工程转换
# 严格在训练集上 fit_transform，测试集上仅 transform 以防数据泄露
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# 7. 使用 SMOTE 方法处理类别不平衡（仅应用于训练集）
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

# 8. 随机森林超参数调优（单进程 n_jobs=1）
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']  # 对应参数：'平衡'
}

rf_clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1  # 显式不使用多进程
)

grid_search.fit(X_train_resampled, y_train_resampled)
best_rf = grid_search.best_estimator_

# 9. 模型评估
y_pred = best_rf.predict(X_test_encoded)
y_prob = best_rf.predict_proba(X_test_encoded)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# 10. 输出评估结果
print("========== 最佳超参数 ==========")
print(grid_search.best_params_)
print("\n========== 模型评估指标 ==========")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall):   {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1 分数 (F1-score): {f1:.4f}")
print(f"ROC AUC 面积:       {roc_auc:.4f}")