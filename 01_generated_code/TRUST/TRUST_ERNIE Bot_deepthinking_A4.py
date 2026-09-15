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

# ============================================================
# 1. 数据加载
# ============================================================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# ============================================================
# 2. 分离特征与目标变量
# ============================================================
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
X = data[feature_cols]
y = data['TRUST']  # 目标列为最后一列

# ============================================================
# 3. 目标变量转换为二分类：TRUST >= 16 为1，否则为0
# ============================================================
y = (y >= 16).astype(int)

# ============================================================
# 4. 定义分类变量和连续变量
# ============================================================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal_cols = ['TPPA']                      # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ============================================================
# 5. 构建预处理管道
# ============================================================
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量（独热编码）
categorical_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 分类变量（序数编码）
categorical_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value',
                                                 unknown_value=-1)

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_cols),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot_cols),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'  # 不包含的列丢弃（此处所有特征已覆盖）
)

# ============================================================
# 6. 划分训练集和测试集
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 7. 对训练集进行预处理
# ============================================================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ============================================================
# 8. 使用SMOTE处理分类不平衡（仅对训练集）
# ============================================================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"SMOTE前训练集类别分布: {np.bincount(y_train)}")
print(f"SMOTE后训练集类别分布: {np.bincount(y_train_smote)}")

# ============================================================
# 9. 构建随机森林模型并进行超参数调优
# ============================================================
rf_classifier = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

grid_search.fit(X_train_smote, y_train_smote)

# ============================================================
# 10. 输出最佳参数
# ============================================================
print(f"\n最佳超参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ============================================================
# 11. 使用最佳模型在测试集上进行预测和评估
# ============================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_processed)
y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# ============================================================
# 12. 输出评估结果
# ============================================================
print("\n" + "="*50)
print("模型评估结果（测试集）")
print("="*50)
print(f"准确率（Accuracy）:  {accuracy:.4f}")
print(f"召回率（Recall）:   {recall:.4f}")
print(f"精确率（Precision）: {precision:.4f}")
print(f"F1分数（F1-score）:  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("="*50)
