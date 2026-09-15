import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# ==================== 1. 读取数据 ====================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ==================== 2. 定义列 ====================
# 特征列（除目标列外的所有列）
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']

# 目标列（最后一列）
target_col = df.columns[-1]  # CSF-T

X = df[feature_cols]
y = (df[target_col] >= 20).astype(int)  # 二分类：>=20 为1，否则为0

# ==================== 3. 划分训练集和验证集（4:1） ====================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== 4. 定义列类型 ====================
# 连续变量（中位数填充）
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 独热编码变量
ohe_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
# 这里保留用户指定的列名，若实际不存在会报错，需根据实际数据调整
ohe_cols_exist = [c for c in ohe_cols if c in X_train.columns]

# 序数编码变量
ordinal_cols = ['SER-T', 'Ink staining']
ordinal_cols_exist = [c for c in ordinal_cols if c in X_train.columns]

# ==================== 5. 构建预处理流水线 ====================
# 步骤1：连续变量 - 中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 步骤2：独热编码
ohe_transformer = Pipeline(steps=[
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 步骤3：序数编码
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_cols),
        ('ohe', ohe_transformer, ohe_cols_exist),
        ('ord', ordinal_transformer, ordinal_cols_exist)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ==================== 6. 使用 imblearn Pipeline 嵌入 SMOTE（防止数据泄露） ====================
model = RandomForestClassifier(random_state=42, n_jobs=1)

# 完整流水线：预处理 -> SMOTE -> 随机森林
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', model)
])

# ==================== 7. 超参数调优（GridSearchCV） ====================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ==================== 8. 输出最佳参数 ====================
print("=" * 60)
print("最佳超参数：")
print(grid_search.best_params_)
print(f"最佳交叉验证 F1 分数：{grid_search.best_score_:.4f}")
print("=" * 60)

# ==================== 9. 在验证集上评估 ====================
best_model = grid_search.best_estimator_

# 预测概率和类别
y_val_pred = best_model.predict(X_val)
y_val_prob = best_model.predict_proba(X_val)[:, 1]

# 计算指标
accuracy = accuracy_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
auc = roc_auc_score(y_val, y_val_prob)

print("\n验证集评估结果：")
print(f"  准确率 (Accuracy):  {accuracy:.4f}")
print(f"  召回率 (Recall):    {recall:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  F1 分数 (F1-score): {f1:.4f}")
print(f"  AUC:                {auc:.4f}")
print("=" * 60)

# ==================== 10. 特征重要性（可选输出） ====================
# 获取预处理后的特征名称
try:
    ohe_feature_names = best_model.named_steps['preprocessor'] \
        .named_transformers_['ohe'] \
        .named_steps['ohe'].get_feature_names_out(ohe_cols_exist)
    num_feature_names = continuous_cols
    ord_feature_names = ordinal_cols_exist
    all_feature_names = np.concatenate([num_feature_names, ohe_feature_names, ord_feature_names])

    importances = best_model.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n特征重要性（前10）：")
    print(feature_importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"\n特征重要性输出失败：{e}")