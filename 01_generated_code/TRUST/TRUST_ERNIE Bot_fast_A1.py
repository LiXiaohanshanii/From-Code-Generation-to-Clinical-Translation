import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ======================== 1. 读取数据 ========================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# ======================== 2. 定义特征列 ========================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal_cols = ['TPPA']
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 目标列（最后一列）
target_col = 'TRUST'

# 分离特征和目标
X = data.drop(columns=[target_col])
y = data[target_col]

# ======================== 3. 目标变量转换（二分类：≥16） ========================
y = (y >= 16).astype(int)

# ======================== 4. 划分训练集和测试集 ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================== 5. 构建预处理管道 ========================
# 5.1 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 5.2 分类变量（独热编码）
onehot_transformer = OneHotEncoder(drop='first', sparse_output=False)

# 5.3 分类变量（序数编码）
ordinal_transformer = OrdinalEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ]
)

# ======================== 6. 构建完整Pipeline（含SMOTE + 随机森林） ========================
# 注意：使用imblearn的Pipeline以支持SMOTE
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
])

# ======================== 7. 定义超参数调优网格 ========================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# ======================== 8. 网格搜索超参数调优 ========================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 输出最佳参数
print("=" * 60)
print("最佳超参数：")
print(grid_search.best_params_)
print("=" * 60)

# ======================== 9. 使用最佳模型进行预测 ========================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# ======================== 10. 模型评估 ========================
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("模型评估结果：")
print(f"  准确率 (Accuracy):  {accuracy:.4f}")
print(f"  召回率 (Recall):    {recall:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  F1分数 (F1-score):  {f1:.4f}")
print(f"  AUC曲线下面积:       {auc:.4f}")
print("=" * 60)

# 详细分类报告
print("\n详细分类报告：")
print(classification_report(y_test, y_pred, target_names=['<16', '≥16']))

# ======================== 11. 特征重要性（可选输出） ========================
# 获取预处理后的特征名
onehot_feature_names = best_model.named_steps['preprocessor'].named_transformers_['onehot'].get_feature_names_out(categorical_onehot_cols)
all_feature_names = (
    list(continuous_cols) +
    list(onehot_feature_names) +
    list(categorical_ordinal_cols)
)

# 随机森林特征重要性
importances = best_model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({
    '特征': all_feature_names,
    '重要性': importances
}).sort_values(by='重要性', ascending=False)

print("\n" + "=" * 60)
print("特征重要性（Top 15）：")
print(feature_importance_df.head(15).to_string(index=False))
print("=" * 60)
