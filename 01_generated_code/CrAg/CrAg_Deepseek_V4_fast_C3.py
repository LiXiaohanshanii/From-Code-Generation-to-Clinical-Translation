import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# 1. 读取数据
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 2. 分离特征和目标列（目标列是最后一列CSF-T）
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. 将目标列转换为二分类：是否≥20
y_binary = (y >= 20).astype(int)

# 4. 划分训练集和验证集（4:1）
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 5. 定义列分组
# 分类变量
# SER-T、Ink staining - 序数编码
ordinal_cols = ['SER-T', 'Ink staining']

# Color、Transparency、SEX、DEPT、DIAGNOSIS - 独热编码
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

# 连续变量 - 需要中位数填充
numeric_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 6. 构建预处理步骤（按照顺序：数值→独热→序数）
# 步骤1: 数值特征 - 中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 步骤2: 独热编码特征 - 无缺失值
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 步骤3: 序数编码特征 - 无缺失值
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合预处理器（按照数值→独热→序数的顺序）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('onehot', onehot_transformer, onehot_cols),
        ('ordinal', ordinal_transformer, ordinal_cols)
    ]
)

# 7. 构建包含SMOTE的完整Pipeline（防止数据泄露）
# 使用imblearn的Pipeline确保SMOTE只在训练折内进行
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 8. 定义超参数搜索空间
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 9. 使用StratifiedKFold交叉验证（确保每次折内都进行SMOTE）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 10. 网格搜索（不使用多进程）
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 11. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)

# 12. 输出最佳参数
print("\n最佳参数:", grid_search.best_params_)
print("最佳交叉验证F1分数:", grid_search.best_score_)

# 13. 在验证集上评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

# 14. 计算评估指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

# 15. 输出评估结果
print("\n" + "=" * 50)
print("验证集评估结果:")
print("=" * 50)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC曲线下面积 (AUC): {auc:.4f}")
print("=" * 50)

# 16. 可选：显示特征重要性
try:
    # 获取特征名称
    feature_names = []

    # 数值特征
    feature_names.extend(numeric_cols)

    # 独热编码特征
    onehot_encoder = best_model.named_steps['preprocessor'].named_transformers_['onehot']
    if hasattr(onehot_encoder, 'get_feature_names_out'):
        onehot_features = onehot_encoder.get_feature_names_out(onehot_cols)
        feature_names.extend(onehot_features)
    else:
        feature_names.extend(onehot_cols)

    # 序数编码特征
    feature_names.extend(ordinal_cols)

    # 获取特征重要性
    importances = best_model.named_steps['classifier'].feature_importances_

    # 创建特征重要性DataFrame并显示前10个
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n特征重要性 (Top 10):")
    print(feature_importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"\n无法显示特征重要性: {e}")

print("\n模型训练和评估完成！")