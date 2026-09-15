import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和目标变量
X = data.iloc[:, :-1]  # 所有特征列
y = data.iloc[:, -1]   # 最后一列TRUST

# 3. 创建二分类目标：是否≥16
y_binary = (y >= 16).astype(int)

# 4. 定义特征类型
categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_cols_ordinal = ['TPPA']  # 序数编码
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# 5. 预处理流水线
# 5.1 数值特征处理：中位数填充
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 5.2 分类特征处理：独热编码（处理未知类别）
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 5.3 分类特征处理：序数编码
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 5.4 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat_onehot', onehot_transformer, categorical_cols_onehot),
        ('cat_ordinal', ordinal_transformer, categorical_cols_ordinal)
    ])

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 7. 构建包含SMOTE的完整流水线
# 先预处理，然后SMOTE，再训练模型
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 8. 定义超参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 9. 网格搜索（不使用多进程）
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 10. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)
print("训练完成！")

# 11. 获取最佳模型
best_model = grid_search.best_estimator_
print(f"\n最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# 12. 在测试集上进行预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 13. 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 14. 输出评估结果
print("\n" + "="*50)
print("模型评估结果:")
print("="*50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC分数:            {auc:.4f}")
print("="*50)

# 15. 输出混淆矩阵相关信息（可选）
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# 16. 显示特征重要性（前10个特征）
print("\n特征重要性（Top 10）:")
feature_names = []
# 获取数值特征名
feature_names.extend(numerical_cols)
# 获取独热编码后的特征名
onehot_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat_onehot']
if hasattr(onehot_encoder, 'get_feature_names_out'):
    onehot_features = onehot_encoder.get_feature_names_out(categorical_cols_onehot)
    feature_names.extend(onehot_features)
else:
    # 手动创建特征名
    for col in categorical_cols_onehot:
        unique_vals = X[col].unique()
        feature_names.extend([f"{col}_{val}" for val in unique_vals])
# 添加序数编码特征
feature_names.extend(categorical_cols_ordinal)

# 获取特征重要性
importances = best_model.named_steps['classifier'].feature_importances_
# 创建特征重要性DataFrame
feature_importance_df = pd.DataFrame({
    'feature': feature_names[:len(importances)],
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance_df.head(10))

print("\n模型构建与评估完成！")