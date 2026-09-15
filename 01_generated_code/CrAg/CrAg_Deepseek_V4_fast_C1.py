import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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
data = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 2. 分离特征和目标变量
# 目标列是最后一列CSF-T
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 3. 将目标变量转换为二分类（是否>=20）
y_binary = (y >= 20).astype(int)

# 4. 划分训练集和验证集（4:1）
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 5. 定义特征列
# 分类变量（独热编码）
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
# 分类变量（序数编码）
ordinal_features = ['SER-T', 'Ink staining']
# 连续变量
numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 6. 创建预处理步骤
# 6.1 数值特征预处理：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 6.2 独热编码特征预处理
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 6.3 序数编码特征预处理
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 7. 组合预处理器（按照顺序：数值→独热→序数）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('onehot', onehot_transformer, onehot_features),
        ('ordinal', ordinal_transformer, ordinal_features)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# 8. 创建完整的pipeline（包含SMOTE和分类器）
# 使用imblearn的Pipeline确保SMOTE在交叉验证内部进行，防止数据泄露
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 9. 定义超参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 10. 使用分层K折交叉验证进行超参数调优
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 11. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)

# 12. 获取最佳模型
best_model = grid_search.best_estimator_
print(f"\n最佳参数组合: {grid_search.best_params_}")

# 13. 在验证集上进行预测
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

# 14. 评估模型
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

# 15. 输出评估结果
print("\n========== 模型评估结果 ==========")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC值 (Area Under ROC): {auc:.4f}")
print("====================================\n")

# 16. 显示交叉验证结果
print("交叉验证结果:")
cv_results = grid_search.cv_results_
for i, params in enumerate(cv_results['params']):
    mean_score = cv_results['mean_test_score'][i]
    std_score = cv_results['std_test_score'][i]
    print(f"  参数 {params}: F1均值 = {mean_score:.4f} (+/- {std_score:.4f})")

# 17. 特征重要性分析（可选）
# 获取特征名称
feature_names = []

# 数值特征
feature_names.extend(numeric_features)

# 独热编码特征（获取编码后的特征名）
onehot_encoder = best_model.named_steps['preprocessor'].named_transformers_['onehot']
if hasattr(onehot_encoder, 'get_feature_names_out'):
    onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_features)
else:
    onehot_feature_names = onehot_features
feature_names.extend(onehot_feature_names)

# 序数编码特征
feature_names.extend(ordinal_features)

# 获取特征重要性
if len(feature_names) == len(best_model.named_steps['classifier'].feature_importances_):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n前10个最重要的特征:")
    print(importance_df.head(10))
else:
    print(
        f"\n警告: 特征名称数量({len(feature_names)})与特征重要性数量({len(best_model.named_steps['classifier'].feature_importances_)})不匹配")

print("\n模型构建完成！")