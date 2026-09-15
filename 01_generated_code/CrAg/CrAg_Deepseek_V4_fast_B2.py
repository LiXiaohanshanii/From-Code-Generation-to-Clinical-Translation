import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重复
RANDOM_STATE = 42

# 1. 读取数据
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 2. 定义特征和目标
# 目标列是最后一列 'CSF-T'
target_col = 'CSF-T'
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']

X = df[feature_cols]
y = df[target_col]

# 3. 创建二分类目标：是否 >= 20
y_binary = (y >= 20).astype(int)

# 4. 定义列类型
# 分类变量（用于独热编码）
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

# 序数编码变量
ordinal_cols = ['SER-T', 'Ink staining']

# 连续变量
numeric_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 5. 定义预处理步骤
# 连续变量的预处理：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码的预处理
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数编码的预处理
# 需要指定每个序数变量的有序类别顺序
# 根据医学领域知识，SER-T和Ink staining通常是有序的
# 请根据实际数据调整以下顺序
ser_t_categories = [['+', '++', '+++', '++++']]  # 示例顺序，请根据实际数据调整
ink_staining_categories = [['-', '+', '++', '+++']]  # 示例顺序，请根据实际数据调整

ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=ser_t_categories + ink_staining_categories))
])

# 6. 构建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('onehot', onehot_transformer, onehot_cols),
        ('ordinal', ordinal_transformer, ordinal_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# 7. 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
)

# 8. 构建包含SMOTE的完整Pipeline
# 先进行预处理，然后应用SMOTE，最后训练模型
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('classifier', RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1  # 不使用多进程
    ))
])

# 9. 定义超参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 10. 使用GridSearchCV进行超参数调优
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # 5折交叉验证
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 11. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)
print("模型训练完成！")

# 12. 输出最佳参数
print("\n最佳参数组合:")
print(grid_search.best_params_)
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 13. 获取最佳模型
best_model = grid_search.best_estimator_

# 14. 在测试集上进行预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类（>=20）的概率

# 15. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 16. 输出评估结果
print("\n" + "="*50)
print("模型评估结果:")
print("="*50)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC值 (Area Under ROC Curve): {auc:.4f}")
print("="*50)

# 17. 可选：输出混淆矩阵的相关信息
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(f"[[TN: {cm[0,0]}, FP: {cm[0,1]}]")
print(f" [FN: {cm[1,0]}, TP: {cm[1,1]}]]")

# 18. 特征重要性分析（可选）
print("\n特征重要性分析:")
print("-"*50)

# 获取特征名称
# 获取预处理后的特征名称
feature_names = []
# 连续变量
feature_names.extend(numeric_cols)

# 独热编码的特征名称
onehot_encoder = best_model.named_steps['preprocessor'].named_transformers_['onehot']
onehot_encoder.fit(X_train[onehot_cols])
for col, categories in zip(onehot_cols, onehot_encoder.categories_):
    for cat in categories:
        feature_names.append(f"{col}_{cat}")

# 序数编码的特征名称
feature_names.extend(ordinal_cols)

# 获取特征重要性
importances = best_model.named_steps['classifier'].feature_importances_

# 创建特征重要性DataFrame并排序
feature_importance_df = pd.DataFrame({
    'feature': feature_names[:len(importances)],
    'importance': importances
}).sort_values('importance', ascending=False)

print("特征重要性（前10个）:")
print(feature_importance_df.head(10))

# 19. 保存模型（可选）
import joblib
joblib.dump(best_model, 'random_forest_model.pkl')
print("\n模型已保存为 'random_forest_model.pkl'")