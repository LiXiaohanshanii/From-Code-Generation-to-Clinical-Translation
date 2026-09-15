import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# 1. 读取数据
print("正在读取数据...")
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和目标
X = data.iloc[:, :-1]  # 所有列除了最后一列
y = data.iloc[:, -1]  # 最后一列 TRUST

# 3. 目标变量二分类处理（是否>=16）
print("处理目标变量...")
y_binary = (y >= 16).astype(int)
print(f"目标变量分布: 0类={sum(y_binary == 0)}, 1类={sum(y_binary == 1)}")

# 4. 定义特征类型
# 分类变量（需要独热编码）：SEX, DEPT, DIAGNOSIS
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
# 分类变量（需要序数编码）：TPPA
categorical_ordinal_cols = ['TPPA']
# 连续变量（需要中位数填充）：AGE, TP, HIV, WBC, RBC, PLT, NC, LY, NLR
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 5. 构建预处理管道
print("构建预处理管道...")

# 5.1 数值变量预处理：中位数填充
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 5.2 独热编码变量预处理
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# 5.3 序数编码变量预处理
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 5.4 组合所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ])

# 6. 构建完整的管道（包含SMOTE和随机森林）
print("构建模型管道...")

# 6.1 基础随机森林分类器
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=1  # 不使用多进程
)

# 6.2 创建完整管道
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf)
])

# 7. 划分训练集和测试集
print("划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 8. 超参数调优
print("开始超参数调优...")
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

grid_search.fit(X_train, y_train)

# 9. 输出最优参数
print("\n" + "=" * 50)
print("最优参数:")
print("=" * 50)
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

# 10. 在测试集上评估
print("\n" + "=" * 50)
print("模型评估结果:")
print("=" * 50)

# 获取最优模型
best_model = grid_search.best_estimator_

# 预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 输出评估结果
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC曲线下面积 (ROC AUC): {roc_auc:.4f}")

# 11. 额外信息：测试集上的混淆矩阵
print("\n" + "=" * 50)
print("测试集分类结果统计:")
print("=" * 50)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵:\n{cm}")
print(f"真阴性 (TN): {cm[0, 0]}")
print(f"假阳性 (FP): {cm[0, 1]}")
print(f"假阴性 (FN): {cm[1, 0]}")
print(f"真阳性 (TP): {cm[1, 1]}")

# 12. 显示特征重要性（可选）
print("\n" + "=" * 50)
print("模型训练完成!")
print("=" * 50)

# 获取特征名称（用于特征重要性展示）
# 注意：特征重要性只在有数据时才能获取
try:
    # 获取预处理后的特征名称
    preprocessor.fit(X_train)
    feature_names = []

    # 数值特征
    feature_names.extend(numerical_cols)

    # 独热编码特征
    onehot_encoder = preprocessor.named_transformers_['onehot'].named_steps['onehot']
    onehot_categories = onehot_encoder.categories_
    for i, col in enumerate(categorical_onehot_cols):
        feature_names.extend([f"{col}_{cat}" for cat in onehot_categories[i][1:]])  # drop='first'

    # 序数编码特征
    feature_names.extend(categorical_ordinal_cols)

    # 获取特征重要性
    importances = best_model.named_steps['classifier'].feature_importances_

    # 显示top 10重要特征
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nTop 10 重要特征:")
    print(feature_importance_df.head(10).to_string(index=False))

except Exception as e:
    print(f"无法显示特征重要性: {e}")

print("\n" + "=" * 50)
print("程序执行完毕!")
print("=" * 50)