import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# 1. 加载数据
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 定义特征和目标
# 目标列是最后一列 TRUST
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 将目标转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# 3. 定义特征类型
categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_cols_ordinal = ['TPPA']  # 序数编码
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# 4. 划分训练集和测试集（确保分层采样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 5. 创建预处理流水线
# 5.1 连续变量的预处理：中位数填充
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 5.2 分类变量的预处理
# 独热编码
categorical_transformer_onehot = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数编码
categorical_transformer_ordinal = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 5.3 组合所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat_onehot', categorical_transformer_onehot, categorical_cols_onehot),
        ('cat_ordinal', categorical_transformer_ordinal, categorical_cols_ordinal)
    ])

# 6. 创建完整流水线（包含SMOTE和随机森林）
# 注意：使用imblearn的Pipeline以支持SMOTE
rf_model = RandomForestClassifier(random_state=42)

# 创建包含SMOTE的流水线
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# 7. 超参数调优
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用GridSearchCV进行超参数调优（不使用多进程）
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 8. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)

# 9. 获取最优模型
best_model = grid_search.best_estimator_
print(f"\n最优参数组合: {grid_search.best_params_}")

# 10. 在测试集上进行预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类概率

# 11. 评估模型
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 12. 输出评估结果
print("\n" + "=" * 50)
print("模型评估结果")
print("=" * 50)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC曲线下面积: {auc:.4f}")
print("=" * 50)

# 13. 可选：输出特征重要性（仅当模型训练完成后）
try:
    # 获取特征名称
    # 获取数值特征名称
    num_features = numerical_cols

    # 获取独热编码特征名称
    onehot_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat_onehot'].named_steps['onehot']
    onehot_features = onehot_encoder.get_feature_names_out(categorical_cols_onehot)

    # 序数编码特征
    ordinal_features = categorical_cols_ordinal

    # 组合所有特征名称
    feature_names = list(num_features) + list(onehot_features) + list(ordinal_features)

    # 获取特征重要性
    feature_importances = best_model.named_steps['classifier'].feature_importances_

    # 创建特征重要性DataFrame并排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    print("\n特征重要性（Top 10）：")
    print(importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"\n无法获取特征重要性: {e}")

print("\n模型训练和评估完成！")