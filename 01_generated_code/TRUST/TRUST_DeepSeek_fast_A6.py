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

# ==================== 1. 数据加载 ====================
print("正在加载数据...")
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征和目标列（目标列为最后一列）
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 将目标列转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

print(f"数据集形状: {data.shape}")
print(f"目标列分布:\n{y_binary.value_counts()}")

# ==================== 2. 定义特征类型 ====================
# 分类变量
categorical_cols = ['SEX', 'DEPT', 'DIAGNOSIS', 'TPPA']
# 连续变量
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 验证列名是否存在
for col in categorical_cols + numerical_cols:
    if col not in X.columns:
        raise ValueError(f"列 '{col}' 不在数据集中")

# 验证TPPA是否在分类列中
if 'TPPA' not in categorical_cols:
    raise ValueError("TPPA未在分类变量列表中")

# ==================== 3. 构建预处理流水线 ====================
print("\n构建预处理流水线...")

# 分类变量编码
# 对SEX、DEPT、DIAGNOSIS使用OneHotEncoder（处理DIAGNOSIS拼写，注意原数据可能为DIAGONSIS）
# 原需求中特征列名为DIAGNOSIS，但预处理要求中写为DIAGONSIS，此处使用实际列名
onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
# 如果数据中列名为DIAGONSIS，则使用DIAGONSIS
if 'DIAGONSIS' in X.columns and 'DIAGNOSIS' not in X.columns:
    onehot_cols = ['SEX', 'DEPT', 'DIAGONSIS']
    categorical_cols = ['SEX', 'DEPT', 'DIAGONSIS', 'TPPA']

# 对TPPA使用序数编码
ordinal_cols = ['TPPA']

# 连续变量：使用中位数填充
numerical_imputer = SimpleImputer(strategy='median')

# 构建ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numerical_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
    ],
    remainder='drop'  # 忽略未指定的列
)

# ==================== 4. 划分训练集和测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ==================== 5. 构建包含SMOTE和随机森林的流水线 ====================
print("\n构建模型流水线...")

# 定义模型
rf_model = RandomForestClassifier(random_state=42)

# 构建完整流水线（先预处理，再SMOTE，再模型）
# 注意：SMOTE需要数值特征，因此放在预处理之后
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('classifier', rf_model)
])

# ==================== 6. 超参数调优 ====================
print("\n开始超参数调优...")

# 定义参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用GridSearchCV进行调优，n_jobs=1禁用多进程
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # 5折交叉验证
    scoring='roc_auc',  # 使用AUC作为优化指标
    n_jobs=1,  # 禁用多进程
    verbose=1
)

grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# ==================== 7. 模型评估 ====================
print("\n模型评估...")

# 获取最佳模型
best_model = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 输出评估结果
print("\n" + "=" * 50)
print("模型评估结果 (测试集)")
print("=" * 50)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC值 (ROC-AUC): {auc:.4f}")
print("=" * 50)

# ==================== 8. 特征重要性（可选） ====================
# 由于使用了ColumnTransformer，需要获取特征名称
try:
    # 获取OneHotEncoder的特征名称
    preprocessor_fitted = best_model.named_steps['preprocessor']
    onehot_encoder = preprocessor_fitted.named_transformers_['onehot']
    onehot_features = onehot_encoder.get_feature_names_out(onehot_cols)

    # 所有特征名称
    all_feature_names = numerical_cols + list(onehot_features) + ordinal_cols

    # 获取特征重要性
    rf_classifier = best_model.named_steps['classifier']
    importances = rf_classifier.feature_importances_

    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n特征重要性 (Top 10):")
    print(feature_importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"\n无法获取特征重要性: {e}")

print("\n模型构建与评估完成！")