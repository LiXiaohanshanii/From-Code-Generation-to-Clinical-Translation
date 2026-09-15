# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, roc_curve
)
import warnings

warnings.filterwarnings('ignore')

# 1. 数据加载
print("=" * 50)
print("1. 加载数据...")
data = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 2. 目标变量处理：转换为二分类（≥20为1，否则为0）
print("2. 处理目标变量...")
data['CSF-T_binary'] = (data['CSF-T'] >= 20).astype(int)

# 分离特征和目标
X = data.drop(['CSF-T', 'CSF-T_binary'], axis=1)
y = data['CSF-T_binary']

print(f"数据集大小: {X.shape}")
print(f"正例比例 (≥20): {y.mean():.4f}")
print(f"类别分布:\n{y.value_counts()}")

# 3. 划分训练集和验证集 (4:1)
print("\n3. 划分训练集和验证集 (4:1)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")

# 4. 定义特征列
print("\n4. 定义特征类型...")
# 分类变量（独热编码）
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
# 分类变量（序数编码）
ordinal_features = ['SER-T', 'Ink staining']
# 连续变量（需要中位数填充）
numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 验证所有特征都在数据中
all_features = onehot_features + ordinal_features + numeric_features
missing_features = [f for f in all_features if f not in X.columns]
if missing_features:
    raise ValueError(f"以下特征在数据中不存在: {missing_features}")

# 5. 构建预处理步骤
print("\n5. 构建数据预处理管道...")

# 步骤1: 数值型特征预处理（中位数填充）
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# 步骤2: 独热编码（处理分类变量）
onehot_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 步骤3: 序数编码（处理有序分类变量）
# 注意：SER-T和Ink staining的取值顺序需要根据业务含义定义
# 这里假设SER-T取值如：'negative', '1:1', '1:2', '1:4'等
# Ink staining取值如：'negative', 'weak', 'moderate', 'strong'等
# 请根据实际数据调整categories顺序
ser_t_values = sorted(X_train['SER-T'].unique()) if 'SER-T' in X_train else []
ink_values = sorted(X_train['Ink staining'].unique()) if 'Ink staining' in X_train else []

ordinal_transformer = Pipeline([
    ('ordinal', OrdinalEncoder(
        categories=[ser_t_values, ink_values] if ser_t_values and ink_values else 'auto',
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

# 组合所有预处理器
preprocessor = ColumnTransformer([
    ('numeric', numeric_transformer, numeric_features),
    ('onehot', onehot_transformer, onehot_features),
    ('ordinal', ordinal_transformer, ordinal_features)
])

# 6. 构建完整的Pipeline（包含SMOTE，防止数据泄露）
print("\n6. 构建完整模型Pipeline（SMOTE嵌入交叉验证内部）...")

# 基础模型
rf = RandomForestClassifier(random_state=42)

# 使用imblearn的Pipeline将SMOTE放在适当位置
# 注意：SMOTE应该在预处理之后，分类器之前
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('classifier', rf)
])

# 7. 超参数调优
print("\n7. 超参数调优...")

# 定义参数网格
# 注意：参数名称需要包含步骤名称
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用分层交叉验证，确保SMOTE在每折内部进行
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=cv_strategy,
    n_jobs=1,  # 不使用多进程
    verbose=1
)

print("开始GridSearchCV训练（这可能需要几分钟）...")
grid_search.fit(X_train, y_train)

# 8. 输出调优结果
print("\n" + "=" * 50)
print("8. 超参数调优结果")
print("=" * 50)
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 9. 在验证集上评估
print("\n" + "=" * 50)
print("9. 验证集评估结果")
print("=" * 50)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 预测
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_proba)

print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC曲线下面积 (ROC-AUC): {roc_auc:.4f}")

# 10. 额外：对比训练集性能（检查过拟合）
print("\n" + "=" * 50)
print("10. 训练集性能（检查过拟合）")
print("=" * 50)
y_train_pred = best_model.predict(X_train)
y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]

train_accuracy = accuracy_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)

print(f"训练集 - 准确率: {train_accuracy:.4f}")
print(f"训练集 - 召回率: {train_recall:.4f}")
print(f"训练集 - 精确率: {train_precision:.4f}")
print(f"训练集 - F1分数: {train_f1:.4f}")
print(f"训练集 - ROC-AUC: {train_roc_auc:.4f}")

# 11. 混淆矩阵（可选）
print("\n" + "=" * 50)
print("11. 混淆矩阵")
print("=" * 50)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, y_pred)
print("混淆矩阵:")
print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")

# 12. 特征重要性（可选）
print("\n" + "=" * 50)
print("12. 特征重要性（Top 10）")
print("=" * 50)

# 获取特征名称
# 注意：由于使用了ColumnTransformer，需要获取转换后的特征名
try:
    # 获取预处理器转换后的特征名
    preprocessor_obj = best_model.named_steps['preprocessor']

    # 获取数值特征名
    numeric_names = numeric_features

    # 获取独热编码特征名
    onehot_encoder = preprocessor_obj.named_transformers_['onehot'].named_steps['onehot']
    onehot_names = onehot_encoder.get_feature_names_out(onehot_features).tolist()

    # 获取序数编码特征名
    ordinal_names = ordinal_features

    # 组合所有特征名
    feature_names = numeric_names + onehot_names + ordinal_names

    # 获取特征重要性
    classifier = best_model.named_steps['classifier']
    importances = classifier.feature_importances_

    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("Top 10 重要特征:")
    print(feature_importance_df.head(10).to_string(index=False))

except Exception as e:
    print(f"无法获取特征重要性: {e}")

print("\n" + "=" * 50)
print("模型训练和评估完成！")
print("=" * 50)