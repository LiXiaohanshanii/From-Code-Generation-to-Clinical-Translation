import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 分离特征和目标变量
X = df.drop('CSF-T', axis=1)
y = df['CSF-T']

# 将目标变量转换为二分类（是否≥20）
y_binary = (y >= 20).astype(int)

# 定义特征类型
categorical_cols_onehot = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
categorical_cols_ordinal = ['SER-T', 'Ink staining']
numerical_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 检查分类变量是否有缺失值（根据要求，分类变量无缺失值，但为了稳健性仍进行检查）
print("检查分类变量缺失值情况：")
print(df[categorical_cols_onehot + categorical_cols_ordinal].isnull().sum())

# 数据预处理管道
# 数值特征：中位数填充
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类特征-独热编码
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# 分类特征-序数编码
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('onehot', onehot_transformer, categorical_cols_onehot),
        ('ordinal', ordinal_transformer, categorical_cols_ordinal)
    ])

# 创建包含SMOTE和随机森林的完整管道
# 注意：使用imblearn的Pipeline来支持SMOTE
model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))  # 不使用多进程
])

# 划分训练集和测试集（20%测试集）
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print(f"训练集正例比例: {y_train.mean():.4f}")
print(f"测试集正例比例: {y_test.mean():.4f}")

# 超参数调优
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

print("\n开始超参数调优...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 如果有预测概率，计算AUC
try:
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
except:
    auc_score = None
    print("无法计算AUC，可能是因为某些类别缺失")

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 输出评估结果
print("\n" + "=" * 50)
print("模型评估结果（测试集）:")
print("=" * 50)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
if auc_score is not None:
    print(f"AUC曲线下面积 (AUC): {auc_score:.4f}")
print("=" * 50)

# 输出混淆矩阵相关信息
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵:")
print(f"真阴性 (TN): {cm[0, 0]}")
print(f"假阳性 (FP): {cm[0, 1]}")
print(f"假阴性 (FN): {cm[1, 0]}")
print(f"真阳性 (TP): {cm[1, 1]}")

# 可选：输出特征重要性（获取特征名称）
print("\n" + "=" * 50)
print("特征重要性分析（Top 10）:")
print("=" * 50)

# 获取预处理后的特征名称
preprocessor_fitted = best_model.named_steps['preprocessor']
onehot_features = []
ordinal_features = []

# 获取独热编码特征名称
if 'onehot' in preprocessor_fitted.named_transformers_:
    onehot_encoder = preprocessor_fitted.named_transformers_['onehot'].named_steps['onehot']
    if hasattr(onehot_encoder, 'get_feature_names_out'):
        onehot_features = onehot_encoder.get_feature_names_out(categorical_cols_onehot).tolist()
    else:
        # 兼容旧版本sklearn
        for col in categorical_cols_onehot:
            unique_vals = df[col].dropna().unique()
            for val in unique_vals:
                onehot_features.append(f"{col}_{val}")

# 序数编码特征名称
ordinal_features = categorical_cols_ordinal

# 数值特征名称
numerical_features = numerical_cols

# 合并所有特征名称
all_feature_names = numerical_features + onehot_features + ordinal_features

# 获取特征重要性
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    importances = best_model.named_steps['classifier'].feature_importances_

    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 输出Top 10
    print(feature_importance_df.head(10).to_string(index=False))
else:
    print("无法获取特征重要性")

print("\n模型构建和评估完成！")