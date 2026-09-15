import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# 1. 读取数据
print("正在读取数据...")
data = pd.read_csv('CrAg_train.csv', encoding='utf-8')
print(f"数据集形状: {data.shape}")

# 2. 定义特征和目标
# 目标列是最后一列 CSF-T
X = data.iloc[:, :-1]  # 所有特征列
y = data.iloc[:, -1]  # 目标列 CSF-T

# 将目标转换为二分类：是否 >= 20
y_binary = (y >= 20).astype(int)
print(f"目标列分布:\n{y_binary.value_counts()}")

# 3. 定义特征类型
# 分类变量（用于OneHotEncoder）
categorical_onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
# 分类变量（用于OrdinalEncoder）
categorical_ordinal_cols = ['SER-T', 'Ink staining']
# 连续变量
numeric_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 4. 数据预处理
# 创建预处理步骤
# 数值特征：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类特征（OneHot编码）
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 分类特征（序数编码）
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ])

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"训练集目标分布:\n{pd.Series(y_train).value_counts()}")

# 6. 构建包含SMOTE和随机森林的流水线
# 使用imblearn的Pipeline以支持SMOTE
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 7. 定义超参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 8. 使用GridSearchCV进行超参数调优
print("\n开始超参数调优...")
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 在训练集上训练
grid_search.fit(X_train, y_train)

# 9. 输出最佳参数
print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 10. 在测试集上进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 获取预测概率（用于AUC）
try:
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
except:
    # 某些情况下可能无法获取概率
    y_pred_proba = None

# 11. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 50)
print("模型评估结果（测试集）:")
print("=" * 50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")

if y_pred_proba is not None:
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC曲线下面积:    {auc:.4f}")
else:
    print("AUC曲线下面积:    无法计算（模型不支持概率预测）")

print("=" * 50)

# 12. 额外信息：训练集上的表现（用于参考）
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
print(f"\n训练集准确率: {train_accuracy:.4f}, 训练集F1: {train_f1:.4f}")

# 13. 特征重要性（可选但有助于理解模型）
print("\n提取特征重要性...")
# 获取预处理后的特征名称
try:
    # 获取数值特征名称
    num_features = numeric_cols

    # 获取onehot编码后的特征名称
    onehot_encoder = best_model.named_steps['preprocessor'].named_transformers_['onehot']
    onehot_features = onehot_encoder.get_feature_names_out(categorical_onehot_cols).tolist()

    # 获取序数编码特征名称
    ordinal_features = categorical_ordinal_cols

    all_feature_names = num_features + onehot_features + ordinal_features

    # 获取特征重要性
    feature_importance = best_model.named_steps['classifier'].feature_importances_

    # 创建特征重要性DataFrame并排序
    importance_df = pd.DataFrame({
        'feature': all_feature_names[:len(feature_importance)],
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\nTop 10 重要特征:")
    print(importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"无法提取特征重要性: {e}")

print("\n模型训练和评估完成！")