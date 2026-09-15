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

# ==================== 1. 数据加载 ====================
print("正在加载数据...")
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征和目标变量
X = data.iloc[:, :-1]  # 所有特征列
y = data.iloc[:, -1]   # 最后一列为目标列TRUST

# 将目标变量转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)
print(f"数据集形状: {X.shape}")
print(f"正样本比例 (>=16): {y_binary.sum() / len(y_binary):.4f}")
print(f"负样本比例 (<16): {(len(y_binary) - y_binary.sum()) / len(y_binary):.4f}")
print("="*50)

# ==================== 2. 数据预处理 ====================
# 定义特征类型
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # One-Hot编码
categorical_ordinal_cols = ['TPPA']  # 序数编码
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# 创建预处理转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_onehot_cols),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ordinal_cols),
        ('numeric', SimpleImputer(strategy='median'), numeric_cols)
    ],
    remainder='drop'  # 删除未指定的列
)

# ==================== 3. 数据划分 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"训练集正样本比例: {y_train.sum() / len(y_train):.4f}")
print(f"测试集正样本比例: {y_test.sum() / len(y_test):.4f}")
print("="*50)

# ==================== 4. 构建包含SMOTE的Pipeline ====================
# 创建包含预处理、SMOTE和分类器的pipeline
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=3)),  # 使用k_neighbors=3防止样本不足
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
])

# ==================== 5. 超参数调优 ====================
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
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")
print("="*50)

# ==================== 6. 模型评估 ====================
print("模型评估结果:")
# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类概率

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC曲线下面积 (Area Under the ROC Curve): {auc:.4f}")
print("="*50)

# ==================== 7. 输出特征重要性（可选） ====================
# 获取特征名称
onehot_feature_names = []
for col in categorical_onehot_cols:
    encoder = best_model.named_steps['preprocessor'].transformers_[0][1]
    if hasattr(encoder, 'get_feature_names_out'):
        feature_names = encoder.get_feature_names_out([col])
        # 去掉第一个类别（drop='first'）
        onehot_feature_names.extend(feature_names[1:] if len(feature_names) > 1 else feature_names)

ordinal_feature_names = categorical_ordinal_cols
numeric_feature_names = numeric_cols

all_feature_names = onehot_feature_names + ordinal_feature_names + numeric_feature_names

# 获取特征重要性
feature_importance = best_model.named_steps['classifier'].feature_importances_

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'feature': all_feature_names[:len(feature_importance)],
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n前10个最重要的特征:")
print(importance_df.head(10))

# ==================== 8. 混淆矩阵结果（仅输出数值） ====================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵:\n{cm}")

print("\n模型构建和评估完成！")