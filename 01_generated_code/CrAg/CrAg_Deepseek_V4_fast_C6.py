import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 分离特征和目标变量
X = df.iloc[:, :-1]  # 除最后一列外的所有列
y = df.iloc[:, -1]  # 最后一列 CSF-T

# 将目标变量转换为二分类：是否 >= 20
y_binary = (y >= 20).astype(int)

# 划分训练集和验证集 (4:1)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 定义特征列
categorical_onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal_cols = ['SER-T', 'Ink staining']
numerical_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 创建预处理步骤
# 1. 数值列：中位数填充
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 2. 独热编码列
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 3. 序数编码列
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'
)

# 构建完整的pipeline（包含SMOTE，防止数据泄露）
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
])

# 定义参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用分层K折交叉验证进行超参数调优（内部交叉验证，避免数据泄露）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    refit=True
)

# 训练模型
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 在验证集上进行预测
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]  # 获取正类概率

# 计算评估指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

# 输出结果
print("=" * 50)
print("随机森林分类模型评估结果")
print("=" * 50)
print(f"最佳参数: {grid_search.best_params_}")
print("-" * 50)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC曲线下面积 (AUC-ROC): {auc:.4f}")
print("=" * 50)

# 输出验证集样本分布信息
print("\n验证集类别分布:")
print(f"负例 (CSF-T < 20): {(y_val == 0).sum()} 个样本")
print(f"正例 (CSF-T >= 20): {(y_val == 1).sum()} 个样本")

# 可选：查看特征重要性（仅作参考）
try:
    # 获取训练好的模型中的随机森林
    rf_model = best_model.named_steps['classifier']
    # 获取预处理后的特征名称
    # 注意：这里需要先对训练数据进行预处理
    X_train_preprocessed = best_model.named_steps['preprocessor'].transform(X_train)

    # 获取特征名称
    feature_names = []
    # 数值特征
    feature_names.extend(numerical_cols)
    # 独热编码特征
    for col in categorical_onehot_cols:
        unique_vals = X_train[col].dropna().unique()
        feature_names.extend([f"{col}_{val}" for val in sorted(unique_vals)])
    # 序数编码特征
    feature_names.extend(categorical_ordinal_cols)

    # 如果特征数量匹配，则显示特征重要性
    if len(feature_names) == X_train_preprocessed.shape[1]:
        importances = rf_model.feature_importances_
        # 排序并显示前10个最重要的特征
        indices = np.argsort(importances)[::-1]
        print("\n特征重要性 Top 10:")
        print("-" * 50)
        for i in range(min(10, len(feature_names))):
            print(f"{i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
except Exception as e:
    print(f"\n特征重要性显示跳过: {e}")