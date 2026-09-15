import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 数据加载与目标变量转换
# =============================================================================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

target_col = 'CSF-T'
y = (df[target_col] >= 20).astype(int)
X = df.drop(columns=[target_col])

print(f"原始数据集形状: {X.shape}")
print(f"原始目标变量分布:\n{y.value_counts()}")
print("-" * 60)

# =============================================================================
# 2. 【关键】划分训练集与测试集 (分层抽样)
# =============================================================================
# ⚠️ 必须在SMOTE之前划分，防止合成样本泄露到测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80%训练, 20%测试
    random_state=42,     # 固定随机种子，保证结果可复现
    stratify=y           # 分层抽样，确保训练/测试集中阳性比例一致
)

print(f"训练集形状: {X_train.shape}, 阳性占比: {y_train.mean():.4f}")
print(f"测试集形状: {X_test.shape}, 阳性占比: {y_test.mean():.4f}")
print("-" * 60)

# =============================================================================
# 3. 定义特征分组
# =============================================================================
ordinal_cols = ['SER-T', 'Ink staining']
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
numeric_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 安全检查
all_expected_cols = ordinal_cols + onehot_cols + numeric_cols
missing_cols = [c for c in all_expected_cols if c not in X.columns]
if missing_cols:
    raise ValueError(f"以下特征列在数据集中未找到: {missing_cols}")

# =============================================================================
# 4. 构建预处理管道
# =============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols)
    ],
    remainder='drop'
)

# =============================================================================
# 5. 构建含SMOTE的完整管道 (仅在训练集上拟合)
# =============================================================================
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# 6. 超参数调优 (仅使用训练集)
# =============================================================================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,
    refit=True,
    verbose=1
)

print("开始在训练集上进行超参数调优...")
grid_search.fit(X_train, y_train)  # ✅ 仅用训练集拟合

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
print("-" * 60)

# =============================================================================
# 7. 模型评估 (使用独立测试集)
# =============================================================================
best_model = grid_search.best_estimator_

# 在测试集上预测（SMOTE不会作用于测试集）
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("【独立测试集评估结果】")
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC曲线下面积:      {auc:.4f}")