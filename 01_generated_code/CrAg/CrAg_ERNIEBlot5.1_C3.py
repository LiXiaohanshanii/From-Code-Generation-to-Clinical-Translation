import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ========== 1. 读取数据 ==========
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ========== 2. 定义特征列和目标列 ==========
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols]
y = df[target_col]

# ========== 3. 目标变量转换为二分类（≥20 为 1，否则为 0） ==========
y = (y >= 20).astype(int)

# ========== 4. 划分训练集和验证集（4:1） ==========
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 5. 定义列类型 ==========
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['SER-T', 'Ink staining']

# 实际使用时请根据数据集确认列名

# ========== 6. 构建预处理管道（数值→独热→序数） ==========
# 6.1 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 6.2 独热编码变量
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 6.3 序数编码变量
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 6.4 组合列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_cols),
        ('onehot', onehot_transformer, onehot_cols),
        ('ordinal', ordinal_transformer, ordinal_cols)
    ],
    remainder='passthrough'  # 防止遗漏列
)

# ========== 7. 构建包含 SMOTE 的完整 Pipeline ==========
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    ))
])

# ========== 8. 超参数调优（GridSearchCV） ==========
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用 StratifiedKFold 进行交叉验证（在 Pipeline 内部，SMOTE 嵌入其中，防止数据泄露）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ========== 9. 输出最佳参数 ==========
print("=" * 60)
print("最佳超参数：")
print(grid_search.best_params_)
print(f"最佳交叉验证 F1 分数：{grid_search.best_score_:.4f}")
print("=" * 60)

# ========== 10. 在验证集上评估 ==========
best_model = grid_search.best_estimator_

# 对验证集进行预测（preprocessor + SMOTE 已在训练时拟合，这里只需 transform）
y_val_pred = best_model.predict(X_val)
y_val_proba = best_model.predict_proba(X_val)[:, 1]

# 计算评估指标
acc = accuracy_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
pre = precision_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
auc = roc_auc_score(y_val, y_val_proba)

print("\n验证集评估结果：")
print(f"准确率 (Accuracy)：  {acc:.4f}")
print(f"召回率 (Recall)：    {rec:.4f}")
print(f"精确率 (Precision)： {pre:.4f}")
print(f"F1 分数 (F1-score)： {f1:.4f}")
print(f"AUC：               {auc:.4f}")
print("=" * 60)