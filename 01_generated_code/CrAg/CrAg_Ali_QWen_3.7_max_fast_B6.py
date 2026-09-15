import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)

# ===================== 1. 数据加载与目标变量构建 =====================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 目标列CSF-T为最后一列，构建二分类标签：>=20为阳性(1)，否则为阴性(0)
target_col = df.columns[-1]  # CSF-T
y = (df[target_col] >= 20).astype(int)
X = df.drop(columns=[target_col])

# ===================== 2. 定义特征分组 =====================
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_features = ['SER-T', 'Ink staining']
numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 验证特征列是否全部存在于数据集中
all_specified_features = onehot_features + ordinal_features + numeric_features
missing_cols = [c for c in all_specified_features if c not in X.columns]
if missing_cols:
    raise ValueError(f"以下指定特征列在数据集中不存在: {missing_cols}")

# ===================== 3. 构建预处理Pipeline =====================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
    ],
    remainder='drop'  # 仅保留上述指定特征
)

# ===================== 4. 定义模型与超参数搜索空间 =====================
rf_clf = RandomForestClassifier(random_state=42)

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# ===================== 5. 构建含SMOTE的完整Pipeline =====================
# 使用imblearn的Pipeline确保SMOTE仅在训练集上执行，避免测试集数据泄露
full_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_clf)
])

# ===================== 6. 划分数据集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 7. 网格搜索超参数调优（单进程） =====================
grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,       # 不使用多进程
    refit=True,
    verbose=1
)

print("开始超参数调优...")
grid_search.fit(X_train, y_train)
print(f"最佳F1分数: {grid_search.best_score_:.4f}")
print(f"最佳参数: {grid_search.best_params_}")

# ===================== 8. 模型评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n===== 测试集模型评估结果 =====")
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC曲线下面积:      {auc:.4f}")