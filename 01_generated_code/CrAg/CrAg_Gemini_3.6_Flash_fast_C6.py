import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 1. 加载数据集
file_path = 'CrAg_train.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 2. 构建目标变量（预测 CSF-T 是否 >= 20）
df['target'] = (df['CSF-T'] >= 20).astype(int)

# 定义特征列分类
num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['SER-T', 'Ink staining']

# 划分特征 X 与 标签 y
X = df[num_cols + onehot_cols + ordinal_cols]
y = df['target']

# 3. 划分训练集和验证集（比例 4:1，采用分层抽样保持类别比例）
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 构建数据预处理流水线（按照要求顺序：数值中位数填充 -> 独热编码 -> 序数编码）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
    ]
)

# 5. 构建 imblearn Pipeline（预处理 -> SMOTE -> 随机森林）
# 将 SMOTE 放入 Pipeline 中可保障仅对交叉验证的训练折重采样，杜绝数据泄露
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 6. 超参数调优配置
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
    n_jobs=1  # 不使用多进程
)

# 7. 模型训练与超参数搜索
grid_search.fit(X_train, y_train)

# 获取最佳模型并在验证集上评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

# 8. 计算模型评估指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

# 打印结果
print("=== 最佳超参数组合 ===")
print(grid_search.best_params_)
print("\n=== 验证集评估指标 ===")
print(f"准确率 (Accuracy)  : {accuracy:.4f}")
print(f"召回率 (Recall)    : {recall:.4f}")
print(f"精确率 (Precision) : {precision:.4f}")
print(f"F1分数 (F1-score)  : {f1:.4f}")
print(f"ROC AUC 面积       : {auc:.4f}")