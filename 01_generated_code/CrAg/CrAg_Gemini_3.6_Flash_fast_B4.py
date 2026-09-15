import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 如果未安装 imbalanced-learn，请在终端执行: pip install imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 1. 加载数据集
file_path = 'CrAg_train.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# 2. 构建目标变量（二分类：预测 CSF-T 是否 >= 20）
y = (data['CSF-T'] >= 20).astype(int)

# 提取特征集 X（排除目标列 CSF-T）
X = data.drop(columns=['CSF-T'])

# 3. 定义变量分组
categorical_onehot = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal = ['SER-T', 'Ink staining']
continuous_vars = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 4. 构建数据预处理流水线 (ColumnTransformer)
# 连续变量：中位数填充
numeric_transformer = SimpleImputer(strategy='median')

# 分类变量独热编码：忽略未知类别以增强模型鲁棒性
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 分类变量序数编码：缺失值/未知值处理
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 组合不同列的预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_vars),
        ('cat_onehot', onehot_transformer, categorical_onehot),
        ('cat_ordinal', ordinal_transformer, categorical_ordinal)
    ]
)

# 5. 划分训练集与测试集（按照目标变量分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. 构建包含 SMOTE 和随机森林分类器的管道 (使用 imblearn 管道)
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 7. 定义网格搜索参数范围
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 8. 使用 GridSearchCV 进行超参数调优（不使用多进程 n_jobs=1，以 F1 为评价指标）
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1
)

# 模型训练
grid_search.fit(X_train, y_train)

# 9. 模型评估
best_model = grid_search.best_estimator_

# 获取测试集预测结果与预测概率
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算各标准评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 10. 输出评估结果
print("================ 最佳模型参数 ================")
print(grid_search.best_params_)
print("\n================ 模型评估结果 ================")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"F1 分数 (F1-Score): {f1:.4f}")
print(f"AUC 面积 (ROC-AUC):  {auc:.4f}")