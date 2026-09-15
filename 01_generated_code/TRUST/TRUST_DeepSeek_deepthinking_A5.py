import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # StandardScaler未实际使用，但保留
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # 支持SMOTE的Pipeline

# 1. 读取数据
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和目标
# 目标列是最后一列（TRUST），将其转换为二分类：是否 >= 16
X = df.iloc[:, :-1]
y = (df.iloc[:, -1] >= 16).astype(int)

# 3. 定义特征列名
# 根据题目描述：分类变量：SEX、DEPT、DIAGNOSIS、TPPA
# 连续变量：AGE、TP、HIV、WBC、RBC、PLT、NC、LY、NLR
categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']   # 使用独热编码
categorical_cols_ordinal = ['TPPA']                      # 使用序数编码
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 4. 数据划分（训练集70%，测试集30%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. 构建预处理步骤
# 连续变量：中位数填充（题目要求）
numerical_transformer = SimpleImputer(strategy='median')

# 分类变量（独热编码）：处理未知类别，忽略测试集中的新类别
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 分类变量（序数编码）：TPPA，未知类别编码为 -1
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 整合到ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('onehot', onehot_transformer, categorical_cols_onehot),
        ('ordinal', ordinal_transformer, categorical_cols_ordinal)
    ]
)

# 6. 构建完整的Pipeline（包含预处理、SMOTE、随机森林）
# 注意：使用imblearn的Pipeline以保证SMOTE在交叉验证的每一折中正确应用
# 随机森林设置n_jobs=1，避免多进程
rf_classifier = RandomForestClassifier(random_state=42, n_jobs=1)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_classifier)
])

# 7. 超参数调优
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用GridSearchCV，不启用多进程（n_jobs=1），评分指标选择roc_auc进行优化
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=1, verbose=1
)

# 8. 模型训练
grid_search.fit(X_train, y_train)

# 9. 获取最佳模型
best_model = grid_search.best_estimator_
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# 10. 在测试集上评估
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 打印评估结果
print("\n=== 模型在测试集上的评估结果 ===")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC (ROC曲线下面积): {auc:.4f}")