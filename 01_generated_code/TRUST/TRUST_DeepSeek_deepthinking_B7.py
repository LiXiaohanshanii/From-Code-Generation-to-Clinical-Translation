import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # 重要：预测时会自动跳过 SMOTE

# 设置随机种子以保证可重复性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==================== 1. 读取数据 ====================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 目标列在最后一列，列名为 'TRUST'
X = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]  # 原始滴度值

# 二分类目标：是否 >= 16
y = (y_raw >= 16).astype(int)

# ==================== 2. 划分训练集和测试集（分层抽样） ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ==================== 3. 定义特征列 ====================
# 根据描述，分类变量：SEX, DEPT, DIAGNOSIS, TPPA（TPPA用序数编码，其余独热）
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal = ['TPPA']
numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ==================== 4. 构建预处理转换器 ====================
# 数值列：中位数填充
numeric_transformer = SimpleImputer(strategy='median')

# 独热编码列（handle_unknown='ignore' 防止测试集出现未知类别）
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 序数编码列（按训练集的唯一值顺序映射为整数）
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('onehot', onehot_transformer, categorical_onehot),
        ('ordinal', ordinal_transformer, categorical_ordinal)
    ],
    remainder='drop'  # 不保留其他列（但我们已经包含了所有特征）
)

# ==================== 5. 构建包含 SMOTE 和随机森林的 Pipeline ====================
# 注意：使用 imblearn.pipeline.Pipeline，预测时 SMOTE 会自动跳过
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('classifier', RandomForestClassifier(
        random_state=RANDOM_STATE,
        # 固定参数（除 n_estimators 外，其他均按用户指定）
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced'
    ))
])

# ==================== 6. 超参数调优（GridSearchCV） ====================
param_grid = {
    'classifier__n_estimators': [100, 200]
    # 其他参数已在分类器中固定，因此不在此处重复
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,                    # 5折交叉验证
    scoring='roc_auc',       # 以 AUC 作为选择标准
    n_jobs=1,                # 不使用多进程
    verbose=1
)

# 在训练数据上执行 GridSearch（内部包含预处理和 SMOTE）
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# ==================== 7. 在测试集上评估最佳模型 ====================
best_model = grid_search.best_estimator_

# 预测
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# 打印结果
print("\n===== 模型评估结果（测试集） =====")
print(f"准确率 (Accuracy)  : {accuracy:.4f}")
print(f"召回率 (Recall)    : {recall:.4f}")
print(f"精确率 (Precision) : {precision:.4f}")
print(f"F1 分数 (F1-score) : {f1:.4f}")
print(f"AUC  (ROC 曲线下面积): {auc:.4f}")