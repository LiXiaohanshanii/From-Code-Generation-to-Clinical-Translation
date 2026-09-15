import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# 设置随机种子，保证结果可重复
RANDOM_STATE = 42

# 1. 读取数据
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和标签（标签为最后一列 TRUST）
X = data.iloc[:, :-1]   # 所有特征列
y_raw = data.iloc[:, -1]  # TRUST 列

# 转换为二分类标签：TRUST >= 16 为正类(1)，否则为负类(0)
y = (y_raw >= 16).astype(int)

# 3. 定义特征分类
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ordinal = ['TPPA']                       # 序数编码
continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 4. 划分训练集和测试集（分层采样，保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 5. 预处理：连续变量缺失值填充（中位数）
imputer_cont = SimpleImputer(strategy='median')
X_train_cont = imputer_cont.fit_transform(X_train[continuous_features])
X_test_cont = imputer_cont.transform(X_test[continuous_features])
# 将填充后的连续变量转为 DataFrame，保留列名
X_train_cont_df = pd.DataFrame(X_train_cont, columns=continuous_features, index=X_train.index)
X_test_cont_df = pd.DataFrame(X_test_cont, columns=continuous_features, index=X_test.index)

# 6. 预处理：独热编码（SEX, DEPT, DIAGNOSIS）
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_onehot = ohe.fit_transform(X_train[categorical_onehot])
X_test_onehot = ohe.transform(X_test[categorical_onehot])
# 生成新列名
ohe_feature_names = ohe.get_feature_names_out(categorical_onehot)
X_train_onehot_df = pd.DataFrame(X_train_onehot, columns=ohe_feature_names, index=X_train.index)
X_test_onehot_df = pd.DataFrame(X_test_onehot, columns=ohe_feature_names, index=X_test.index)

# 7. 预处理：序数编码（TPPA）
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_ord = ord_enc.fit_transform(X_train[categorical_ordinal])
X_test_ord = ord_enc.transform(X_test[categorical_ordinal])
X_train_ord_df = pd.DataFrame(X_train_ord, columns=categorical_ordinal, index=X_train.index)
X_test_ord_df = pd.DataFrame(X_test_ord, columns=categorical_ordinal, index=X_test.index)

# 8. 合并所有预处理后的特征
X_train_processed = pd.concat([X_train_cont_df, X_train_onehot_df, X_train_ord_df], axis=1)
X_test_processed = pd.concat([X_test_cont_df, X_test_onehot_df, X_test_ord_df], axis=1)

# 9. SMOTE 过采样处理训练集中的类别不平衡
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# 10. 定义随机森林模型和超参数网格
rf = RandomForestClassifier(random_state=RANDOM_STATE)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

# 11. 网格搜索（不使用多进程，cv=5 折交叉验证）
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',          # 以 AUC 作为调优指标
    n_jobs=1,                   # 不使用多进程
    verbose=1
)

# 12. 在过采样后的训练集上训练
grid_search.fit(X_train_resampled, y_train_resampled)

# 13. 获取最佳模型
best_rf = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}\n")

# 14. 在测试集上预测
y_pred = best_rf.predict(X_test_processed)
y_pred_proba = best_rf.predict_proba(X_test_processed)[:, 1]  # 正类概率

# 15. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 16. 打印结果
print("========== 模型评估结果 ==========")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:               {auc:.4f}")