import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# 1. 加载数据
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和目标列
X = data.iloc[:, :-1]  # 所有特征列
y = data.iloc[:, -1]  # 目标列（TRUST）

# 3. 将目标列转换为二分类：是否 >= 16
y_binary = (y >= 16).astype(int)

# 4. 定义特征类型
categorical_cols = ['SEX', 'DEPT', 'DIAGNOSIS', 'TPPA']
numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 验证列名是否存在
missing_cols = [col for col in categorical_cols + numerical_cols if col not in X.columns]
if missing_cols:
    print(f"警告：以下列在数据中不存在：{missing_cols}")
    # 仅保留存在的列
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    numerical_cols = [col for col in numerical_cols if col in X.columns]

# 5. 数据预处理
# 分类变量编码（独热编码用于SEX、DEPT、DIAGNOSIS，序数编码用于TPPA）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_cols),
        ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         [col for col in categorical_cols if col != 'TPPA']),
        ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         ['TPPA'] if 'TPPA' in categorical_cols else [])
    ],
    remainder='drop'
)

# 6. 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 7. 构建包含SMOTE和随机森林的pipeline
# 注意：imblearn的Pipeline需要特殊处理
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 8. 定义超参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 9. 网格搜索（不使用多进程）
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 10. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)
print("训练完成！")

# 11. 获取最佳模型
best_model = grid_search.best_estimator_
print(f"\n最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 12. 在测试集上进行预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类的概率

# 13. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# 14. 打印评估结果
print("\n" + "=" * 50)
print("模型评估结果（测试集）:")
print("=" * 50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC值 (AUC-ROC):    {auc_roc:.4f}")
print("=" * 50)

# 15. 保存评估结果到文件（可选）
with open('model_evaluation_results.txt', 'w', encoding='utf-8') as f:
    f.write("随机森林分类模型评估结果\n")
    f.write("=" * 50 + "\n")
    f.write(f"最佳参数组合: {grid_search.best_params_}\n")
    f.write(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}\n\n")
    f.write("测试集评估指标:\n")
    f.write(f"准确率 (Accuracy):  {accuracy:.4f}\n")
    f.write(f"召回率 (Recall):    {recall:.4f}\n")
    f.write(f"精确率 (Precision): {precision:.4f}\n")
    f.write(f"F1分数 (F1-score):  {f1:.4f}\n")
    f.write(f"AUC值 (AUC-ROC):    {auc_roc:.4f}\n")
    f.write("=" * 50 + "\n")

print("\n评估结果已保存到 'model_evaluation_results.txt'")

# 16. 可选：查看特征重要性（需要从pipeline中提取）
try:
    # 获取训练数据经过预处理的形状以确定特征数量
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    # 获取特征名称（用于显示）
    feature_names = []
    # 数值特征
    feature_names.extend(numerical_cols)
    # 独热编码特征
    if [col for col in categorical_cols if col != 'TPPA']:
        onehot_encoder = preprocessor.named_transformers_['cat_onehot']
        for col in [col for col in categorical_cols if col != 'TPPA']:
            categories = onehot_encoder.categories_[categorical_cols.index(col)]
            feature_names.extend([f"{col}_{cat}" for cat in categories])
    # 序数编码特征
    if 'TPPA' in categorical_cols:
        feature_names.append('TPPA')

    # 获取特征重要性
    importances = best_model.named_steps['classifier'].feature_importances_

    # 按重要性排序并显示前10个特征
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n" + "=" * 50)
    print("Top 10 重要特征:")
    print("=" * 50)
    print(feature_importance_df.head(10).to_string(index=False))
    print("=" * 50)

except Exception as e:
    print(f"\n无法显示特征重要性: {e}")

print("\n程序执行完毕！")