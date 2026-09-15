import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE


def build_crag_rf_model(file_path='CrAg_train.csv'):
    # 1. 数据读取与目标变量构造
    # 读取UTF-8编码的数据集
    data = pd.read_csv(file_path, encoding='utf-8')

    # 构造目标列：CSF-T >= 20 为 1，否则为 0
    # 假设最后一列列名为 'CSF-T'
    target_col = data.columns[-1]
    data['target'] = (data[target_col] >= 20).astype(int)

    # 分离特征和目标
    X = data.drop(columns=[target_col, 'target'])
    y = data['target']

    # 2. 定义特征类型
    continuous_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['SER-T', 'Ink staining']

    # 3. 构造预处理 Pipeline
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量：独热编码（处理未在训练集出现的类别时忽略错误）
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 序数编码：针对 SER-T 和 Ink staining
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 使用 ColumnTransformer 整合不同列的处理逻辑
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, continuous_features),
            ('cat_onehot', onehot_transformer, onehot_features),
            ('cat_ordinal', ordinal_transformer, ordinal_features)
        ]
    )

    # 4. 数据划分 (80% 训练集, 20% 测试集)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 拟合预处理转换器并转换数据
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # 6. 使用 SMOTE 处理训练集样本不平衡问题
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)

    # 7. 定义网格搜索参数与随机森林模型
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    rf = RandomForestClassifier(random_state=42)

    # 参数调优：使用 f1 评分指标，单进程运行 (n_jobs=1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,
        verbose=1
    )

    # 在 SMOTE 采样后的训练数据上进行模型训练与调优
    grid_search.fit(X_train_res, y_train_res)
    best_model = grid_search.best_estimator_

    # 8. 在测试集上评估模型
    y_pred = best_model.predict(X_test_prep)
    y_pred_proba = best_model.predict_proba(X_test_prep)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估结果
    print("=" * 40)
    print("最优超参数配置：", grid_search.best_params_)
    print("-" * 40)
    print("模型评估指标结果 (测试集)：")
    print(f"准确率 (Accuracy):  {acc:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1 分数 (F1-score):  {f1:.4f}")
    print(f"AUC 值 (ROC-AUC):   {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    # 请确保 CrAg_train.csv 放置在当前项目运行目录下
    build_crag_rf_model('CrAg_train.csv')