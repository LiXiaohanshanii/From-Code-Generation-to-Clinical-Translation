import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 设置随机种子以保证可复现性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    # 1. 读取数据
    data = pd.read_csv('train_data.csv', encoding='utf-8')

    # 目标列为最后一列（TRUST），特征为其余列
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 2. 二分类目标：TRUST >= 16 为 1，否则 0
    y_binary = (y >= 16).astype(int)

    # 3. 划分训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
    )

    # 4. 定义特征列
    categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
    categorical_ordinal_cols = ['TPPA']  # 序数编码
    numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 5. 构建预处理流水线
    # 5.1 对连续变量：中位数填充 + 标准化（可选，随机森林不必须，但此处保留缩放无影响）
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 5.2 对分类变量（独热）
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # 5.3 对分类变量（序数） – 按出现顺序编码
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat_onehot', onehot_transformer, categorical_onehot_cols),
            ('cat_ordinal', ordinal_transformer, categorical_ordinal_cols)
        ],
        remainder='drop'  # 丢弃未指定的列（若有）
    )

    # 6. 构建完整的流水线（包含SMOTE和随机森林）
    #    注意：SMOTE只应用于训练集，测试集仅做预处理，不进行过采样
    #    使用 imblearn 的 Pipeline 确保交叉验证时 SMOTE 在每折的训练部分执行
    model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1  # 单进程，符合要求
        ))
    ])

    # 7. 超参数调优网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 网格搜索（交叉验证，不使用多进程）
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='roc_auc',  # 以AUC为优化目标
        n_jobs=1,  # 单进程
        verbose=1,
        refit=True  # 用最佳参数重新拟合全部训练数据
    )

    print("开始网格搜索...")
    grid_search.fit(X_train, y_train)
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证 AUC: {grid_search.best_score_:.4f}\n")

    # 9. 使用最佳模型在测试集上进行评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

    # 计算各项指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("========== 测试集评估结果 ==========")
    print(f"准确率 (Accuracy)  : {accuracy:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"F1分数             : {f1:.4f}")
    print(f"AUC                : {auc:.4f}")

    # （可选）展示分类报告
    # from sklearn.metrics import classification_report
    # print("\n详细分类报告：")
    # print(classification_report(y_test, y_pred, target_names=['<16', '>=16']))


if __name__ == "__main__":
    main()