import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 清理列名两端可能存在的不可见空格
    df.columns = df.columns.str.strip()

    # 2. 目标变量处理
    # 确保 TRUST 列为数值类型（防止数据中存在字符串格式）
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')

    # 二分类任务转换：预测目标列的结果是否 ≥ 16 (1表示>=16, 0表示<16)
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # 3. 定义特征列
    # 注：题目描述中的 DIAGONSIS 修正为正确的拼写 DIAGNOSIS
    cat_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ord_cols = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 划分训练集和测试集 (80%训练，20%测试，使用分层抽样保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 构建数据预处理管道 (ColumnTransformer)
    # 连续变量：中位数填充缺失值
    numeric_transformer = SimpleImputer(strategy='median')

    # 分类变量（独热编码）：handle_unknown='ignore' 防止测试集出现新类别报错
    categorical_ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量（序数编码）：handle_unknown='use_encoded_value' 防止测试集出现新类别报错
    categorical_ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat_ohe', categorical_ohe_transformer, cat_ohe_cols),
            ('cat_ord', categorical_ord_transformer, cat_ord_cols)
        ]
    )

    # 6. 构建包含 SMOTE 和 随机森林 的完整管道
    # 注意：必须使用 imblearn 的 Pipeline，以确保 SMOTE 在数据预处理（编码和填充）之后应用，
    # 并且仅在训练集的交叉验证折内应用，防止数据泄露。
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数调优设置
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 配置 GridSearchCV，n_jobs=1 满足“不使用多进程”的要求
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',  # 针对不平衡数据，使用 F1-score 作为调优寻优指标
        cv=5,
        n_jobs=1,
        verbose=1
    )

    # 8. 训练与调优模型
    print("\n开始进行超参数调优与模型训练（不使用多进程，可能需要一些时间）...")
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print(f"\n最佳超参数组合: {grid_search.best_params_}")

    # 9. 模型评估
    print("正在测试集上评估模型...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估结果
    print("\n" + "=" * 30)
    print("       模型评估结果       ")
    print("=" * 30)
    print(f"准确率 (Accuracy)  : {accuracy:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积      : {auc:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
