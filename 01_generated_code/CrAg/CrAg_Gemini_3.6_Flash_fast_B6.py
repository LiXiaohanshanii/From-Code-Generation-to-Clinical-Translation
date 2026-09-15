import pandas as pd
import numpy as np

# 导入机器学习与预处理模块
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 导入非平衡数据处理模块 (需安装 imbalanced-learn 库: pip install imbalanced-learn)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # ---------------------------------------------------------
    # 1. 加载数据
    # ---------------------------------------------------------
    file_path = 'CrAg_train.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # ---------------------------------------------------------
    # 2. 构建目标变量（二分类：CSF-T >= 20 为 1，否则为 0）
    # ---------------------------------------------------------
    target_col = 'CSF-T'
    X = df.drop(columns=[target_col])

    # 转换二分类标签：>= 20 为 1，< 20 为 0
    y = (df[target_col] >= 20).astype(int)

    # ---------------------------------------------------------
    # 3. 区分特征类型
    # ---------------------------------------------------------
    # 连续变量
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 序数编码变量
    ord_cols = ['SER-T', 'Ink staining']

    # 独热编码变量
    ohe_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # ---------------------------------------------------------
    # 4. 构建数据预处理流水线（ColumnTransformer）
    # ---------------------------------------------------------
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 序数编码变量：OrdinalEncoder
    ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 独热编码变量：OneHotEncoder
    ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 组合预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('ord', ord_transformer, ord_cols),
            ('ohe', ohe_transformer, ohe_cols)
        ]
    )

    # ---------------------------------------------------------
    # 5. 划分训练集与测试集（按标签比例分层抽样）
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # 6. 构建带有 SMOTE 的 Imbalance Pipeline
    # ---------------------------------------------------------
    # 注意：必须使用 imblearn.pipeline.Pipeline，确保 SMOTE 只对训练集做采样
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # ---------------------------------------------------------
    # 7. 设置超参数网格并进行 GridSearchCV 调优
    # ---------------------------------------------------------
    # 映射管道中的随机森林参数 (前缀需加上 'rf__')
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',  # 以 F1 作为评估和筛选最优参数的指标
        cv=5,  # 5折交叉验证
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    print("正在运行超参数调优...")
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print(f"\n最佳模型参数：{grid_search.best_params_}")

    # ---------------------------------------------------------
    # 8. 模型评估（在独立测试集上）
    # ---------------------------------------------------------
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 输出评估结果
    print("\n================ 模型评估结果 ================")
    print(f"准确率 (Accuracy)   : {accuracy:.4f}")
    print(f"精确率 (Precision)  : {precision:.4f}")
    print(f"召回率 (Recall)     : {recall:.4f}")
    print(f"F1 分数 (F1-Score)  : {f1:.4f}")
    print(f"AUC 面积 (ROC AUC)  : {roc_auc:.4f}")
    print("==============================================")


if __name__ == '__main__':
    main()