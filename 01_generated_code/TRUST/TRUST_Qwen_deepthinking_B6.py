import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # ==========================================
    # 1. 数据加载与目标变量转换
    # ==========================================
    print("正在加载数据...")
    # 读取数据集
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 目标列处理：将TRUST滴度转换为二分类 (>=16 为 1，<16 为 0)
    # 注：滴度数据在csv中应为数值型，若为字符串(如"1:16")，需先做正则提取，此处假设已为纯数值
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # ==========================================
    # 2. 数据集划分
    # ==========================================
    # 使用 stratify=y 确保训练集和测试集中正负样本的比例一致，防止数据划分带来的偏差
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 数据预处理管道构建
    # ==========================================
    # 定义特征列 (注：修正了提示词中 DIAGONSIS 的拼写为 DIAGNOSIS)
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    categorical_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ord_cols = ['TPPA']

    # 连续变量：中位数填充 (Median Imputation)
    continuous_transformer = SimpleImputer(strategy='median')

    # 分类变量 (独热编码)：handle_unknown='ignore' 防止测试集出现未知类别报错
    # sparse_output=False 输出密集矩阵，以完美兼容后续的 SMOTE
    ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量 (序数编码)：TPPA 采用 OrdinalEncoder
    ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器 (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_transformer, continuous_cols),
            ('cat_ohe', ohe_transformer, categorical_ohe_cols),
            ('cat_ord', ord_transformer, categorical_ord_cols)
        ]
    )

    # ==========================================
    # 4. 构建包含 SMOTE 的完整 Pipeline
    # ==========================================
    # 注意：必须使用 imblearn 的 Pipeline，原生的 sklearn Pipeline 不支持在中间步骤改变样本数量(SMOTE)
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),  # 在训练集上应用 SMOTE 处理类别不平衡
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("开始进行网格搜索与超参数调优 (不使用多进程)...")
    # 对于医疗不平衡数据，推荐使用 'roc_auc' 或 'f1' 作为调优评分标准
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=1,  # 明确要求不使用多进程
        verbose=1
    )

    # 训练模型 (SMOTE 会在 CV 的每一折训练集中自动应用，防止数据泄露)
    grid_search.fit(X_train, y_train)

    # ==========================================
    # 6. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 在测试集上进行预测
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各项评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估结果
    print("\n" + "=" * 40)
    print("🎯 模型训练与评估完成！")
    print("=" * 40)
    print(f"最佳超参数组合: {grid_search.best_params_}")
    print(f"准确率 (Accuracy) : {accuracy:.4f}")
    print(f"召回率 (Recall)   : {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score) : {f1:.4f}")
    print(f"AUC曲线下面积     : {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
