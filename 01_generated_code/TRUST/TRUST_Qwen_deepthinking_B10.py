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
    # ==========================================
    # 1. 数据加载与目标变量处理
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 目标列处理：兼容纯数值(如 16)或带比例格式(如 1:16)，提取最后的数值
    # 题目说明以1、2、4、8等数值表示，这里做鲁棒性处理
    df['TRUST'] = df['TRUST'].astype(str).str.extract(r'(\d+)$').astype(float)

    # 构建二分类目标：TRUST >= 16 为 1 (阳性/高滴度)，否则为 0
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # ==========================================
    # 2. 数据集划分
    # ==========================================
    # 使用分层抽样(stratify)确保训练集和测试集中正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 数据预处理管道设计
    # ==========================================
    # 定义特征列表 (注：修正了题目描述中DIAGONSIS的拼写为DIAGNOSIS)
    numeric_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    cat_onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ordinal_features = ['TPPA']

    # 连续变量：中位数填充
    numeric_transformer = SimpleImputer(strategy='median')

    # 分类变量(独热编码)：设置 sparse_output=False 以输出密集矩阵，对SMOTE更友好
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量(序数编码)：处理测试集中可能出现的未知类别
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat_onehot', onehot_transformer, cat_onehot_features),
            ('cat_ord', ordinal_transformer, cat_ordinal_features)
        ]
    )

    # ==========================================
    # 4. 构建包含 SMOTE 的 Imblearn Pipeline
    # ==========================================
    # 使用 imblearn 的 Pipeline 可以确保 SMOTE 仅在交叉验证的训练折中应用，防止数据泄露
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
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

    print("开始进行超参数调优与模型训练 (不使用多进程)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='f1',  # 针对不平衡数据，使用 F1-score 作为调优的主要评价指标
        n_jobs=1,  # 明确要求不使用多进程
        verbose=1  # 打印训练进度
    )

    grid_search.fit(X_train, y_train)
    print(f"最佳参数组合: {grid_search.best_params_}")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 在测试集上进行预测
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类概率用于计算AUC

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 打印评估结果
    print("\n" + "=" * 30)
    print("测试集模型评估结果：")
    print("=" * 30)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"精确率 (Precision) : {pre:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积      : {auc:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
