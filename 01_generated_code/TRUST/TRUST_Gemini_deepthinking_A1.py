import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

# 使用 imblearn 的 Pipeline 以确保 SMOTE 仅应用于训练集交叉验证 fold，防止数据泄露
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 定义特征变量与目标变量
    # 连续变量列表
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    # 独热编码分类变量列表
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    # 序数编码分类变量列表
    ordinal_cols = ['TPPA']

    # 所有特征列
    feature_cols = num_cols + onehot_cols + ordinal_cols
    X = df[feature_cols]

    # 二分类目标变量转换：TRUST >= 16 标记为 1，否则标记为 0
    y = (df['TRUST'] >= 16).astype(int)

    # 3. 划分训练集与测试集 (80% 训练, 20% 测试，分层抽样保持类比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 构建数据预处理流水线 (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            # 连续变量：中位数填充缺失值
            ('num', SimpleImputer(strategy='median'), num_cols),
            # 分类变量：独热编码 (handle_unknown='ignore' 处理未见过的类别)
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
            # 分类变量：序数编码
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ]
    )

    # 5. 构建组合流水线：预处理 -> SMOTE -> 随机森林分类器
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 6. 定义超参数网格 (注意：网格参数名前缀需要对应流水线中的 'rf' 步骤)
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']  # sklearn 中“平衡”对应的参数值为 'balanced'
    }

    # 7. 超参数调优 (GridSearchCV，设置 n_jobs=1 不使用多进程)
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # 使用 F1-score 作为模型评估调优的参考指标
        n_jobs=1       # 显式限制单进程运行
    )

    print("正在进行模型训练和超参数搜索...")
    grid_search.fit(X_train, y_train)

    # 8. 获取最佳模型并在测试集上评估
    best_model = grid_search.best_estimator_

    # 预测标签和预测概率
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 9. 输出结果
    print("\n" + "=" * 40)
    print("【最佳超参数配置】")
    for param, value in grid_search.best_params_.items():
        print(f"  {param.replace('rf__', '')}: {value}")

    print("\n【测试集模型评估指标】")
    print(f"  准确率 (Accuracy):           {acc:.4f}")
    print(f"  召回率 (Recall):             {rec:.4f}")
    print(f"  精确率 (Precision):          {prec:.4f}")
    print(f"  F1 分数 (F1-score):          {f1:.4f}")
    print(f"  ROC 曲线下面积 (AUC):        {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()