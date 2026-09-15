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
    print("[1/6] 正在加载数据...")
    # 读取数据集，指定 utf-8 编码
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 将 TRUST 滴度转换为数值型（防止数据中混入字符串格式）
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')

    # 构建二分类目标变量：TRUST >= 16 为 1 (阳性/高滴度)，否则为 0
    y = (df['TRUST'] >= 16).astype(int)

    # 提取特征矩阵 (剔除目标列)
    X = df.drop(columns=['TRUST'])

    # ==========================================
    # 2. 数据集划分
    # ==========================================
    print("[2/6] 正在划分训练集与测试集...")
    # 使用分层抽样(stratify)保证训练集和测试集中高滴度样本的比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 数据预处理管道构建
    # ==========================================
    print("[3/6] 正在构建数据预处理管道...")
    # 定义各类特征列
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码列
    ord_cols = ['TPPA']  # 序数编码列

    # 连续变量：中位数填充缺失值
    continuous_transformer = SimpleImputer(strategy='median')

    # 分类变量(独热)：处理未知类别，输出密集矩阵
    ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量(序数)：TPPA编码，处理测试集中可能出现的未知类别
    ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合 ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_transformer, continuous_cols),
            ('ohe', ohe_transformer, ohe_cols),
            ('ord', ord_transformer, ord_cols)
        ]
    )

    # ==========================================
    # 4. 构建包含 SMOTE 的完整 Pipeline
    # ==========================================
    # 注意：必须使用 imblearn 的 Pipeline，以确保 SMOTE 仅在训练集的 CV 折叠中应用，防止数据泄露
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    print("[4/6] 开始进行超参数调优（不使用多进程），请稍候...")
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # n_jobs=1 确保不使用多进程；scoring='roc_auc' 适合不平衡分类任务的调优
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n[5/6] 调优完成！最佳参数组合: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # ==========================================
    # 6. 模型评估
    # ==========================================
    print("[6/6] 正在测试集上评估模型性能...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类概率用于计算 AUC

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估报告
    print("\n" + "=" * 45)
    print(" 随机森林模型评估结果 (测试集) ".center(40, "="))
    print("=" * 45)
    print(f" 准确率 (Accuracy)   : {accuracy:.4f}")
    print(f" 召回率 (Recall)     : {recall:.4f}  <-- 临床最关注：漏诊率")
    print(f" 精确率 (Precision)  : {precision:.4f}")
    print(f" F1分数 (F1-score)   : {f1:.4f}")
    print(f" AUC曲线下面积 (AUC) : {auc:.4f}  <-- 综合区分能力")
    print("=" * 45)


if __name__ == "__main__":
    main()
