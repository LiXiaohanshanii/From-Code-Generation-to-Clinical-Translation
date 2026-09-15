import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

# 忽略无关警告，保持控制台输出整洁
warnings.filterwarnings('ignore')


def main():
    # ==========================================
    # 1. 数据加载与目标变量转换
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 提取特征和目标变量
    # 目标列 TRUST 为滴度，转换为二分类：>=16 为 1 (高滴度/活动期)，<16 为 0
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    print(f"数据加载完成。总样本数: {len(df)}, 高滴度(>=16)样本数: {y.sum()} ({y.mean() * 100:.2f}%)")

    # ==========================================
    # 2. 划分训练集与测试集
    # ==========================================
    # 使用 stratify=y 确保训练集和测试集中的正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 数据预处理管道构建
    # ==========================================
    # 定义特征列
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ordinal_cols = ['TPPA']

    # 连续变量预处理：中位数填充 (检验数据中，中位数比均值更能抵抗极端异常值的影响)
    continuous_transformer = SimpleImputer(strategy='median')

    # 分类变量预处理 (独热编码)：handle_unknown='ignore' 防止测试集出现新类别报错
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量预处理 (序数编码)：TPPA结果通常有阴性、1:80、1:160等有序关系
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_transformer, continuous_cols),
            ('cat_onehot', onehot_transformer, categorical_onehot_cols),
            ('cat_ordinal', ordinal_transformer, categorical_ordinal_cols)
        ]
    )

    # ==========================================
    # 4. 构建包含 SMOTE 和 随机森林 的完整 Pipeline
    # ==========================================
    # 注意：必须使用 imblearn.pipeline.Pipeline，以确保 SMOTE 在交叉验证中只在训练集上应用，防止数据泄露
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    # 参数网格 (注意需要加上 'classifier__' 前缀以指定 Pipeline 中的步骤)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("开始进行超参数调优与模型训练 (不使用多进程)...")
    # n_jobs=1 明确指定不使用多进程；scoring='f1' 因为医疗场景下我们更关注正类的识别能力
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳参数组合: {grid_search.best_params_}")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    print("\n正在测试集上评估模型...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类(>=16)的预测概率用于计算AUC

    # 计算各项指标
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 打印评估结果
    print("-" * 40)
    print("【模型评估指标】")
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积      : {auc:.4f}")
    print("-" * 40)

    # 打印详细分类报告
    print("\n【详细分类报告】")
    print(classification_report(y_test, y_pred, target_names=['<16 (低滴度/治愈)', '>=16 (高滴度/活动期)']))

    # 检验科视角的业务解读
    print("【检验科技术人员的业务解读】:")
    print("1. 召回率(Recall)是本项目最核心的指标。漏诊高滴度患者(假阴性)可能导致神经梅毒或心血管梅毒的延误治疗。")
    print("2. 若精确率(Precision)偏低，说明模型存在一定的假阳性，临床医生需结合患者病史和TPPA结果进行综合研判。")
    print("3. 随机森林的 class_weight='balanced' 配合 SMOTE，有效缓解了高滴度样本稀缺带来的模型偏倚问题。")


if __name__ == "__main__":
    main()
