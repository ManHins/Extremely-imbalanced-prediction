#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器学习转化概率预测系统 - 快速开始脚本
=====================================

这个脚本提供了项目的快速开始功能，包括：
1. 环境检查
2. 数据加载和预处理
3. 模型训练（简化版）
4. 预测和结果输出

使用方法：
python quick_start.py --data_path your_data.xlsx --action train
python quick_start.py --data_path your_data.xlsx --action predict --model_path saved_models_xxx/ensemble_predictor.pkl
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# 导入配置
try:
    from config import *
except ImportError:
    print("❌ 无法导入config.py，请确保config.py文件存在于同一目录")
    sys.exit(1)

warnings.filterwarnings('ignore')

def check_dependencies():
    """检查必要的依赖包是否已安装"""
    
    print("🔍 检查依赖包...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'seaborn', 'shap', 'joblib', 'openpyxl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包检查通过!")
    return True

def load_and_validate_data(file_path):
    """加载和验证数据"""
    
    print(f"📂 加载数据文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 数据文件不存在: {file_path}")
        return None
    
    try:
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print("❌ 不支持的文件格式，请使用CSV、Excel或Parquet格式")
            return None
        
        print(f"✅ 数据加载成功: {df.shape[0]}行 x {df.shape[1]}列")
        
        # 检查必要列
        missing_columns = []
        for col in FEATURE_CONFIG['REQUIRED_COLUMNS']:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"⚠️  缺少必要列: {missing_columns}")
            print("可用列:", list(df.columns))
            return None
        
        print("✅ 数据格式验证通过!")
        return df
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def simple_feature_engineering(df):
    """简化的特征工程"""
    
    print("🔧 执行特征工程...")
    
    from sklearn.preprocessing import LabelEncoder
    
    # 创建副本
    data = df.copy()
    
    # 基础特征编码
    le = LabelEncoder()
    categorical_features = ['资源渠道']
    
    # 处理可选的分类特征
    optional_categorical = ['long_呼叫时段', 'long_是否工作日', 'long_周几',
                           'long_客户意向_AI', 'long_客户意向_人工']
    
    for col in optional_categorical:
        if col in data.columns:
            categorical_features.append(col)
    
    for col in categorical_features:
        if col in data.columns:
            data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
    
    # 衍生特征
    data['打通率'] = data['打通次数'] / (data['拨打次数'] + 1e-10)
    data['完播率'] = data['利益点完播次数'] / (data['打通次数'] + 1e-10)
    
    if '通话60s以上次数' in data.columns:
        data['长通话率'] = data['通话60s以上次数'] / (data['打通次数'] + 1e-10)
    
    # 选择特征
    feature_columns = ['拨打次数', '打通次数', '利益点完播次数', '打通率', '完播率']
    
    # 添加编码后的特征
    for col in categorical_features:
        encoded_col = f'{col}_encoded'
        if encoded_col in data.columns:
            feature_columns.append(encoded_col)
    
    # 添加可选数值特征
    optional_numeric = ['通话15-30s次数', '通话30-60s次数', '通话60s以上次数', 'long_通话时长']
    for col in optional_numeric:
        if col in data.columns:
            feature_columns.append(col)
    
    # 过滤存在的特征
    available_features = [col for col in feature_columns if col in data.columns]
    
    X = data[available_features].fillna(0)
    y = data['是否转化成交'] if '是否转化成交' in data.columns else None
    
    print(f"✅ 特征工程完成: {len(available_features)}个特征")
    print(f"   特征: {available_features}")
    
    return X, y, available_features

def quick_train(X, y, feature_names, n_models=5):
    """快速训练模型（简化版）"""
    
    print(f"🤖 开始快速训练（{n_models}个模型）...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score
    import joblib
    
    # 分离数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape[0]}样本, 测试集: {X_test.shape[0]}样本")
    print(f"训练集转化率: {y_train.mean():.4f}")
    
    # 训练多个模型
    models = []
    
    for i in range(n_models):
        print(f"训练模型 {i+1}/{n_models}...")
        
        # 欠采样处理
        positive_indices = np.where(y_train == 1)[0]
        negative_indices = np.where(y_train == 0)[0]
        
        # 随机选择负样本
        selected_negative = np.random.choice(
            negative_indices, 
            size=len(positive_indices), 
            replace=False
        )
        
        balanced_indices = np.concatenate([positive_indices, selected_negative])
        X_balanced = X_train.iloc[balanced_indices]
        y_balanced = y_train.iloc[balanced_indices]
        
        # 标准化
        scaler = StandardScaler()
        X_balanced_scaled = scaler.fit_transform(X_balanced)
        
        # 训练模型
        model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        model.fit(X_balanced_scaled, y_balanced)
        
        models.append({
            'model': model,
            'scaler': scaler
        })
    
    # 集成预测
    print("🔍 评估模型性能...")
    
    # 在测试集上评估
    all_probas = []
    for model_info in models:
        X_test_scaled = model_info['scaler'].transform(X_test)
        proba = model_info['model'].predict_proba(X_test_scaled)[:, 1]
        all_probas.append(proba)
    
    ensemble_proba = np.mean(all_probas, axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    # 性能评估
    print("\n📊 模型性能:")
    print(classification_report(y_test, ensemble_pred))
    
    try:
        auc_score = roc_auc_score(y_test, ensemble_proba)
        print(f"AUC分数: {auc_score:.4f}")
    except:
        print("无法计算AUC（可能由于类别不平衡）")
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"quick_trained_models_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'models': models,
        'feature_names': feature_names,
        'timestamp': timestamp,
        'n_models': n_models
    }
    
    model_file = f"{model_dir}/quick_ensemble_model.pkl"
    joblib.dump(model_data, model_file)
    
    print(f"✅ 模型保存至: {model_file}")
    
    return model_data, model_file

def quick_predict(X, model_path, output_path=None):
    """快速预测"""
    
    print(f"🔮 使用模型进行预测: {model_path}")
    
    import joblib
    
    # 加载模型
    try:
        model_data = joblib.load(model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 预测
    models = model_data['models']
    feature_names = model_data['feature_names']
    
    # 确保特征顺序一致
    X_pred = X[feature_names]
    
    all_probas = []
    for model_info in models:
        X_scaled = model_info['scaler'].transform(X_pred)
        proba = model_info['model'].predict_proba(X_scaled)[:, 1]
        all_probas.append(proba)
    
    ensemble_proba = np.mean(all_probas, axis=0)
    
    print(f"✅ 预测完成: {len(ensemble_proba)}个样本")
    print(f"转化概率范围: {ensemble_proba.min():.3f} - {ensemble_proba.max():.3f}")
    print(f"平均转化概率: {ensemble_proba.mean():.3f}")
    
    # 保存结果
    if output_path:
        results_df = pd.DataFrame({
            '转化概率': ensemble_proba,
            '概率百分比': (ensemble_proba * 100).round(1)
        })
        
        results_df.to_excel(output_path, index=False)
        print(f"✅ 预测结果保存至: {output_path}")
    
    return ensemble_proba

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='机器学习转化概率预测系统 - 快速开始')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--action', type=str, choices=['train', 'predict'], required=True, help='执行动作：train(训练) 或 predict(预测)')
    parser.add_argument('--model_path', type=str, help='模型文件路径（预测时需要）')
    parser.add_argument('--output_path', type=str, help='输出文件路径')
    parser.add_argument('--n_models', type=int, default=5, help='训练模型数量（默认5个）')
    
    args = parser.parse_args()
    
    print("🚀 机器学习转化概率预测系统 - 快速开始")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查环境
    check_environment()
    
    # 加载数据
    df = load_and_validate_data(args.data_path)
    if df is None:
        sys.exit(1)
    
    # 特征工程
    X, y, feature_names = simple_feature_engineering(df)
    
    if args.action == 'train':
        if y is None:
            print("❌ 训练模式需要目标变量'是否转化成交'")
            sys.exit(1)
        
        print(f"正负样本比例: {(y==0).sum()}:{(y==1).sum()}")
        
        # 训练模型
        model_data, model_file = quick_train(X, y, feature_names, args.n_models)
        
        print(f"\n🎉 训练完成！")
        print(f"模型文件: {model_file}")
        print(f"使用命令进行预测: python quick_start.py --data_path 新数据.xlsx --action predict --model_path {model_file}")
        
    elif args.action == 'predict':
        if not args.model_path:
            print("❌ 预测模式需要指定 --model_path")
            sys.exit(1)
        
        # 预测
        output_path = args.output_path or f"预测结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        probabilities = quick_predict(X, args.model_path, output_path)
        
        if probabilities is not None:
            print(f"\n🎉 预测完成！")
            print(f"结果文件: {output_path}")
            
            # 显示概率分布
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            print(f"\n概率分布:")
            for threshold in thresholds:
                count = (probabilities >= threshold).sum()
                percentage = count / len(probabilities) * 100
                print(f"  概率 ≥ {threshold}: {count}个 ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 