# 机器学习转化概率预测系统 - 配置文件
# ===========================================

import os

# 数据路径配置
# ================

# 输入数据文件路径（请修改为你的实际文件路径）
DATA_PATH = {
    'TRAIN_DATA': 'data/训练数据.parquet',  # 训练数据文件
    'PREDICT_DATA': 'data/预测数据.xlsx',   # 需要预测的数据文件
    'OUTPUT_DIR': 'output/',               # 输出结果目录
}

# 模型配置
# ============

MODEL_CONFIG = {
    # 多重欠采样配置
    'N_MODELS': 10,                        # 训练的模型数量（原项目使用100个，可根据需要调整）
    'POSITIVE_SAMPLE_MULTIPLIER': 1,       # 负样本相对正样本的倍数（1表示1:1平衡）
    'RANDOM_STATE': 42,                    # 随机种子，确保结果可重现
    
    # 逻辑回归参数
    'REGULARIZATION_PARAMS': [0.001, 0.01, 0.1, 1, 10, 100],  # L2正则化参数候选
    'MAX_ITER': 1000,                      # 最大迭代次数
    'CLASS_WEIGHT': 'balanced',            # 类别权重处理
    
    # 交叉验证
    'CV_FOLDS': 5,                         # 交叉验证折数
    'SCORING_METRIC': 'f2',                # 优化目标（f2分数适合转化预测）
}

# 特征工程配置
# =================

FEATURE_CONFIG = {
    # 基础特征列名（如果你的数据列名不同，请修改）
    'REQUIRED_COLUMNS': [
        '加密手机号码',
        '是否转化成交',
        '拨打次数', 
        '打通次数',
        '利益点完播次数',
        '资源渠道'
    ],
    
    # 可选特征列名
    'OPTIONAL_COLUMNS': [
        'long_呼叫时段',
        'long_是否工作日', 
        'long_周几',
        'long_客户意向_AI',
        'long_客户意向_人工',
        'long_通话时长'
    ],
    
    # 分位数阈值（用于创建高/低特征）
    'QUANTILE_THRESHOLDS': {
        '打通率': 0.25,
        '完播率': 0.25,
        '长通话率': 0.25,
        '参与度总分': 0.25
    }
}

# 预测策略配置
# ==================

PREDICTION_STRATEGIES = {
    '最高召回率策略': {
        'description': '最大化转化用户识别率，适合营销推广',
        'default_threshold': 0.1
    },
    '最高F1策略': {
        'description': '平衡精确率和召回率，通用推荐',
        'default_threshold': 0.5
    },
    '最高精确率策略': {
        'description': '最小化误判率，适合精准营销',
        'default_threshold': 0.8
    },
    '业务平衡策略': {
        'description': '适中的业务平衡点',
        'default_threshold': 0.5
    }
}

# 输出配置
# ============

OUTPUT_CONFIG = {
    # 模型保存
    'SAVE_MODELS': True,                   # 是否保存训练的模型
    'MODEL_DIR_PREFIX': 'saved_models_',   # 模型目录前缀
    
    # 结果输出
    'SAVE_PREDICTIONS': True,              # 是否保存预测结果
    'PREDICTION_FILENAME': '转化概率预测结果.xlsx',
    
    # 分析报告
    'SAVE_FEATURE_IMPORTANCE': True,       # 是否保存特征重要性分析
    'SAVE_PERFORMANCE_REPORT': True,       # 是否保存性能报告
    
    # Excel输出格式
    'EXCEL_CONFIG': {
        'include_probability': True,        # 包含概率值
        'include_percentage': True,         # 包含百分比
        'include_original_data': True,      # 包含原始数据
        'sheet_names': {
            'predictions': '预测结果',
            'summary': '预测汇总',
            'feature_importance': '特征重要性'
        }
    }
}

# 可视化配置
# ===============

VISUALIZATION_CONFIG = {
    # 图表样式
    'FIGURE_SIZE': (12, 8),               # 默认图表尺寸
    'DPI': 300,                           # 图片分辨率
    'FONT_FAMILY': 'SimHei',              # 中文字体
    'COLOR_PALETTE': 'Set2',              # 颜色主题
    
    # 保存图片
    'SAVE_PLOTS': True,                   # 是否保存图片
    'PLOT_FORMAT': 'png',                 # 图片格式
    'PLOT_DIR': 'plots/',                 # 图片保存目录
}

# SHAP分析配置
# ================

SHAP_CONFIG = {
    'ENABLE_SHAP': True,                  # 是否启用SHAP分析
    'MAX_SAMPLES_FOR_SHAP': 1000,        # SHAP分析的最大样本数（太多会很慢）
    'PLOT_TOP_FEATURES': 20,              # 显示前N个重要特征
    'SAVE_SHAP_VALUES': True,             # 是否保存SHAP值
}

# 性能优化配置
# ==================

PERFORMANCE_CONFIG = {
    # 内存管理
    'CHUNK_SIZE': 10000,                  # 大数据分块处理大小
    'USE_CHUNKING': False,                # 是否启用分块处理（大数据时设为True）
    
    # 并行处理
    'N_JOBS': -1,                         # 并行任务数（-1表示使用所有CPU核心）
    'ENABLE_PARALLEL': True,              # 是否启用并行处理
    
    # 内存优化
    'OPTIMIZE_DTYPES': True,              # 是否优化数据类型以节省内存
    'DELETE_INTERMEDIATE': True,          # 是否删除中间变量释放内存
}

# 验证集配置
# ===============

VALIDATION_CONFIG = {
    'FINAL_VALIDATION_SIZE': 30,          # 最终验证集正样本数量
    'VALIDATION_NEGATIVE_RATIO': 3885,    # 验证集负样本比例（相对于正样本）
    'USE_STRATIFIED_SPLIT': True,         # 是否使用分层抽样
}

# 日志配置
# ============

LOGGING_CONFIG = {
    'LOG_LEVEL': 'INFO',                  # 日志级别：DEBUG, INFO, WARNING, ERROR
    'LOG_FILE': 'logs/ml_conversion.log', # 日志文件路径
    'ENABLE_CONSOLE_LOG': True,           # 是否在控制台输出日志
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# 环境检查函数
# ===============

def check_environment():
    """检查环境配置是否正确"""
    
    # 检查必要目录
    required_dirs = [
        OUTPUT_CONFIG['OUTPUT_DIR'],
        VISUALIZATION_CONFIG['PLOT_DIR'],
        os.path.dirname(LOGGING_CONFIG['LOG_FILE'])
    ]
    
    for dir_path in required_dirs:
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ 创建目录: {dir_path}")
    
    # 检查数据文件
    if os.path.exists(DATA_PATH['TRAIN_DATA']):
        print(f"✅ 找到训练数据: {DATA_PATH['TRAIN_DATA']}")
    else:
        print(f"⚠️  训练数据文件不存在: {DATA_PATH['TRAIN_DATA']}")
        print("   请在config.py中修改DATA_PATH['TRAIN_DATA']为正确路径")
    
    print("环境检查完成！")

def get_config_summary():
    """获取配置摘要"""
    
    summary = f"""
配置摘要:
========
• 训练模型数量: {MODEL_CONFIG['N_MODELS']}
• 交叉验证折数: {MODEL_CONFIG['CV_FOLDS']}  
• 优化指标: {MODEL_CONFIG['SCORING_METRIC']}
• 并行处理: {'启用' if PERFORMANCE_CONFIG['ENABLE_PARALLEL'] else '禁用'}
• SHAP分析: {'启用' if SHAP_CONFIG['ENABLE_SHAP'] else '禁用'}
• 输出目录: {OUTPUT_CONFIG['OUTPUT_DIR']}
"""
    return summary

# 在导入时自动检查环境（可选）
if __name__ == "__main__":
    check_environment()
    print(get_config_summary()) 