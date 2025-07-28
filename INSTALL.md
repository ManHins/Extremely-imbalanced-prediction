# 安装和快速入门指南

## 系统要求

### 最低配置
- **操作系统**：Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python版本**：Python 3.7 或更高版本
- **内存**：至少 4GB RAM（推荐 8GB+）
- **存储**：至少 2GB 可用空间

### 推荐配置
- **Python版本**：Python 3.8-3.10
- **内存**：8GB+ RAM
- **处理器**：多核CPU（提升训练速度）

## 安装步骤

### 1. 环境准备

#### 方法一：使用虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv ml_conversion_env

# 激活虚拟环境
# Windows:
ml_conversion_env\Scripts\activate
# macOS/Linux:
source ml_conversion_env/bin/activate
```

#### 方法二：使用conda
```bash
# 创建conda环境
conda create -n ml_conversion python=3.8
conda activate ml_conversion
```

### 2. 安装依赖包

```bash
# 克隆项目（如果还没有）
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 安装所有依赖
pip install -r requirements.txt
```

#### 常见安装问题解决

**问题1：SHAP安装失败**
```bash
# 先安装必要的编译工具
pip install wheel setuptools
pip install shap --no-cache-dir
```

**问题2：matplotlib中文显示问题**
```bash
# Windows用户可能需要安装中文字体支持
pip install matplotlib --upgrade
```

**问题3：scikit-learn版本冲突**
```bash
# 指定特定版本
pip install scikit-learn==1.1.0 --force-reinstall
```

### 3. 验证安装

创建测试文件 `test_installation.py`：

```python
# test_installation.py
import pandas as pd
import numpy as np
import sklearn
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("✅ 所有核心库安装成功！")
print(f"pandas版本: {pd.__version__}")
print(f"numpy版本: {np.__version__}")
print(f"scikit-learn版本: {sklearn.__version__}")
print(f"shap版本: {shap.__version__}")

# 测试基本功能
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
model = LogisticRegression()
model.fit(X, y)
print("✅ 机器学习模型测试成功！")
```

运行测试：
```bash
python test_installation.py
```

## 快速入门

### 1. 准备数据

确保你的数据文件包含以下必要字段：

**必需字段：**
- `加密手机号码`：用户唯一标识
- `是否转化成交`：目标变量（0或1）
- `拨打次数`：外呼次数
- `打通次数`：成功接通次数
- `利益点完播次数`：完整播放次数
- `资源渠道`：客户来源渠道

**可选字段：**
- `long_呼叫时段`：呼叫时间段
- `long_是否工作日`：是否工作日
- `long_客户意向_AI`：AI评估的客户意向
- `long_客户意向_人工`：人工评估的客户意向

### 2. 数据格式示例

```csv
加密手机号码,是否转化成交,拨打次数,打通次数,利益点完播次数,资源渠道
user_001,1,5,3,2,外部白名单
user_002,0,3,1,0,自然流量
user_003,1,8,6,4,合作渠道
```

### 3. 启动Jupyter Notebook

```bash
# 启动Jupyter
jupyter notebook

# 或使用JupyterLab
jupyter lab
```

### 4. 运行项目

1. 打开 `机器学习预测转化概率3shap.ipynb`
2. 将你的数据文件路径替换为实际路径：
   ```python
   df = pd.read_parquet('你的数据文件.parquet')
   # 或
   df = pd.read_excel('你的数据文件.xlsx')
   ```
3. 逐个运行代码单元格

### 5. 模型训练（简化版）

如果只想快速训练模型：

```python
# 简化训练流程
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 1. 加载数据
df = pd.read_excel('your_data.xlsx')

# 2. 基础特征工程
X, y, feature_names, processed_data = create_features(df)

# 3. 快速训练（使用默认参数）
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 4. 预测
probabilities = model.predict_proba(X_test_scaled)[:, 1]
print(f"预测完成，转化概率范围：{probabilities.min():.3f} - {probabilities.max():.3f}")
```

## 性能优化建议

### 大数据处理
```python
# 对于大型数据集，可以分批处理
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # 处理每个数据块
    process_chunk(chunk)
```

### 内存优化
```python
# 优化数据类型以节省内存
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df
```

## 常见问题

### Q1: 模型训练很慢怎么办？
**A:** 
- 减少数据集大小进行测试
- 减少交叉验证折数
- 使用更少的模型数量

### Q2: 预测结果不理想？
**A:**
- 检查数据质量和特征工程
- 调整模型参数
- 增加训练数据

### Q3: 内存不足？
**A:**
- 使用数据分批处理
- 优化数据类型
- 减少模型数量

### Q4: SHAP分析出错？
**A:**
- 确保SHAP版本兼容
- 减少样本数量进行SHAP分析
- 检查特征数据类型

## 技术支持

遇到问题时：

1. **检查错误日志**：仔细阅读错误信息
2. **查看GitHub Issues**：搜索类似问题
3. **创建新Issue**：详细描述问题和环境信息
4. **提供复现步骤**：帮助快速定位问题

## 下一步

安装完成后，建议：

1. 阅读完整的 `README.md`
2. 查看代码注释了解详细实现
3. 尝试使用自己的数据进行训练
4. 探索不同的模型参数和策略

祝你使用愉快！🚀 