# Translation Checker 安装指南

## 系统要求

### 硬件要求
- **内存**: 8GB+ (推荐 16GB+)
- **存储**: 2GB+ 可用空间
- **GPU**: 可选，用于加速相似度计算

### 软件要求
- **Python**: 3.8 或更高版本
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)

## 安装步骤

### 1. 克隆或下载项目

```bash
git clone https://github.com/your-username/translation-checker.git
cd translation-checker
```

### 2. 创建虚拟环境 (推荐)

```bash
# 使用 venv
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量 (可选)

复制环境变量示例文件：
```bash
cp .env.example .env
```

编辑 `.env` 文件，添加你的API密钥：
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. 验证安装

运行演示脚本验证安装：
```bash
python run_demo.py
```

## 依赖说明

### 核心依赖
- **pandas**: 数据处理
- **openpyxl**: Excel文件读写
- **sentence-transformers**: 语义相似度计算
- **transformers**: 预训练模型支持
- **torch**: 深度学习框架

### 可选依赖
- **openai**: OpenAI API支持
- **matplotlib/seaborn**: 图表生成
- **rich**: 美化命令行输出

## 常见问题

### 1. 安装 sentence-transformers 失败

**问题**: 网络连接问题或依赖冲突

**解决方案**:
```bash
# 使用清华源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence-transformers

# 或者分步安装
pip install torch torchvision
pip install sentence-transformers
```

### 2. 内存不足错误

**问题**: 模型加载时内存不足

**解决方案**:
- 关闭其他程序释放内存
- 在配置文件中调整 `batch_size` 参数
- 使用更小的模型

### 3. GPU 相关问题

**问题**: CUDA 版本不匹配或GPU不可用

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. 权限问题

**问题**: Windows上文件权限错误

**解决方案**:
- 以管理员身份运行命令提示符
- 或者安装到用户目录：`pip install --user`

## 性能优化

### 1. GPU 加速

如果有NVIDIA GPU，安装CUDA版本的PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 模型缓存

首次运行时会下载模型，后续运行会使用缓存。确保有足够的存储空间。

### 3. 批处理大小

在 `config/config.yaml` 中调整批处理大小：
```yaml
detection:
  similarity:
    batch_size: 16  # 根据内存情况调整
```

## 开发环境设置

### 安装开发依赖

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black src/ tests/
flake8 src/ tests/
```

## Docker 部署 (可选)

创建 Dockerfile：
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

构建和运行：
```bash
docker build -t translation-checker .
docker run -v $(pwd)/data:/app/data translation-checker
```

## 故障排除

### 查看日志

检查 `logs/` 目录下的日志文件获取详细错误信息。

### 重置配置

删除配置文件使用默认配置：
```bash
rm config/config.yaml
```

### 清理缓存

清理模型缓存：
```bash
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/
```

## 获取帮助

如果遇到问题，请：

1. 查看 [FAQ](FAQ.md)
2. 搜索 [Issues](https://github.com/your-username/translation-checker/issues)
3. 提交新的 Issue
4. 联系支持邮箱: support@translationchecker.com
