# Translation Checker

一个基于机器学习的翻译质量检查工具，用于验证翻译的准确性和一致性。

## 功能特性

- 📊 Excel翻译文件批量处理
- 🔍 多层次翻译质量检测
- 🎯 关键词规范性验证
- 📈 语义相似度分析
- 🤖 LLM深度质量评估
- 📋 详细问题报告生成

## 系统架构

```
数据输入 → 预处理 → 规则检测 → 语义检测 → LLM评估 → 结果输出
```

### 检测层级

1. **基础规则检测**：关键词一致性、格式规范、长度合理性
2. **语义相似度检测**：使用sentence-transformers计算语义相似度
3. **LLM质量评估**：对可疑翻译进行深度分析

## 安装和使用

### 环境要求

- Python 3.12+
- 32GB+ 内存推荐
- GPU支持（可选，用于加速模型推理）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```bash
# 检查单个Excel文件
python main.py check --file translation.xlsx

# 批量检查多个文件
python main.py batch --input-dir ./translations/

# 生成详细报告
python main.py check --file translation.xlsx --detailed-report
```

## 配置

编辑 `config.yaml` 文件来自定义检测参数：

- 相似度阈值
- 关键词规则
- LLM API配置
- 输出格式设置

## 项目结构

```
TranslationChecker/
├── src/
│   ├── core/           # 核心检测模块
│   ├── processors/     # 数据处理模块
│   ├── models/         # 模型相关
│   └── utils/          # 工具函数
├── config/             # 配置文件
├── tests/              # 测试文件
├── examples/           # 示例文件
└── docs/               # 文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

```bash
python run_demo.py
```

### 3. 检查翻译文件

```bash
# 基础检查
python main.py check --file your_translation.xlsx

# 启用LLM评估（需要OpenAI API密钥）
python main.py check --file your_translation.xlsx --llm

# 批量处理
python main.py batch --input-dir ./translations/
```

### 4. 配置API密钥（可选）

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，添加你的OpenAI API密钥
OPENAI_API_KEY=your_api_key_here
```

## 主要特性

### ✅ 已实现功能

- **📊 Excel文件处理**: 支持 .xlsx, .xls, .csv 格式
- **🔍 基础规则检测**: 长度比例、空值、特殊字符、术语一致性
- **🧠 语义相似度分析**: 使用多语言sentence-transformers模型
- **🤖 LLM质量评估**: 支持OpenAI GPT进行深度分析
- **📈 综合评分系统**: 多维度质量评估和分级
- **📋 详细报告生成**: Excel、CSV、JSON格式输出
- **⚡ 批量处理**: 支持目录级批量检查
- **🎛️ 灵活配置**: 可自定义检测参数和规则

### 🚀 核心优势

- **分层检测架构**: 从基础规则到AI评估的渐进式质量检查
- **高性能处理**: 支持GPU加速和批量优化
- **可扩展设计**: 易于添加新的检测规则和语言支持
- **用户友好**: 丰富的命令行界面和详细的进度显示

## 开发进度

- [x] 项目框架搭建
- [x] Excel数据处理
- [x] 基础规则检测
- [x] 语义相似度分析
- [x] LLM集成
- [x] 报告生成
- [x] CLI接口
- [x] 批量处理
- [x] 配置管理
- [x] 测试用例
- [ ] 多语言支持扩展
- [ ] Web界面
- [ ] API服务

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License
