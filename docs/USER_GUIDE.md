# Translation Checker 用户指南

## 快速开始

### 1. 运行演示

首次使用建议先运行演示：

```bash
python run_demo.py
```

这会创建示例文件并展示完整的检查流程。

### 2. 检查单个文件

```bash
python main.py check --file your_translation.xlsx
```

### 3. 查看帮助

```bash
python main.py --help
python main.py check --help
```

## 命令行使用

### 基本检查

```bash
# 检查Excel文件
python main.py check --file translation.xlsx

# 指定输出文件
python main.py check --file translation.xlsx --output results.xlsx

# 选择输出格式
python main.py check --file translation.xlsx --format csv
```

### 高级选项

```bash
# 禁用相似度检测
python main.py check --file translation.xlsx --no-similarity

# 启用LLM评估（需要API密钥）
python main.py check --file translation.xlsx --llm

# 只对低相似度的翻译进行LLM评估
python main.py check --file translation.xlsx --llm --llm-filter "similarity_score < 0.7"

# 生成详细报告
python main.py check --file translation.xlsx --detailed-report
```

### 批量处理

```bash
# 批量处理目录中的所有Excel文件
python main.py batch --input-dir ./translations/

# 指定文件模式
python main.py batch --input-dir ./translations/ --pattern "*.xlsx"

# 指定输出目录
python main.py batch --input-dir ./translations/ --output-dir ./results/
```

### 系统状态

```bash
# 查看系统状态
python main.py status

# 使用自定义配置文件
python main.py status --config my_config.yaml
```

## 文件格式要求

### Excel文件结构

Translation Checker 支持以下Excel文件结构：

```
| 中文 | 英文 |
|------|------|
| 原文1 | 译文1 |
| 原文2 | 译文2 |
```

### 支持的列名

**源语言列** (中文):
- `中文`, `Chinese`, `Source`, `源文本`

**目标语言列** (英文):
- `英文`, `English`, `Target`, `翻译`, `译文`

### 文件格式

- `.xlsx` (推荐)
- `.xls`
- `.csv`

## 配置说明

### 主配置文件

编辑 `config/config.yaml` 来自定义检测参数：

```yaml
# 相似度阈值
detection:
  similarity:
    threshold: 0.7  # 调整相似度阈值

# 长度比例检查
detection:
  rules:
    length_ratio_min: 0.3  # 最小长度比例
    length_ratio_max: 3.0  # 最大长度比例
```

### 关键词字典

编辑 `config/keywords.yaml` 来自定义术语检查：

```yaml
terminology:
  "人工智能": ["artificial intelligence", "AI"]
  "机器学习": ["machine learning", "ML"]

brands:
  - "Microsoft"
  - "Google"

forbidden_words:
  - "机翻"
  - "[翻译]"
```

### 环境变量

创建 `.env` 文件配置API密钥：

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 检测功能详解

### 1. 基础规则检测

**长度比例检查**
- 检查译文长度是否合理
- 可配置最小/最大比例

**空值检查**
- 检测缺失的翻译
- 标记空白内容

**特殊字符检查**
- 检测机器翻译标记
- 识别异常字符重复

**术语一致性**
- 验证专业术语翻译
- 检查品牌名称一致性

### 2. 语义相似度检测

**工作原理**
- 使用多语言sentence-transformers模型
- 计算中英文语义向量的余弦相似度
- 识别语义偏差较大的翻译

**模型选择**
- 默认：`paraphrase-multilingual-MiniLM-L12-v2`
- 可在配置中更换其他模型

### 3. LLM质量评估

**评估维度**
- 准确性：是否准确传达原文意思
- 流畅性：译文是否自然流畅
- 一致性：术语是否一致
- 完整性：是否有遗漏或添加

**支持的LLM**
- OpenAI GPT系列
- 可扩展支持其他API

## 结果解读

### 输出文件结构

检查结果包含以下列：

**原始数据**
- 源文本和目标文本
- 原始行索引

**基础统计**
- `source_length`: 源文本长度
- `target_length`: 目标文本长度
- `length_ratio`: 长度比例

**规则检测结果**
- `rule_check_passed`: 是否通过规则检查
- `rule_check_issues`: 发现的规则问题

**相似度检测结果**
- `similarity_score`: 相似度分数 (0-1)
- `similarity_passed`: 是否通过相似度检查
- `similarity_message`: 相似度检查信息

**LLM评估结果**
- `llm_score`: LLM总体评分 (1-10)
- `llm_accuracy`: 准确性评分
- `llm_fluency`: 流畅性评分
- `llm_consistency`: 一致性评分
- `llm_completeness`: 完整性评分

**综合评估**
- `overall_score`: 综合评分 (0-10)
- `overall_status`: 质量等级 (excellent/good/fair/poor)
- `overall_issues`: 发现的所有问题

### 质量等级说明

- **优秀 (8-10分)**: 翻译质量很高，无明显问题
- **良好 (6-8分)**: 翻译质量较好，有轻微问题
- **一般 (4-6分)**: 翻译质量中等，需要改进
- **较差 (0-4分)**: 翻译质量较差，需要重新翻译

## 性能优化

### 批处理大小

根据内存情况调整批处理大小：

```yaml
detection:
  similarity:
    batch_size: 32  # 减少以节省内存
```

### GPU 加速

如果有GPU，会自动使用GPU加速相似度计算。

### 选择性检测

对于大文件，可以选择性启用检测功能：

```bash
# 只进行规则检测，跳过相似度计算
python main.py check --file large_file.xlsx --no-similarity

# 只对问题翻译进行LLM评估
python main.py check --file large_file.xlsx --llm --llm-filter "rule_check_passed == False"
```

## 常见使用场景

### 1. 翻译质量审核

```bash
# 全面检查，包含LLM评估
python main.py check --file translation.xlsx --llm --detailed-report
```

### 2. 快速质量筛查

```bash
# 只进行规则和相似度检查
python main.py check --file translation.xlsx --no-llm
```

### 3. 术语一致性检查

配置专业术语字典后：
```bash
python main.py check --file technical_translation.xlsx
```

### 4. 批量质量监控

```bash
# 定期批量检查
python main.py batch --input-dir ./daily_translations/ --output-dir ./quality_reports/
```

## 故障排除

### 常见错误

**1. 模型下载失败**
```bash
# 手动设置缓存目录
export HF_HOME=/path/to/cache
```

**2. 内存不足**
```bash
# 减少批处理大小
# 在config.yaml中设置 batch_size: 8
```

**3. API限制**
```bash
# 在LLM评估中添加延迟
# 代码会自动处理API限制
```

### 日志查看

检查 `logs/` 目录下的日志文件获取详细信息：

```bash
tail -f logs/translation_checker.log
```

## 最佳实践

### 1. 配置优化

- 根据翻译类型调整相似度阈值
- 维护完整的术语字典
- 定期更新禁用词汇列表

### 2. 工作流程

1. 先进行规则和相似度检查
2. 对问题翻译进行LLM评估
3. 根据报告进行有针对性的改进
4. 建立质量标准和检查流程

### 3. 结果分析

- 关注综合评分趋势
- 分析常见问题类型
- 根据建议改进翻译流程

## 扩展功能

### 自定义检测规则

可以在 `src/core/rule_checker.py` 中添加自定义检测规则。

### 支持新语言

修改配置文件中的列名映射来支持其他语言对。

### 集成到工作流

Translation Checker 可以集成到CI/CD流程中进行自动化质量检查。
