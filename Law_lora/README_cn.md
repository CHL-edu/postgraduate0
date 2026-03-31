# EmbeddingRAG - 基于 FAISS 的法律文档智能检索系统

一个基于 FAISS 向量索引和 Ollama 本地嵌入模型的 RAG（检索增强生成）系统，专为中文法律文档（民法、劳动法）的智能检索而设计。

## ✨ 核心特性

- **🎯 长文本分段匹配**：智能切分长查询文本，分段检索后聚合结果，突破单次查询长度限制
- **🧠 本地 Embedding 模型**：使用 Ollama 的 `qwen3-embedding:0.6b` 模型，完全离线运行
- **⚡ FAISS 高性能检索**：基于 Facebook FAISS 库的向量相似度搜索，快速高效
- **🔍 智能去重聚合**：自动合并多片段检索结果，按相似度重排序
- **📚 专为法律文本优化**：针对中文法律文本特点设计，支持章节条号结构化展示

## 🏗️ 系统架构

```
原始法律文本 (Markdown) → JSON 格式化 → Embedding 生成 → FAISS 索引 → 用户查询
                                      ↓
                            分段匹配 → 并行检索 → 结果聚合 → 返回相关法条
```

## 📦 安装部署

### 环境要求

- Python 3.12 或更高版本
- Ollama（本地运行，需安装 `qwen3-embedding`类模型）

### 安装依赖

```bash
pip install -r requirement.txt
```

### 安装并启动 Ollama

1. 安装 [Ollama](https://ollama.com/)
2. 下载 embedding 模型：

```bash
ollama pull qwen3-embedding:0.6b
```

3. 启动 Ollama 服务：

```bash
ollama serve
```

验证服务运行正常：

```bash
curl http://localhost:11434/api/embeddings -d '{"model":"qwen3-embedding:0.6b","prompt":"测试"}'
```

## 🚀 快速开始

### 1️⃣ 准备法律数据

将法律文本转换为 JSON 格式（`lawdata/` 目录下）：

```json
[
  {
    "id": "1",
    "law_name": "劳动法",
    "chapter": "第一章",
    "section": "第一章 总则",
    "article_number": "第一条",
    "content": "为了保护..."
  }
]
```

### 2️⃣ 生成 Embedding

编辑 `ollama_embedding.py` 中的路径：

```python
INPUT_JSON_PATH = "lawdata/labour_law_full.json"
OUTPUT_JSON_PATH = "lawdata/labour_law_embedding_full.json"
```

运行生成脚本：

```bash
python3 ollama_embedding.py
```

### 3️⃣ 合并多个法律库（可选）

如需将多个法律文件合并为一个知识库：

编辑 `combinejson.py` 中的文件列表：

```python
INPUT_FILES = [
    "lawdata/labour_law_embedding_full.json",
    "lawdata/civil_law_embedding_full.json"
]
OUTPUT_FILE = "lawdata/combined_law_embedding_full.json"
```

运行合并脚本：

```bash
python3 combinejson.py
```

### 4️⃣ 启动 RAG 检索系统

确保 `config.py` 中知识库路径正确：

```python
KNOWLEDGE_BASE_PATH = "lawdata/combined_law_embedding_full.json"
```

运行主程序：

```bash
python3 faiss_RAG.py
```

进入交互式问答界面：

```
================================================================================
FAISS RAG 长文本分段匹配检索系统
================================================================================

系统已就绪！输入您的问题（输入 'quit' 或 'exit' 退出）
================================================================================

请输入问题: 
```

## ⚙️ 配置说明

### 核心配置（`config.py`）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `OLLAMA_API_URL` | Ollama API 地址 | `http://localhost:11434/api/embeddings` |
| `EMBEDDING_MODEL` | Embedding 模型名称 | `qwen3-embedding:0.6b` |
| `KNOWLEDGE_BASE_PATH` | 知识库 JSON 文件路径 | `lawdata/combined_law_embedding_full.json` |

### 分段参数（`faiss_RAG.py` 顶部）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `SEGMENT_MAX_LENGTH` | 单段最大字符数 | 500 |
| `SEGMENT_MIN_LENGTH` | 单段最小字符数 | 200 |
| `SEGMENT_OVERLAP` | 段间重叠字符数 | 50 |
| `SEGMENT_THRESHOLD` | 触发分段的文本长度阈值 | 500 |

### 检索参数（`faiss_RAG.py` 顶部）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `TOP_K_PER_SEGMENT` | 每个片段检索的 Top-K | 5 |
| `TOP_K_FINAL` | 最终返回的 Top-N | 5 |
| `SIMILARITY_THRESHOLD` | 相似度过滤阈值 | 0.5 |
| `AGG_METHOD` | 相似度聚合策略（`"max"` 或 `"avg"`） | `"max"` |

## 📁 项目结构

```
embedingRAG/
├── faiss_RAG.py              # 主 RAG 系统（分段匹配核心）
├── ollama_embedding.py       # Embedding 生成（调用 Ollama API）
├── combinejson.py            # 合并多个 JSON 文件
├── json_add_label.py         # 为 JSON 条目添加字段
├── config.py                 # 集中配置文件
├── test.py                   # 配置测试脚本
├── faiss_test_tensordatabase.py  # 原始简单 RAG 实现(原型测试)
├── lawdata/                 # 法律数据目录
│   ├── combined_law_embedding_full.json  # 合并后的完整知识库
│   ├── civil_law_embedding_full.json     # 民法典（含 embedding）
│   └── labour_law_embedding_full.json    # 劳动法（含 embedding）
└── README_cn.md             # 本文档
```

## 🎯 使用示例

### 原始法条 md 转换为 JSON 格式

```bash
python3 ./lawdata/original_resource/civillawtojson.py
```

### JSON 格式法条修改label

```bash
python3 json_add_label.py
```

### 生成 Embedding

```bash
python3 ollama_embedding.py
```

### 合并多个法律库生成总数据库

```bash
python3 combinejson.py
```

### 启动 RAG 检索系统

```bash
python3 faiss_RAG.py
```


## 🎯 示例展示

### 短文本查询（直接检索）

```
请输入问题: 劳动合同试用期最长多久？

找到 3 条相关知识（相似度阈值: 0.5）：
================================================================================

【结果 1】相似度: 0.8567 | 距离: 0.1433
章节: 第四章+第十九条
法条: 劳动合同期限三个月以上不满一年的，试用期不得超过一个月...
---------------------------------------------------------------------------------
```

### 长文本查询（分段匹配）

```
请输入问题: 我在公司工作了三年，最近因为绩效问题被辞退，公司没有提前通知也没有支付赔偿金，请问公司的做法合法吗？我应该如何维护自己的权益？

查询文本较长（47 字符），分段为 2 个片段
批量检索完成，获得 10 条原始结果
聚合完成，最终返回 5 条结果

找到 5 条相关知识（相似度阈值: 0.5）：
================================================================================

【结果 1】相似度: 0.7823 | 距离: 0.2187
匹配片段数: 2 | 片段索引: [0, 1]
法律: 劳动法
章节: 第五章+第四十条
法条: 有下列情形之一的，用人单位提前三十日以书面形式通知劳动者本人...
---------------------------------------------------------------------------------
```

## 🔧 技术细节

### 文本分段策略

1. **优先级断句**：寻找句子结束符（`。！？`）
2. **次级断句**：寻找逗号或分号（`，；`）
3. **强制断句**：如无标点，强制在最大长度处断开
4. **重叠窗口**：段间保留 50 字符重叠，保证上下文连续性

### 去重逻辑

- **唯一键**：`(article_number, section)` 元组
- **相似度聚合**：取所有片段匹配的最大相似度
- **来源追踪**：记录哪些片段索引匹配了该条法条

### FAISS 配置

- **索引类型**：`IndexFlatL2`（精确 L2 距离搜索）
- **向量归一化**：使用 `normalize_L2()` 实现余弦相似度等价
- **相似度计算**：`similarity = 1 / (1 + distance)`

## ⚠️ 注意事项

1. **Ollama 必须运行**：生成 embedding 和执行检索前，确保 Ollama 服务已启动
2. **Embedding 生成耗时**：处理大量法律文本时需耐心等待
3. **中文优化**：系统专为中文法律文本设计，分段逻辑使用中文标点


## 🛠️ 故障排除

### Ollama 连接失败

```bash
# 检查 Ollama 服务状态
curl http://localhost:11434/api/tags

# 重启 Ollama
ollama serve
```

### FAISS 索引错误

确保知识库 JSON 文件包含有效的 `embedding` 字段：

```bash
python3 -c "import json; data=json.load(open('lawdata/combined_law_embedding_full.json')); print(f'总记录数: {len(data)}'); print(f'含embedding: {sum(1 for x in data if \"embedding\" in x)}')"
```

### 查询无结果

尝试降低相似度阈值（`faiss_RAG.py` 中的 `SIMILARITY_THRESHOLD`）：

```python
SIMILARITY_THRESHOLD = 0.3  # 从 0.5 降至 0.3
```
## 📝 更新日志
**作者**：CHL-edu
**创建日期**：2026-01-14
**最后更新**：2026-01-16
参考 /data/chl/code/LawLLM_submit/Chat_test.py和/data/chl/code/LawLLM_submit/faiss_RAG.py，实现
1、基于RAG外挂法典的法条检索。
2、实现微调LLM读取法律卷宗等并输出格式化隐含裁判规则的对应内容。
