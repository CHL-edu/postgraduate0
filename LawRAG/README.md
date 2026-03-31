# FAISS-based Legal RAG System

A Retrieval-Augmented Generation (RAG) system for Chinese legal document retrieval, built with FAISS vector indexing and Ollama's `qwen3-embedding:0.6b` embedding model.

## Features

- **Segmented matching for long texts**: Intelligently splits long input into segments, performs parallel retrieval for each segment, then aggregates and deduplicates results
- **Local embedding generation**: Uses Ollama's `qwen3-embedding:0.6b` model — no external API required
- **FAISS vector search**: Efficient similarity search with L2 distance metric
- **Chinese legal text support**: Segmentation logic optimized for Chinese punctuation

## Quick Start

### Prerequisites

1. Python 3.12+
2. Ollama running locally with the `qwen3-embedding:0.6b` model:
   ```bash
   ollama pull qwen3-embedding:0.6b
   ollama serve
   ```
3. Install dependencies:
   ```bash
   pip install faiss-cpu numpy requests
   ```

### Run the RAG System

```bash
python faiss_RAG.py
```

### Generate Embeddings

```bash
python ollama_embedding.py
```

### Combine Multiple JSON Files

```bash
python combinejson.py
```

## Project Structure

```
.
├── faiss_RAG.py              # Main RAG system (segmented matching)
├── ollama_embedding.py       # Embedding generation via Ollama
├── combinejson.py            # Merge multiple JSON files
├── json_add_label.py         # Add fields to JSON entries
├── config.py                 # Centralized configuration
├── test.py                   # Simple config test
├── faiss_test_tensordatabase.py  # Original simple RAG (reference)
└── lawdata/                  # Legal text data
    ├── combined_law_embedding_full.json  # Main knowledge base
    ├── civil_law_embedding_full.json
    └── labour_law_embedding_full.json
```

## Configuration

All configuration is centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OLLAMA_API_URL` | `http://localhost:11434/api/embeddings` | Ollama API endpoint |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model name |
| `KNOWLEDGE_BASE_PATH` | `lawdata/combined_law_embedding_full.json` | Knowledge base file |

### Segmentation Parameters (in `faiss_RAG.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEGMENT_MAX_LENGTH` | 500 | Max characters per segment |
| `SEGMENT_MIN_LENGTH` | 200 | Min characters per segment |
| `SEGMENT_OVERLAP` | 50 | Overlap between segments |
| `SEGMENT_THRESHOLD` | 500 | Text length triggering segmentation |
| `TOP_K_PER_SEGMENT` | 5 | Results per segment |
| `TOP_K_FINAL` | 5 | Final results returned |
| `SIMILARITY_THRESHOLD` | 0.5 | Minimum similarity for results |
| `AGG_METHOD` | `"max"` | Aggregation strategy (`"max"` or `"avg"`) |

## Data Format

### Knowledge Base JSON Structure

```json
[
  {
    "id": "1",
    "law_name": "劳动法",
    "chapter": "第一章",
    "section": "第一章 总则",
    "article_number": "第一条",
    "content": "为了保护...",
    "embedding": [0.123, 0.456, ...]
  }
]
```

### Retrieval Result Format

```python
{
    "content": "法条内容",
    "article_number": "第十二条",
    "section": "第四章",
    "similarity": 0.8567,
    "distance": 0.1433,
    "match_count": 3,
    "source_segments": [0, 2]
}
```

## How It Works

1. **Text Splitting**: Long text is split at Chinese sentence boundaries (`。！？`), with fallback to comma/semicolon boundaries and overlap for context continuity
2. **Batch Retrieval**: Each segment is embedded via Ollama and searched against the FAISS index in parallel
3. **Result Aggregation**: Results are deduplicated using `(article_number, section)` as the unique key, with similarity aggregated via max (or avg) across segments

## Citation

If you use this work, please cite the following papers:

```bibtex
@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}

@ARTICLE{11202651,
  author={Douze, Matthijs and Guzhva, Alexandr and Deng, Chengqi and Johnson, Jeff and Szilvasy, Gergely and Mazaré, Pierre-Emmanuel and Lomeli, Maria and Hosseini, Lucas and Jégou, Hervé},
  journal={IEEE Transactions on Big Data},
  title={The Faiss Library},
  year={2026},
  volume={12},
  number={2},
  pages={346-361},
  keywords={Vectors;Libraries;Databases;Indexing;Measurement;Media;Quantization (signal);Partitioning algorithms;Feature extraction;Big Data;Vector search;data compression;quantization;information retrieval;numerical library},
  doi={10.1109/TBDATA.2025.3618474}
}
```
