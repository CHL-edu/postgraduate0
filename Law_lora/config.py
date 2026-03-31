# ===================== 核心配置 =====================
import os

# 项目根目录（绝对路径）
BASE_DIR = "/data/chl/code/embeddingRAG"

# 数据目录
LAWDATA_PATH   = os.path.join(BASE_DIR, "lawdata")
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"  # Ollama默认API地址
EMBEDDING_MODEL = "qwen3-embedding:0.6b"                   # 你本地的embedding模型

# 民法典文件路径
CIVIL_LAW_INPUT = os.path.join(LAWDATA_PATH, "civil_law_full.json")
CIVIL_LAW_OUTPUT = os.path.join(LAWDATA_PATH, "civil_law_embedding_full.json")

# 劳动法文件路径
LABOUR_LAW_INPUT = os.path.join(LAWDATA_PATH, "labour_law_full.json")
LABOUR_LAW_OUTPUT = os.path.join(LAWDATA_PATH, "labour_law_embedding_full.json")

# 兼容旧配置（默认使用民法典）
INPUT_LAW_JSON_PATH = CIVIL_LAW_INPUT
OUTPUT_LAW_JSON_PATH = CIVIL_LAW_OUTPUT

# 知识库路径
KNOWLEDGE_BASE_PATH = os.path.join(LAWDATA_PATH, "combined_law_embedding_full.json")
class configinit():
    def __init__(self):
        self.OLLAMA_API_URL = OLLAMA_API_URL
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
        self.INPUT_LAW_JSON_PATH = LAWDATA_PATH+INPUT_LAW_JSON_PATH
        self.OUTPUT_LAW_JSON_PATH = LAWDATA_PATH+OUTPUT_LAW_JSON_PATH
        self.KNOWLEDGE_BASE_PATH  = KNOWLEDGE_BASE_PATH # 知识库文件
        self.testpath = LAWDATA_PATH + INPUT_LAW_JSON_PATH
    