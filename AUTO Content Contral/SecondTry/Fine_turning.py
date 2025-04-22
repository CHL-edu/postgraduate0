# 自定义CSV词汇添加到BERT分词器
import pandas as pd
from transformers import BertModel
import os

# 定义路径
ORIGIN_BERT_PATH = '/content/drive/MyDrive/Colab Notebooks/BERT/Fine_turning/Bert-Chinese_base'
CSV_PATH = '/content/drive/MyDrive/Colab Notebooks/BERT/Fine_turning/ci-100.csv'
SAVE_PATH = '/content/drive/MyDrive/Colab Notebooks/BERT/Fine_turning/Tokenizer-ci'

try:
    # 加载预训练分词器
    if not os.path.exists(ORIGIN_BERT_PATH):
        raise FileNotFoundError(f"BERT model path not found: {ORIGIN_BERT_PATH}")
    tokenizer = BertTokenizerFast.from_pretrained(ORIGIN_BERT_PATH)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
    exit(1)

try:
    # 加载 ci.csv 文件中的领域词汇
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # 验证CSV文件是否为空
    if df.empty:
        raise ValueError("CSV file is empty.")
    
    # 假设词汇在第一列，检查列是否存在
    if df.shape[1] < 1:
        raise ValueError("CSV file has no columns.")
    new_tokens = df.iloc[:, 0].dropna().tolist()  # 提取第一列，去除空值并转为列表
except Exception as e:
    print(f"Error processing CSV file: {str(e)}")
    exit(1)

try:
    # 添加新词汇到分词器
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} new tokens to the tokenizer.")
except Exception as e:
    print(f"Error adding tokens to tokenizer: {str(e)}")
    exit(1)

try:
    # 保存分词器
    os.makedirs(SAVE_PATH, exist_ok=True)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"Tokenizer saved to {SAVE_PATH}")
except Exception as e:
    print(f"Error saving tokenizer: {str(e)}")
    exit(1)
