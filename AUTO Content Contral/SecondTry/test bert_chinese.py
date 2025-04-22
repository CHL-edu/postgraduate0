#test Launch Fine_turning
from transformers import BertTokenizerFast
import os
# 加载预训练分词器
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

# 准备领域词汇（手动或从语料提取）
new_tokens = ["奇怪", "乌鸦"]  # 示例词汇

# 添加新词汇
tokenizer.add_tokens(new_tokens)

# 保存分词器
SAVE_PATH = '/content/drive/MyDrive/Colab Notebooks/BERT/Fine_turning/Tokenizer'
os.makedirs(SAVE_PATH, exist_ok=True)
tokenizer.save_pretrained(SAVE_PATH)
