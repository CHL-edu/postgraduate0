#test above Fine_turning learing what is deferent
from transformers import BertTokenizerFast
import os
# 加载预训练分词器
SAVE_PATH = '/content/drive/MyDrive/Colab Notebooks/BERT/Fine_turning/Tokenizer'
try:
  tokenizer = BertTokenizerFast.from_pretrained(SAVE_PATH)
  #tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
except:
  print("tokenizer not found")

text = "奇怪的乌鸦需要喝水"
tokens = tokenizer(text, return_tensors="pt")
print(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))
