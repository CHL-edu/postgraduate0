#test above Fine_turning learing what is deferent
from transformers import BertTokenizerFast
import os
# 加载预训练分词器
Ci100_PATH = '/content/drive/MyDrive/Colab Notebooks/BERT/Fine_turning/Tokenizer-ci'
try:
  tokenizer = BertTokenizerFast.from_pretrained(SAVE_PATH)
  #tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
except:
  print("tokenizer not found")

text = "晨夕晨乌大傻子"
tokens = tokenizer(text, return_tensors="pt")
print(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))
