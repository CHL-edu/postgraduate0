#处理json保留100项
from google.colab import drive
drive.mount('/content/drive')
import json
import csv
import os

# 路径设置
JSON_PATH = '/content/drive/MyDrive/Colab Notebooks/BERT/Fine_turning/ci.json'
CSV_PATH = '/content/drive/MyDrive/Colab Notebooks/BERT/Fine_turning/ci-100.csv'
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)  # 确保目录存在

# 读取JSON数据
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 假设data是列表格式

# 提取前100项的ci字段
ci_list = []
for item in data[:100]:  # 只处理前100项
    if isinstance(item, dict) and 'ci' in item:
        ci_list.append([item['ci']])  # 每行作为列表元素

# 写入CSV
with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(ci_list)  # 写入数据

print(f"前100项数据已保存至 {CSV_PATH}")
