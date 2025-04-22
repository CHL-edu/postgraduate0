#本地路径加载
from google.colab import drive
from transformers import BertModel
import os

# 1. 挂载Google Drive
drive.mount('/content/drive')

# 2. 定义模型路径（请根据实际情况修改）
model_path = '/content/drive/MyDrive/Colab Notebooks/BERT/Fine_turning/Bert-Chinese_base'

# 3. 验证路径是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型路径不存在: {model_path}")

# 4. 验证必要的模型文件是否存在
required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]

if missing_files:
    raise FileNotFoundError(f"缺少必要的模型文件: {missing_files}")

# 5. 打印验证通过信息
print("验证通过，模型文件完整:")
for f in required_files:
    print(f"- {f} ✔")

# 6. 加载模型
try:
    print("\n正在加载模型...")
    model = BertModel.from_pretrained(model_path)
    print("模型加载成功!")
    
    # 7. 打印模型信息
    print("\n模型架构:")
    print(model.config)
    
except Exception as e:
    print(f"\n模型加载失败: {str(e)}")
    print("\n可能的原因:")
    print("1. 模型文件损坏")
    print("2. 文件权限问题")
    print("3. 路径中包含特殊字符或空格")
    print("4. Transformers版本不兼容")
    print("\n建议解决方案:")
    print("1. 重新克隆模型仓库")
    print("2. 检查并修复文件权限")
    print("3. 尝试简化路径名称")
    print("4. 更新transformers库: !pip install --upgrade transformers")
