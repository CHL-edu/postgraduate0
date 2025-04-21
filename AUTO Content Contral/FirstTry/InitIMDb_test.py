# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 导入库
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from googletrans import Translator
# 定义模型和分词器保存路径
model_save_path = '/content/drive/MyDrive/Colab Notebooks/BERT/IMDb/InitModal'

# 加载训练好的模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForSequenceClassification.from_pretrained(model_save_path)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置为评估模式

# 定义推理函数
def predict_sentiment(text, max_length=512):
    """
    对输入文本进行情感分类。
    参数：
        text (str): 输入的文本
        max_length (int): 分词的最大长度，默认为 512
    返回：
        dict: 包含预测标签、情感类别和概率
    """
    # 预处理文本
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"  # 返回 PyTorch 张量
    )

    # 将输入移动到 GPU（如果可用）
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 进行推理
    with torch.no_grad():  # 禁用梯度计算以加速推理
        outputs = model(**inputs)
        logits = outputs.logits

    # 计算概率
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    predicted_label = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    # 映射标签到情感类别
    label_map = {0: "负面", 1: "正面"}
    sentiment = label_map[predicted_label]

    # 返回结果
    return {
        "text": text,
        "label": predicted_label,
        "sentiment": sentiment,
        "positive_prob": float(probabilities[1]),  # 正面概率
        "negative_prob": float(probabilities[0])   # 负面概率
    }

user_input = input("请输入要分析的文本：")
text_zh = user_input
translator = Translator()
translation = translator.translate(text_zh, dest='en')  # 翻译成英文
test_texts = translation.text  # 获取翻译后的文本字符串

result = predict_sentiment(test_texts)
print("输入文本:", result["text"])
print("预测情感:", result["sentiment"])
print(f"正面概率: {result['positive_prob']:.4f}, 负面概率: {result['negative_prob']:.4f}")
