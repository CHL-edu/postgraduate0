# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 安装必要的依赖
!pip install transformers torch deep-translator ipywidgets

# 导入库
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
from collections import Counter
from deep_translator import GoogleTranslator
from IPython.display import display
import ipywidgets as widgets

# 定义模型和分词器保存路径
model_save_path = '/content/drive/MyDrive/Colab Notebooks/BERT/IMDb/InitModal'

# 加载训练好的模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForSequenceClassification.from_pretrained(model_save_path)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置为评估模式

# 定义英文情感词典
sentiment_dict_en = {
    "positive": [
        "fantastic", "great", "wonderful", "amazing", "excellent",
        "love", "enjoy", "brilliant", "awesome", "fabulous",
        "happy", "delightful", "superb", "beautiful", "incredible"
    ],
    "negative": [
        "terrible", "awful", "bad", "boring", "disappointing",
        "poor", "hate", "dreadful", "worst", "horrible",
        "sad", "annoying", "pathetic", "lame", "stupid",
        "shabi", "idiot", "fool", "jerk"  # 扩展负面词汇
    ]
}

# 定义中文情感词典
sentiment_dict_zh = {
    "positive": [
        "精彩", "棒", "好", "喜欢", "优秀", "美妙", "开心",
        "了不起", "迷人", "愉快", "卓越", "完美", "赞"
    ],
    "negative": [
        "糟糕", "差", "无聊", "失望", "讨厌", "恶心", "垃圾",
        "悲伤", "烦人", "烂", "愚蠢", "沙比", "白痴", "傻逼"
    ]
}

# 字典-based情感分析函数（支持英文和中文）
def dictionary_sentiment_analysis(text, lang="en"):
    text = text.lower()
    words = re.findall(r'\b[\w]+\b', text, re.UNICODE)  # 支持中文和英文分词
    word_counts = Counter(words)
    
    if lang == "zh":
        positive_words = sentiment_dict_zh["positive"]
        negative_words = sentiment_dict_zh["negative"]
    else:
        positive_words = sentiment_dict_en["positive"]
        negative_words = sentiment_dict_en["negative"]
    
    positive_count = sum(word_counts[word.lower()] for word in positive_words if word.lower() in word_counts)
    negative_count = sum(word_counts[word.lower()] for word in negative_words if word.lower() in word_counts)
    
    if positive_count > negative_count:
        sentiment = "正面"
        score = positive_count / (positive_count + negative_count + 1e-10)
    elif negative_count > positive_count:
        sentiment = "负面"
        score = negative_count / (positive_count + negative_count + 1e-10)
    else:
        sentiment = "中性"
        score = 0.5
    
    return {
        "sentiment": sentiment,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "score": score,
        "matched_words": [word for word in words if word.lower() in positive_words + negative_words]
    }

# BERT 模型推理函数
def predict_sentiment(text, max_length=512):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    predicted_label = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    label_map = {0: "负面", 1: "正面"}
    return {
        "text": text,
        "label": predicted_label,
        "sentiment": label_map[predicted_label],
        "positive_prob": float(probabilities[1]),
        "negative_prob": float(probabilities[0])
    }

# 翻译并对比函数
def translate_and_compare(text_zh):
    # 中文情感分析
    dict_result_zh = dictionary_sentiment_analysis(text_zh, lang="zh")
    
    # 翻译为英文
    try:
        translator = GoogleTranslator(source='zh-CN', target='en')
        text_en = translator.translate(text_zh)
    except Exception as e:
        return {"error": f"翻译失败: {str(e)}"}
    
    # 英文情感分析
    bert_result = predict_sentiment(text_en)
    dict_result_en = dictionary_sentiment_analysis(text_en, lang="en")
    
    # 对比结果
    agreement_en = bert_result["sentiment"] == dict_result_en["sentiment"]
    return {
        "text_zh": text_zh,
        "text_en": text_en,
        "bert_sentiment": bert_result["sentiment"],
        "bert_positive_prob": bert_result["positive_prob"],
        "bert_negative_prob": bert_result["negative_prob"],
        "dict_sentiment_en": dict_result_en["sentiment"],
        "dict_positive_count_en": dict_result_en["positive_count"],
        "dict_negative_count_en": dict_result_en["negative_count"],
        "dict_score_en": dict_result_en["score"],
        "dict_matched_words_en": dict_result_en["matched_words"],
        "dict_sentiment_zh": dict_result_zh["sentiment"],
        "dict_positive_count_zh": dict_result_zh["positive_count"],
        "dict_negative_count_zh": dict_result_zh["negative_count"],
        "dict_score_zh": dict_result_zh["score"],
        "dict_matched_words_zh": dict_result_zh["matched_words"],
        "agreement_en": agreement_en,
        "comment_en": "一致" if agreement_en else "不一致"
    }

# 交互界面
text_input = widgets.Textarea(description="输入中文文本:", layout={'width': '500px', 'height': '100px'})
button = widgets.Button(description="分析")
output = widgets.Output()

def on_button_clicked(b):
    with output:
        output.clear_output()
        if not text_input.value.strip():
            print("请输入文本！")
            return
        result = translate_and_compare(text_input.value)
        if "error" in result:
            print(result["error"])
            return
        print(f"输入中文: {result['text_zh']}")
        print(f"翻译英文: {result['text_en']}")
        print(f"中文字典情感: {result['dict_sentiment_zh']} (正面: {result['dict_positive_count_zh']}, 负面: {result['dict_negative_count_zh']}, 得分: {result['dict_score_zh']:.4f})")
        print(f"匹配词汇: {result['dict_matched_words_zh']}")
        print(f"BERT 预测情感: {result['bert_sentiment']} (正面概率: {result['bert_positive_prob']:.4f}, 负面概率: {result['bert_negative_prob']:.4f})")
        print(f"英文字典情感: {result['dict_sentiment_en']} (正面: {result['dict_positive_count_en']}, 负面: {result['dict_negative_count_en']}, 得分: {result['dict_score_en']:.4f})")
        print(f"匹配词汇: {result['dict_matched_words_en']}")
        print(f"BERT vs 英文字典对比: {result['comment_en']}")

button.on_click(on_button_clicked)
display(text_input, button, output)

# 示例调用
test_texts_zh = [
    "你个沙比",
    "这部电影非常精彩，我非常喜欢！",
    "这个电影很无聊，浪费时间。",
    "剧情还可以，但角色不太吸引人。"
]
for text in test_texts_zh:
    result = translate_and_compare(text)
    if "error" in result:
        print(result["error"])
        continue
    print(f"输入中文: {result['text_zh']}")
    print(f"翻译英文: {result['text_en']}")
    print(f"中文字典情感: {result['dict_sentiment_zh']} (正面: {result['dict_positive_count_zh']}, 负面: {result['dict_negative_count_zh']}, 得分: {result['dict_score_zh']:.4f})")
    print(f"匹配词汇: {result['dict_matched_words_zh']}")
    print(f"BERT 预测情感: {result['bert_sentiment']} (正面概率: {result['bert_positive_prob']:.4f}, 负面概率: {result['bert_negative_prob']:.4f})")
    print(f"英文字典情感: {result['dict_sentiment_en']} (正面: {result['dict_positive_count_en']}, 负面: {result['dict_negative_count_en']}, 得分: {result['dict_score_en']:.4f})")
    print(f"匹配词汇: {result['dict_matched_words_en']}")
    print(f"BERT vs 英文字典对比: {result['comment_en']}")
    print("-" * 60)
