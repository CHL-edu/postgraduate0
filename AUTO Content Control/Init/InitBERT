# 挂载 Google Drive 用于保存模型
from google.colab import drive
drive.mount('/content/drive')

# 安装必要的依赖
!pip install --upgrade transformers datasets torch evaluate

# 导入库
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

# 加载 TensorBoard 扩展用于可视化
%load_ext tensorboard

# 加载预训练 BERT 模型和分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 加载和预处理 IMDb 数据集
dataset = load_dataset("imdb")

# 定义分词函数
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# 应用分词并格式化数据集
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 分割训练和测试集（小型子集以加速训练）
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",              # 输出目录
    eval_strategy="epoch",              # 每个 epoch 进行评估
    learning_rate=2e-5,                 # 学习率
    per_device_train_batch_size=8,      # 训练批次大小
    per_device_eval_batch_size=8,       # 评估批次大小
    num_train_epochs=3,                 # 训练轮数
    weight_decay=0.01,                  # 权重衰减
    logging_dir="./logs",               # TensorBoard 日志目录
    logging_steps=10,                   # 每 10 步记录一次日志
    report_to="none",                   # 禁用 W&B 等外部日志
    save_strategy="epoch",              # 每个 epoch 保存模型
    load_best_model_at_end=True,        # 训练结束时加载最佳模型
    metric_for_best_model="accuracy"    # 根据准确率选择最佳模型
)

# 定义计算指标函数
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# 启动 TensorBoard 可视化
%tensorboard --logdir ./logs

# 开始训练
trainer.train()

# 保存模型和分词器
model_save_path = '/content/drive/MyDrive/Colab Notebooks/BERT/IMDb/InitModal'
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# 评估模型
eval_results = trainer.evaluate()
print("评估结果:", eval_results)
