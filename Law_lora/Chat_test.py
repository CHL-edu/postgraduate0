# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------------- 核心配置 --------------------------
# 本地模型路径
MODEL_PATH = "./Law_docjudge_distill"

# 是否使用量化（降低显存占用，建议开启）
USE_QUANTIZATION = False
# 设备配置（自动检测 GPU/CPU）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 对话参数
MAX_NEW_TOKENS = 2048  # 最大生成字数
MAX_TOTAL_TOKENS = 10000  # 最大总字数
TEMPERATURE = 0.7      # 生成随机性（0-1，越小越固定）
TOP_P = 0.90           # 采样概率

def load_model_and_tokenizer():
    """加载本地模型和分词器"""
    print(f"[INFO] 开始加载模型，路径：{MODEL_PATH}，设备：{DEVICE}")
    
    # 量化配置（可选，降低显存占用）
    quantization_config = None
    if USE_QUANTIZATION and DEVICE == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 4位量化（也可设为 load_in_8bit=True）
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="fp4",  # 替换为 fp4，兼容低版本 CUDA
            bnb_4bit_use_double_quant=True
        )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,  # Qwen 需要开启
        padding_side="right"
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"  # 自动分配模型到可用设备
    )
    
    # 禁用梯度计算，提升速度
    model.eval()
    print(f"[INFO] 模型加载完成！")
    return tokenizer, model

def build_dialog_prompt(history):
    """构建 Qwen 格式的对话提示词"""
    prompt = ""
    for turn in history:
        user_msg, assistant_msg = turn
        prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
    return prompt

def interactive_chat(tokenizer, model):
    """交互式对话主函数"""
    print("\n" + "-"*50)
    print("Qwen3-8B 交互式对话已启动（输入 'exit' 或 'quit' 退出）")
    print("-"*50 + "\n")
    
    # 对话历史存储
    history = []
    
    while True:
        # 获取用户输入
        user_input = input("你：").strip()
        
        # 退出条件
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("助手：再见！")
            break
        
        if not user_input:
            print("助手：请输入有效的对话内容！")
            continue
        
        # 构建对话历史（修复重复输入问题）
        history.append((user_input, ""))
        prompt = build_dialog_prompt(history[:-1])  # 只取历史对话，不含当前空轮次
        prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        try:
            # 编码输入
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOTAL_TOKENS  # Qwen3-8B 上下文长度
            ).to(DEVICE)
            
            # 生成回复
            with torch.no_grad():  # 禁用梯度，节省显存
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码回复（只取新增生成的部分）
            response = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            # 更新对话历史
            history[-1] = (user_input, response)
            
            # 输出回复
            print(f"\nAssistant：{response}\n")
        
        except Exception as e:
            print(f"[ERROR] 生成回复失败：{e}")
            # 出错时移除当前轮次的历史，避免影响后续对话
            history.pop()

if __name__ == "__main__":
    try:
        # 加载模型和分词器
        tokenizer, model = load_model_and_tokenizer()
        # 启动交互式对话
        interactive_chat(tokenizer, model)
    except Exception as e:
        print(f"[FATAL ERROR] 程序启动失败：{e}")