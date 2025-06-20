import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging
import os
from datetime import datetime
import time

log_path = "/data/chl/pyproject/backup/testLLM_613.txt"
remote_model_path = "/data/chl/download/DeepSeek-R1-0528-Qwen3-8B"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置 transformers 日志级别以减少冗余输出
transformers_logging.set_verbosity_error()


def select_system_prompt():
    """让用户输入初始化对话历史（system prompt），默认为 '你是一名通信专家。'"""
    try:
        prompt = input("Custom Instructions（默认为 '你是一名通信专家。'）：").strip()
        return prompt if prompt else "你是一名通信专家。"
    except Exception as e:
        logger.error(f"System Warning:Input failure，default: {e}")
        return "你是一名通信专家。"


def select_gpus():
    """让用户选择 GPU，默认为 [2, 3]"""
    try:
        print("Available GPU:", torch.cuda.device_count())
        gpu_input = input("请输入要使用的 GPU 编号（逗号分隔，默认为 2,3;'all'为所有可用GPU）：").strip()

        if gpu_input == "all":
            gpu_available = torch.cuda.device_count()
            gpu_selected = list(range(gpu_available))
            return gpu_selected

        if not gpu_input:
            return [2, 3]

        gpus = [int(gpu.strip()) for gpu in gpu_input.split(",")]
        if not all(0 <= gpu < torch.cuda.device_count() for gpu in gpus):
            raise ValueError("GPU 编号超出范围")
        return gpus
    except Exception as e:
        print(f"发生错误: {e}")
        return [2, 3]  # 默认返回 [2, 3] 作为容错处理


def select_max_tokens():
    """让用户选择最大生成 token 数，默认为 4096"""
    try:
        max_tokens = input("请输入最大生成 token 数（默认为 4096）：").strip()
        return int(max_tokens) if max_tokens else 4096
    except ValueError as e:
        logger.error(f"无效 token 数，使用默认 4096: {e}")
        return 4096


def main():
    logger.info("启动")

    # 获取用户输入的系统提示
    system_prompt = select_system_prompt()
    logger.info(f"使用系统提示: {system_prompt}")

    # 获取用户输入
    gpus = select_gpus()
    max_tokens = select_max_tokens()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    logger.info(f"使用 GPU: {gpus}")
    logger.info(f"最大生成 token 数: {max_tokens}")

    # 加载模型和分词器
    model_path = remote_model_path
    logger.info(f"加载模型: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    # 初始化对话历史
    conversation = [
        {"role": "system", "content": system_prompt}
    ]

    print("\n对话,输入 'quit' END")

    while True:
        # 获取用户输入
        user_input = input("\n您: ").strip()
        if user_input.lower() == "quit":
            logger.info("killing process...")
            break

        # 添加用户消息
        conversation.append({"role": "user", "content": user_input})
        logger.info(f"用户输入: {user_input}")

        # 构造提示
        try:
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.error(f"构造提示失败: {e}")
            print("错误：无法处理输入，请重试。")
            conversation.pop()  # 移除无效输入
            continue

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 生成参数（可编辑参数及注释）
        generate_params = {
            "max_new_tokens": max_tokens,  # 最大生成 token 数，用户指定，控制输出长度
            "temperature": 0.7,  # 控制生成随机性，0.0-2.0，值越大越随机
            "do_sample": True,  # 是否启用采样，True 时启用 temperature 和 top_p
            "top_p": 1.0,  # 核采样概率，0.0-1.0，保留累积概率最高的 token
            "top_k": 50,  # 限制采样的 token 数量，1-∞，值越小越聚焦
            "repetition_penalty": 1.0,  # 重复惩罚，1.0 表示无惩罚，>1.0 减少重复
            "no_repeat_ngram_size": 0,  # 禁止重复的 n-gram 大小，0 表示禁用
            "length_penalty": 1.0,  # 长度惩罚，>1.0 鼓励长输出，<1.0 鼓励短输出
            "num_beams": 1,  # 束搜索数量，1 表示禁用束搜索，>1 启用
            "early_stopping": False  # 是否启用早期停止（束搜索时有效）
        }
        logger.info(f"Generate Params: {generate_params}")

        # 推理并计算时间
        try:
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, **generate_params)
            end_time = time.time()
            inference_time = end_time - start_time

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()

            # 添加模型回答到对话历史
            conversation.append({"role": "assistant", "content": generated_text})
            logger.info(f"AI回答: {generated_text}")
            logger.info(f"推理时间: {inference_time:.2f} 秒")
            logger.info("*"*50+"\n")
            # 打印回答和推理时间
            print(f"\nAI: {generated_text}")
            print(f"Loading: {inference_time:.2f} 秒")

        except Exception as e:
            logger.error(f"推理失败: {e}")
            print("错误：生成回答失败，请重试。")
            conversation.pop()  # 移除用户输入
            conversation.pop()  # 移除系统提示（若有）


if __name__ == "__main__":
    main()