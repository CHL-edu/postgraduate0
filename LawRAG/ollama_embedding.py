import json
import requests
import time
from typing import List, Dict
import os
import config
# ===================== 核心配置 =====================
OLLAMA_API_URL = config.configinit().OLLAMA_API_URL  # Ollama默认API地址
EMBEDDING_MODEL = config.configinit().EMBEDDING_MODEL                # 你本地的embedding模型
INPUT_JSON_PATH = config.LABOUR_LAW_INPUT         # 原始法律数据文件（含content字段）
OUTPUT_JSON_PATH = config.LABOUR_LAW_OUTPUT        # 输出带embedding的文件

# ===================== 调用Ollama生成Embedding =====================
def get_embedding(text: str) -> List[float]:
    """
    调用Ollama的qwen3-embedding:生成文本的embedding向量
    :param text: 要生成embedding的文本（如法律条文content）
    :return: 一维浮点数向量列表
    """
    # 过滤空文本，避免模型报错
    if not text or text.strip() == "":
        raise ValueError("输入文本不能为空")
    
    try:
        # 构造请求体
        payload = {
            "model": EMBEDDING_MODEL,
            "prompt": text.strip(),
            "options": {
                "temperature": 0.0  # embedding生成固定温度，保证结果稳定
            }
        }
        
        # 发送请求到Ollama API
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=60  # 超时时间（8B模型处理稍慢，设长一点）
        )
        response.raise_for_status()  # 抛出HTTP错误
        
        # 解析响应，提取embedding向量
        result = response.json()
        embedding = result.get("embedding", [])
        
        if not embedding:
            raise ValueError("Ollama返回空的embedding向量")
        
        return embedding
    
    except requests.exceptions.RequestException as e:
        print(f"调用Ollama API失败：{e}")
        raise
    except Exception as e:
        print(f"生成embedding失败：{e}")
        raise

# ===================== 处理数据并填充Embedding =====================
def process_law_data(input_path: str, output_path: str) -> None:
    """
    读取原始法律数据，为每个条目生成embedding并保存
    :param input_path: 原始数据文件路径
    :param output_path: 输出带embedding的文件路径
    """
    # 1. 读取原始数据
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        # 确保原始数据是列表格式
        if not isinstance(raw_data, list):
            raise ValueError("原始数据必须是JSON数组（列表）格式")
        print(f"成功读取 {len(raw_data)} 条法律数据")
    
    except FileNotFoundError:
        print(f"错误：未找到文件 {input_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：{input_path} 不是合法的JSON文件")
        return
    
    # 2. 逐个生成embedding，每50个写入一次文件
    processed_data = []
    write_count = 0
    batch_size = 50

    for idx, item in enumerate(raw_data):
        try:
            # 检查是否有content字段
            if "content" not in item:
                print(f"跳过第{idx+1}条：无content字段")
                continue

            # 生成embedding（等待完成后才继续）
            print(f"正在处理第{idx+1}/{len(raw_data)}条数据...")
            embedding = get_embedding(item["content"])

            # 填充embedding字段（覆盖/新增）
            item["embedding"] = embedding
            processed_data.append(item)

            # 每50个写一次文件
            if len(processed_data) >= batch_size:
                _append_to_file(output_path, processed_data)
                write_count += len(processed_data)
                print(f"已保存 {write_count} 条数据到文件")
                processed_data.clear()

        except Exception as e:
            print(f"处理第{idx+1}条失败：{e}")
            continue

    # 3. 保存剩余数据到JSON文件
    if processed_data:
        _append_to_file(output_path, processed_data)
        write_count += len(processed_data)

    print(f"\n处理完成！成功生成 {write_count} 条带embedding的数据")
    print(f"结果已保存到：{output_path}")


def _append_to_file(file_path: str, data: List[Dict]) -> None:
    """
    将数据追加写入JSON文件（如果文件不存在则创建，存在则读取并追加）
    :param file_path: 文件路径
    :param data: 要追加的数据列表
    """
    # 如果文件不存在，直接写入
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return

    # 文件存在，读取现有数据并追加
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        existing_data = []

    # 合并数据并写回
    existing_data.extend(data)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

# ===================== 测试与运行 =====================
if __name__ == "__main__":
    try:
        process_law_data(INPUT_JSON_PATH, OUTPUT_JSON_PATH)
    except Exception as e:
        print(f"测试失败：{e}")
        exit(1)
    
