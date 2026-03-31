import json
from typing import List, Dict
import os


def load_json_file(file_path: str) -> List[Dict]:
    """
    加载JSON文件
    :param file_path: JSON文件路径
    :return: 数据列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON文件必须包含数组（列表）格式")

        print(f"成功加载: {file_path} ({len(data)} 条记录)")
        return data

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"错误：{file_path} 不是合法的JSON文件 - {e}")
        return []
    except Exception as e:
        print(f"加载 {file_path} 时发生错误: {e}")
        return []


def reassign_ids(data: List[Dict]) -> List[Dict]:
    """
    重新分配ID，确保ID唯一且连续（从1开始）
    :param data: 原始数据列表
    :return: 重新分配ID后的数据列表
    """
    if not data:
        return data

    for idx, item in enumerate(data, start=1):
        item["id"] = str(idx)

    print(f"已重新分配ID: 1 到 {len(data)}")
    return data


def combine_json_files(
    input_files: List[str],
    output_file: str,
    reassign_id: bool = True
) -> bool:
    """
    合并多个JSON文件
    :param input_files: 输入文件路径列表
    :param output_file: 输出文件路径
    :param reassign_id: 是否重新分配ID（默认True）
    :return: 是否成功
    """
    print("=" * 80)
    print("JSON文件合并工具")
    print("=" * 80)

    # 1. 加载所有文件
    combined_data = []
    total_count = 0

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"警告：文件不存在，跳过 - {file_path}")
            continue

        data = load_json_file(file_path)
        if data:
            combined_data.extend(data)
            total_count += len(data)

    if not combined_data:
        print("\n错误：没有成功加载任何数据")
        return False

    print(f"\n合并完成: 共 {total_count} 条记录")

    # 2. 重新分配ID
    if reassign_id:
        print(f"\n正在重新分配ID...")
        combined_data = reassign_ids(combined_data)

    # 3. 保存合并后的文件
    print(f"\n正在保存到: {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

        print(f"\n成功保存 {len(combined_data)} 条记录到: {output_file}")
        return True

    except Exception as e:
        print(f"保存文件失败: {e}")
        return False


def print_summary(combined_file: str) -> None:
    """
    打印合并后的文件摘要信息
    :param combined_file: 合并后的文件路径
    """
    try:
        with open(combined_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("\n" + "=" * 80)
        print("合并文件摘要")
        print("=" * 80)
        print(f"总记录数: {len(data)}")
        print(f"ID范围: {data[0].get('id', 'N/A')} ~ {data[-1].get('id', 'N/A')}")

        # 统计各法律类型数量
        law_count = {}
        for item in data:
            law_name = item.get("law_name", "未知")
            law_count[law_name] = law_count.get(law_name, 0) + 1

        print("\n各法律类型统计:")
        for law_name, count in law_count.items():
            print(f"  - {law_name}: {count} 条")

        # 显示第一条和最后一条记录
        print(f"\n第一条记录:")
        print(f"  ID: {data[0].get('id')}")
        print(f"  法律: {data[0].get('law_name')}")
        print(f"  章节: {data[0].get('chapter')}")
        print(f"  条号: {data[0].get('article_number')}")
        print(f"  内容: {data[0].get('content', '')[:100]}...")

        print(f"\n最后一条记录:")
        print(f"  ID: {data[-1].get('id')}")
        print(f"  法律: {data[-1].get('law_name')}")
        print(f"  章节: {data[-1].get('chapter')}")
        print(f"  条号: {data[-1].get('article_number')}")
        print(f"  内容: {data[-1].get('content', '')[:100]}...")

        print("=" * 80)

    except Exception as e:
        print(f"生成摘要失败: {e}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    import config

    # 配置输入输出文件
    INPUT_FILES = [
        config.LABOUR_LAW_OUTPUT,
        config.CIVIL_LAW_OUTPUT
    ]
    OUTPUT_FILE = config.KNOWLEDGE_BASE_PATH

    # 执行合并
    success = combine_json_files(
        input_files=INPUT_FILES,
        output_file=OUTPUT_FILE,
        reassign_id=True  # 重新分配ID确保唯一性
    )

    if success:
        # 打印合并摘要
        print_summary(OUTPUT_FILE)
    else:
        print("\n合并失败！")
        exit(1)
