import json

def add_tag_to_json(json_data, new_tag, tag_value):
    """
    为JSON数据的每个元素新增标签
    
    Args:
        json_data: 解析后的JSON数据（字典或列表）
        new_tag: 要新增的标签名
        tag_value: 新增标签对应的值
    
    Returns:
        修改后的JSON数据
    """
    # 处理列表类型（JSON数组）
    if isinstance(json_data, list):
        for item in json_data:
            # 确保列表中的元素是字典类型
            if isinstance(item, dict):
                item[new_tag] = tag_value
    # 处理字典类型（JSON对象）
    elif isinstance(json_data, dict):
        json_data[new_tag] = tag_value
    
    return json_data

# ==================== 示例1：读取JSON文件并修改 ====================
import config

input_file = config.LABOUR_LAW_INPUT
output_file = config.LAWDATA_PATH + "/labour_law_embedding.json"
# 1. 读取JSON文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 为每个元素新增标签（例如新增"status": "processed"标签）
modified_data = add_tag_to_json(data, "embedding","")

# 3. 将修改后的元素一个一个的写入civil_law_embedding.json文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('[\n')
f.close()

with open(output_file, 'a', encoding='utf-8') as f:
    for item in modified_data:
        json.dump(item, f, ensure_ascii=False)
        if item != modified_data[-1]:
            f.write(',\n')
        else:  
            f.write('')
f.close()

with open(output_file, 'a', encoding='utf-8') as f:
    f.write('\n]')
f.close()

