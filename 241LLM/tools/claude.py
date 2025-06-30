import re


def keep_only_chinese(input_file, output_file):
    """
    从文本文件中只保留汉字，并输出到新文件

    参数:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    # 定义匹配汉字的正则表达式（包括基本汉字和扩展汉字）
    chinese_pattern = re.compile(
        r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]+')

    with open(input_file, 'r', encoding='utf-8') as f_in:
        text = f_in.read()

    # 提取所有汉字
    chinese_text = ''.join(chinese_pattern.findall(text))

    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(chinese_text)


# 示例用法
input_file = 'principle.txt'  # 替换为你的输入文件路径
output_file = 'output.txt'  # 替换为你的输出文件路径
keep_only_chinese(input_file, output_file)
print(f"处理完成，结果已保存到 {output_file}")