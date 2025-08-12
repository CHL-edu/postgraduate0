import shutil
from pathlib import Path
import os

def delete_folder_content(folder_path: str | Path, dry_run: bool = False) -> None:
    folder_path = Path(folder_path)

    # 检查路径是否存在且为文件夹
    if not folder_path.exists():
        raise FileNotFoundError(f"指定的路径不存在: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"指定的路径不是文件夹: {folder_path}")

    # 遍历文件夹内的所有内容
    for item in folder_path.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                if dry_run:
                    print(f"[模拟删除] 文件: {item}")
                else:
                    item.unlink()
                    print(f"已删除文件: {item}")
            elif item.is_dir():
                if dry_run:
                    print(f"[模拟删除] 文件夹: {item}")
                else:
                    shutil.rmtree(item)
                    print(f"已删除文件夹: {item}")
        except Exception as e:
            print(f"删除 {item} 时出错: {e}")


def delete_files_by_extension(folder_path, extensions):
    """
    删除指定文件夹中特定类型的文件（非递归，不处理子文件夹）

    参数:
        folder_path (str): 要处理的文件夹路径
        extensions (list): 要删除的文件扩展名列表，例如 ['.txt', '.log']
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个有效的文件夹")
        return

    # 确保扩展名以点开头
    normalized_extensions = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = f'.{ext}'
        normalized_extensions.append(ext.lower())

    # 遍历文件夹中的所有项目
    deleted_count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # 只处理文件，不处理子文件夹
        if os.path.isfile(item_path):
            # 获取文件扩展名并转换为小写
            file_ext = os.path.splitext(item)[1].lower()

            # 检查是否是要删除的文件类型
            if file_ext in normalized_extensions:
                try:
                    os.remove(item_path)
                    print(f"已删除: {item_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除文件 '{item_path}' 失败: {str(e)}")

    print(f"操作完成，共删除 {deleted_count} 个文件")


if __name__ == "__main__":
    # 设置要清理的文件夹路径
    # 获取用户输入并存储到变量中
    TARGET_Index = input("请输入delet,0为整个文件夹,其他png")
    print(f"你输入的是：{TARGET_Index}")

    TARGET_FOLDER = r"E:\PythonProject\KSP\log" # 请替换为实际要清理的文件夹路径
    print("\n=== 实际删除开始 ===")
    if(TARGET_Index == 0):
        try:
            delete_folder_content(TARGET_FOLDER, dry_run=False)
        except Exception as e:
            print(f"空文件夹")
    else:
        try:
            delete_files_by_extension(TARGET_FOLDER, ['.png'])
        except Exception as e:
            print(f"空图片")
    print("\n删除操作完成！")
