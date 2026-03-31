import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
import ollama_embedding
import config


# ===================== 配置参数 =====================
KNOWLEDGE_BASE_PATH =  config.configinit().KNOWLEDGE_BASE_PATH # 知识库路径
SIMILARITY_THRESHOLD = 0.5  # 相似度阈值（0-1，越高越严格）
TOP_K = 5  # 返回最相似的前k个结果


def load_knowledge_base(file_path: str) -> List[Dict]:
    """
    加载知识库数据
    :param file_path: 知识库JSON文件路径
    :return: 知识库列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)

        # 验证数据格式
        if not isinstance(knowledge_base, list):
            raise ValueError("知识库必须是JSON数组格式")

        # 过滤掉没有embedding的数据
        valid_data = [item for item in knowledge_base if "embedding" in item]

        if len(valid_data) < len(knowledge_base):
            print(f"警告：过滤掉 {len(knowledge_base) - len(valid_data)} 条无embedding的数据")

        print(f"成功加载 {len(valid_data)} 条知识库数据")
        return valid_data

    except FileNotFoundError:
        print(f"错误：未找到知识库文件 {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"错误：{file_path} 不是合法的JSON文件")
        return []


def init_faiss_index(knowledge_base: List[Dict]) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """
    初始化FAISS索引
    :param knowledge_base: 知识库数据（包含embedding字段）
    :return: FAISS索引对象和知识库
    """
    if not knowledge_base:
        raise ValueError("知识库为空，无法初始化索引")

    # 提取所有embedding向量
    embeddings = np.array([item["embedding"] for item in knowledge_base]).astype("float32")

    print(f"Embedding向量维度: {embeddings.shape[1]}")
    print(f"索引向量数量: {embeddings.shape[0]}")

    # 创建FAISS索引（使用L2距离，需先归一化以等价于余弦相似度）
    index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss.normalize_L2(embeddings)  # 归一化向量
    index.add(embeddings)

    print(f"FAISS索引初始化完成，索引包含 {index.ntotal} 个向量")
    return index, knowledge_base


def search_similar_knowledge(
    query: str,
    index: faiss.IndexFlatL2,
    knowledge_base: List[Dict],
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD
) -> List[Dict]:
    """
    使用FAISS检索相似知识
    :param query: 用户问题
    :param index: FAISS索引对象
    :param knowledge_base: 知识库数据
    :param top_k: 返回前k个最相似结果
    :param threshold: 相似度阈值（低于此值的结果会被过滤）
    :return: 匹配结果列表
    """
    # 1. 生成问题的embedding
    print(f"\n正在为问题生成embedding...")
    query_embedding = ollama_embedding.get_embedding(query)

    # 2. 用FAISS搜索最相似的向量
    query_vec = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k)

    # 3. 转换距离为相似度并过滤
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:  # FAISS返回-1表示没有足够的结果
            continue

        # L2距离转换为相似度（距离越小，相似度越高）
        similarity = 1 / (1 + dist)

        # 应用相似度阈值过滤
        if similarity < threshold:
            continue

        item = knowledge_base[idx]
        results.append({
            "content": item.get("content", ""),
            "article_number": item.get("article_number", ""),
            "similarity": round(similarity, 4),
            "distance": round(dist, 4),
            "section": item.get("section", ""),
            "article_number": item.get("article_number", "")
        })

    return results


def print_results(results: List[Dict]) -> None:
    """
    打印检索结果
    :param results: 检索结果列表
    """
    if not results:
        print(f"\n未找到相似度 >= {SIMILARITY_THRESHOLD} 的相关知识")
        return

    print(f"\n找到 {len(results)} 条相关知识（相似度阈值: {SIMILARITY_THRESHOLD}）：")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n【结果 {i}】相似度: {result['similarity']:.4f} | 距离: {result['distance']:.4f}")
        if result.get("article_number"):
            print(f"章节: {result['section']}+{result['article_number']}" )    
        print(f"法条: {result['content'][:200]}..." if len(result['content']) > 200 else f"法条: {result['content']}")
        print("-" * 80)


# ===================== 主程序 =====================
def main():
    """主程序入口"""
    print("=" * 80)
    print("法律知识库RAG检索系统")
    print("=" * 80)

    # 1. 加载知识库
    print(f"\n正在加载知识库: {KNOWLEDGE_BASE_PATH}")
    knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_PATH)

    if not knowledge_base:
        print("知识库加载失败，程序退出")
        return

    # 2. 初始化FAISS索引
    print("\n正在初始化FAISS索引...")
    try:
        index, knowledge_base = init_faiss_index(knowledge_base)
    except Exception as e:
        print(f"索引初始化失败: {e}")
        return

    # 3. 交互式问答循环
    print("\n" + "=" * 80)
    print("系统已就绪！输入您的问题（输入 'quit' 或 'exit' 退出）")
    print("=" * 80)

    while True:
        try:
            # 获取用户输入
            query = input("\n请输入问题: ").strip()

            if not query:
                print("问题不能为空，请重新输入")
                continue

            if query.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break

            # 检索相似知识
            results = search_similar_knowledge(query, index, knowledge_base)

            # 打印结果，相似度
            print_results(results)

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n处理问题时发生错误: {e}")
            continue


if __name__ == "__main__":
    main()
