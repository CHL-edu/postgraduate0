"""
FAISS RAG 长文本分段匹配检索系统

功能：将长文本分段生成 embedding，通过"分段匹配 + 结果聚合"的方式，
      找到所有和长文本相关的知识库内容。

核心特性：
1. 智能文本分段：按句子边界切分，支持重叠窗口
2. 批量 FAISS 检索：为每个片段独立检索知识库
3. 结果聚合去重：合并多片段结果，去重并重排序
4. 可配置参数：所有参数均支持灵活配置

作者：基于 faiss_test_tensordatabase.py 改进
日期：2026-01-14
"""

import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
import ollama_embedding
import config


# ===================== 配置参数 =====================
# 所有参数均为可配置常量，支持运行时调整

# 分段参数
SEGMENT_MAX_LENGTH = 500          # 单段最大字符数
SEGMENT_MIN_LENGTH = 200          # 单段最小字符数
SEGMENT_OVERLAP = 50              # 段间重叠字符数（可设为 0 禁用）
SEGMENT_THRESHOLD = 500           # 触发分段的文本长度阈值

# 检索参数
TOP_K_PER_SEGMENT = 5             # 每个片段检索的 Top-K
TOP_K_FINAL = 5                   # 最终返回的 Top-N
SIMILARITY_THRESHOLD = 0.5        # 相似度过滤阈值

# 聚合参数
AGG_METHOD = "max"                # 相似度聚合策略："max" 或 "avg"

# 知识库路径（从 config 读取）
KNOWLEDGE_BASE_PATH = config.configinit().KNOWLEDGE_BASE_PATH


# ===================== 基础函数（从 faiss_test_tensordatabase.py 复制）=====================

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


def print_results(results: List[Dict]) -> None:
    """
    打印检索结果（适配扩展字段）
    :param results: 检索结果列表
    """
    if not results:
        print(f"\n未找到相似度 >= {SIMILARITY_THRESHOLD} 的相关知识")
        return

    print(f"\n找到 {len(results)} 条相关知识（相似度阈值: {SIMILARITY_THRESHOLD}）：")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n【结果 {i}】相似度: {result['similarity']:.4f} | 距离: {result['distance']:.4f}")

        # 显示扩展字段
        if 'match_count' in result:
            print(f"匹配片段数: {result['match_count']} | 片段索引: {result.get('source_segments', [])}")

        if result.get("article_number"):
            # 显示法律名称（如果有）
            law_name = result.get('law_name', '')
            if law_name:
                print(f"法律: {law_name}")
            print(f"章节: {result['section']}+{result['article_number']}")

        print(f"法条: {result['content'][:200]}..." if len(result['content']) > 200 else f"法条: {result['content']}")
        print("-" * 80)


# ===================== 模块 1：文本分段器 =====================

class TextSplitter:
    """长文本智能分段器"""

    def __init__(self, max_length: int = SEGMENT_MAX_LENGTH,
                 min_length: int = SEGMENT_MIN_LENGTH,
                 overlap: int = SEGMENT_OVERLAP):
        """
        初始化分段器
        :param max_length: 单段最大长度（字符数）
        :param min_length: 单段最小长度（字符数）
        :param overlap: 段间重叠字符数（设为 0 可禁用重叠）
        """
        self.max_length = max_length
        self.min_length = min_length
        self.overlap = overlap

        # 定义句子结束符
        self.sentence_endings = ['。', '！', '？', '\n', '.', '!', '?']

    def split(self, text: str) -> List[str]:
        """
        将长文本切分为多个语义片段
        :param text: 输入文本
        :return: 文本片段列表
        """
        # 清理文本
        text = text.strip()
        if not text:
            return []

        # 如果文本长度小于阈值，不分段
        if len(text) <= SEGMENT_THRESHOLD:
            return [text]

        segments = []
        current_pos = 0
        text_length = len(text)

        while current_pos < text_length:
            # 计算当前段的结束位置
            end_pos = min(current_pos + self.max_length, text_length)

            # 如果剩余文本不足 min_length，直接添加剩余部分
            if text_length - current_pos <= self.min_length:
                segments.append(text[current_pos:].strip())
                break

            # 在 max_length 范围内寻找最佳断句位置
            if end_pos < text_length:
                best_break = self._find_best_break_point(text, current_pos, end_pos)
                if best_break > current_pos:
                    end_pos = best_break

            # 提取当前段
            segment = text[current_pos:end_pos].strip()
            if segment:
                segments.append(segment)

            # 移动到下一段（考虑重叠）
            current_pos = end_pos - self.overlap if end_pos < text_length else end_pos

        return segments

    def _find_best_break_point(self, text: str, start: int, end: int) -> int:
        """
        在 [start, end] 范围内寻找最佳断句位置
        :param text: 完整文本
        :param start: 起始位置
        :param end: 结束位置
        :return: 最佳断句位置
        """
        # 优先寻找句子结束符
        for i in range(end - 1, start + self.min_length - 1, -1):
            if text[i] in self.sentence_endings:
                return i + 1

        # 如果没有找到句子结束符，寻找逗号或分号
        for i in range(end - 1, start + self.min_length - 1, -1):
            if text[i] in ['，', ',', '；', ';']:
                return i + 1

        # 如果都没找到，返回 end（强制断开）
        return end


# ===================== 模块 2：批量检索器 =====================

class BatchRetriever:
    """批量 FAISS 检索器"""

    def __init__(self, index: faiss.IndexFlatL2, knowledge_base: List[Dict]):
        """
        初始化检索器
        :param index: FAISS 索引对象
        :param knowledge_base: 知识库数据
        """
        self.index = index
        self.knowledge_base = knowledge_base

    def retrieve(self, segments: List[str], top_k: int = TOP_K_PER_SEGMENT,
                 threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """
        批量检索多个文本片段
        :param segments: 文本片段列表
        :param top_k: 每个片段返回的前 K 个结果
        :param threshold: 相似度阈值
        :return: 所有片段的检索结果（未聚合）
        """
        all_results = []

        for seg_idx, segment in enumerate(segments):
            # 为当前片段生成 embedding
            try:
                segment_embedding = ollama_embedding.get_embedding(segment)
            except Exception as e:
                print(f"警告：片段 {seg_idx} 生成 embedding 失败：{e}")
                continue

            # 使用 FAISS 搜索
            query_vec = np.array([segment_embedding]).astype("float32")
            faiss.normalize_L2(query_vec)

            distances, indices = self.index.search(query_vec, top_k)

            # 处理搜索结果
            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1:  # FAISS 返回 -1 表示没有足够的结果
                    continue

                # L2 距离转换为相似度
                similarity = 1 / (1 + dist)

                # 应用相似度阈值过滤
                if similarity < threshold:
                    continue

                item = self.knowledge_base[idx]
                result = {
                    "content": item.get("content", ""),
                    "article_number": item.get("article_number", ""),
                    "section": item.get("section", ""),
                    "law_name": item.get("law_name", ""),  # 新增：法律名称
                    "similarity": round(similarity, 4),
                    "distance": round(dist, 4),
                    "segment_index": seg_idx,  # 记录来源片段索引
                }
                all_results.append(result)

        return all_results


# ===================== 模块 3：结果聚合器 =====================

class ResultAggregator:
    """检索结果聚合与去重器"""

    def __init__(self, agg_method: str = AGG_METHOD):
        """
        初始化聚合器
        :param agg_method: 相似度聚合策略（"max" 取最大值，"avg" 取平均）
        """
        if agg_method not in ["max", "avg"]:
            raise ValueError(f"不支持的聚合策略：{agg_method}，必须是 'max' 或 'avg'")
        self.agg_method = agg_method

    def aggregate(self, raw_results: List[Dict], top_n: int = TOP_K_FINAL) -> List[Dict]:
        """
        聚合多片段检索结果
        :param raw_results: 原始检索结果列表
        :param top_n: 最终返回的前 N 个结果
        :return: 去重重排序后的结果
        """
        if not raw_results:
            return []

        # 使用字典去重（以 article_number + section 为唯一键）
        aggregated = {}

        for result in raw_results:
            # 构造唯一键
            key = (result.get("article_number", ""), result.get("section", ""))

            if key not in aggregated:
                # 首次遇到此条文，直接添加
                aggregated[key] = {
                    "content": result["content"],
                    "article_number": result["article_number"],
                    "section": result["section"],
                    "law_name": result.get("law_name", ""),  # 修正字段名
                    "similarity": result["similarity"],
                    "distance": result["distance"],
                    "match_count": 1,
                    "source_segments": [result["segment_index"]],
                    "all_similarities": [result["similarity"]],  # 用于计算平均相似度
                }
            else:
                # 已存在此条文，更新信息
                existing = aggregated[key]
                existing["match_count"] += 1
                existing["source_segments"].append(result["segment_index"])
                existing["all_similarities"].append(result["similarity"])

                # 更新相似度（保留最大值）
                if result["similarity"] > existing["similarity"]:
                    existing["similarity"] = result["similarity"]
                    existing["distance"] = result["distance"]

        # 计算最终相似度并排序
        final_results = []
        for item in aggregated.values():
            # 根据聚合策略计算最终相似度
            if self.agg_method == "max":
                final_similarity = max(item["all_similarities"])
            else:  # avg
                final_similarity = sum(item["all_similarities"]) / len(item["all_similarities"])

            item["similarity"] = round(final_similarity, 4)

            # 移除临时字段
            del item["all_similarities"]

            # 去重 source_segments
            item["source_segments"] = sorted(list(set(item["source_segments"])))

            final_results.append(item)

        # 按相似度降序排序
        final_results.sort(key=lambda x: x["similarity"], reverse=True)

        # 返回 Top-N
        return final_results[:top_n]


# ===================== 模块 4：主流程控制器 =====================

class SegmentedRAG:
    """长文本分段匹配 RAG 主流程控制器"""

    def __init__(self, knowledge_base_path: str = KNOWLEDGE_BASE_PATH):
        """
        初始化 RAG 系统
        :param knowledge_base_path: 知识库 JSON 文件路径
        """
        print("=" * 80)
        print("初始化长文本分段匹配 RAG 系统")
        print("=" * 80)

        # 加载知识库
        print(f"\n正在加载知识库: {knowledge_base_path}")
        self.knowledge_base = load_knowledge_base(knowledge_base_path)
        if not self.knowledge_base:
            raise ValueError("知识库加载失败，无法初始化系统")

        # 初始化 FAISS 索引
        print("\n正在初始化 FAISS 索引...")
        self.index, self.knowledge_base = init_faiss_index(self.knowledge_base)

        # 初始化各个模块
        self.splitter = TextSplitter()
        self.retriever = BatchRetriever(self.index, self.knowledge_base)
        self.aggregator = ResultAggregator()

        print("\n系统初始化完成！")

    def search(self, query: str, top_k: int = TOP_K_FINAL,
               threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """
        长文本检索接口
        :param query: 用户查询文本（可长可短）
        :param top_k: 最终返回的前 N 个结果
        :param threshold: 相似度过滤阈值
        :return: 检索结果列表
        """
        if not query or not query.strip():
            return []

        query = query.strip()

        # 判断是否需要分段
        if len(query) <= SEGMENT_THRESHOLD:
            # 短文本：直接检索
            print(f"\n查询文本较短（{len(query)} 字符），使用直接检索")
            return self._direct_search(query, top_k, threshold)
        else:
            # 长文本：分段检索
            segments = self.splitter.split(query)
            print(f"\n查询文本较长（{len(query)} 字符），分段为 {len(segments)} 个片段")

            # 批量检索
            raw_results = self.retriever.retrieve(segments, top_k=TOP_K_PER_SEGMENT, threshold=threshold)
            print(f"批量检索完成，获得 {len(raw_results)} 条原始结果")

            # 聚合结果
            aggregated_results = self.aggregator.aggregate(raw_results, top_n=top_k)
            print(f"聚合完成，最终返回 {len(aggregated_results)} 条结果")

            return aggregated_results

    def _direct_search(self, query: str, top_k: int, threshold: float) -> List[Dict]:
        """
        短文本直接检索（不分段）
        :param query: 查询文本
        :param top_k: 返回前 K 个结果
        :param threshold: 相似度阈值
        :return: 检索结果列表
        """
        # 生成 embedding
        query_embedding = ollama_embedding.get_embedding(query)

        # FAISS 搜索
        query_vec = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, top_k)

        # 处理结果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            similarity = 1 / (1 + dist)
            if similarity < threshold:
                continue

            item = self.knowledge_base[idx]
            results.append({
                "content": item.get("content", ""),
                "article_number": item.get("article_number", ""),
                "section": item.get("section", ""),
                "law_name": item.get("law_name", ""),  # 新增：法律名称
                "similarity": round(similarity, 4),
                "distance": round(dist, 4),
                "match_count": 1,
                "source_segments": [0],
            })

        return results


# ===================== 主程序 =====================

def main():
    """主程序入口（交互式问答界面）"""
    print("=" * 80)
    print("FAISS RAG 长文本分段匹配检索系统")
    print("=" * 80)

    # 初始化 RAG 系统
    try:
        rag = SegmentedRAG()
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return

    # 交互式问答循环
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

            # 检索
            results = rag.search(query)

            # 打印结果
            print_results(results)

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n处理问题时发生错误: {e}")
            continue


if __name__ == "__main__":
    main()
