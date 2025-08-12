#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KSP首次适应算法模块 - K最短路径首次适应算法
该算法使用K最短路径算法寻找多条候选路径，然后在每条路径上
使用首次适应策略分配频谱槽，选择第一个找到的可用频谱块。
"""

import networkx as nx
import time
from typing import List, Tuple, Optional
from algorithm_base import AllocationAlgorithm, AlgorithmConfig, Request, register_algorithm


@register_algorithm("KSP_FF")
class KSPFirstFitAllocation(AllocationAlgorithm):
    """KSP_FF: K最短路径首次适应算法
    
    该算法使用K最短路径算法寻找多条候选路径，然后在每条路径上
    使用首次适应策略分配频谱槽，选择第一个找到的可用频谱块。
    
    算法特点：
    - 快速分配：首次适应策略保证快速找到可用频谱
    - 多路径选择：K最短路径提供多个候选路径
    - 简单高效：算法复杂度低，适合实时应用
    """

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        """
        初始化KSP_FF算法
        
        Args:
            config: 算法配置参数，如果为None则使用默认配置
        """
        if config is None:
            config = AlgorithmConfig(
                algorithm_name="KSP_FF",
                k_value=3
            )
        else:
            config.algorithm_name = "KSP_FF"
        
        super().__init__(config)
        self.k = config.k_value
        
        print(f"🔧 KSP_FF算法初始化完成，K={self.k}")

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """
        使用K最短路径和首次适应算法分配频谱

        算法流程：
        1. 构建可用链路图（只包含满足带宽和频谱槽要求的链路）
        2. 计算K条最短路径
        3. 对每条路径，使用首次适应策略寻找连续频谱槽
        4. 返回第一个成功分配的路径和频谱槽

        Args:
            request: 网络请求对象
            network: NSFNet网络实例

        Returns:
            成功时返回(路径, 频谱槽列表)，失败时返回None
        """
        start_time = time.perf_counter()
        
        try:
            # 更新统计信息
            self.allocation_stats['total_attempts'] += 1
            
            # 构建满足资源要求的可用链路图
            G = self._build_available_graph(request, network)
            
            # 为每条边添加跳数权重，优先选择跳数少的路径
            for u, v in G.edges():
                G[u][v]['weight'] = 1
            
            # 计算K条最短路径
            try:
                paths = list(nx.shortest_simple_paths(
                    G, request.source, request.destination, weight='weight'
                ))[:self.k]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # 无可用路径
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

            # 对每条路径尝试首次适应分配
            for path_index, path in enumerate(paths):
                possible_blocks = self._find_available_continuous_slots(path, request.spectrum_slots, network)
                
                if possible_blocks:
                    # 首次适应：选择第一个找到的连续频谱块
                    selected_block = possible_blocks[0]
                    
                    processing_time = time.perf_counter() - start_time
                    self._update_statistics(True, processing_time)
                    
                    # 记录选择的路径索引（用于分析）
                    self._record_path_selection(path_index, len(paths), len(possible_blocks))
                    
                    return path, selected_block

            # 所有路径都无法分配
            processing_time = time.perf_counter() - start_time
            self._update_statistics(False, processing_time)
            return None

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self._update_statistics(False, processing_time)
            print(f"❌ KSP_FF算法执行错误: {e}")
            return None
    
    def _record_path_selection(self, selected_path_index: int, total_paths: int, available_blocks: int) -> None:
        """记录路径选择信息（用于算法分析）"""
        if 'path_selection_stats' not in self.allocation_stats:
            self.allocation_stats['path_selection_stats'] = {
                'first_path_selections': 0,
                'total_path_evaluations': 0,
                'average_available_blocks': 0.0,
                'path_index_distribution': {}
            }
        
        stats = self.allocation_stats['path_selection_stats']
        stats['total_path_evaluations'] += 1
        
        # 记录选择的路径索引
        if selected_path_index == 0:
            stats['first_path_selections'] += 1
        
        # 更新路径索引分布
        if selected_path_index not in stats['path_index_distribution']:
            stats['path_index_distribution'][selected_path_index] = 0
        stats['path_index_distribution'][selected_path_index] += 1
        
        # 更新平均可用块数
        current_avg = stats['average_available_blocks']
        n = stats['total_path_evaluations']
        stats['average_available_blocks'] = (current_avg * (n - 1) + available_blocks) / n
    
    def get_detailed_statistics(self) -> dict:
        """获取详细的算法统计信息"""
        base_stats = self.get_statistics()
        
        # 添加KSP_FF特有的统计信息
        if 'path_selection_stats' in self.allocation_stats:
            path_stats = self.allocation_stats['path_selection_stats']
            base_stats['algorithm_specific'] = {
                'k_value': self.k,
                'first_path_selection_rate': (
                    path_stats['first_path_selections'] / max(1, path_stats['total_path_evaluations'])
                ),
                'average_available_blocks_per_path': path_stats['average_available_blocks'],
                'path_index_distribution': path_stats['path_index_distribution']
            }
        
        return base_stats
    
    def configure_k_value(self, k: int) -> None:
        """动态配置K值"""
        if k <= 0:
            raise ValueError("K值必须大于0")
        
        old_k = self.k
        self.k = k
        self.config.k_value = k
        
        print(f"🔧 KSP_FF算法K值已更新: {old_k} -> {k}")
    
    def get_algorithm_info(self) -> dict:
        """获取算法信息"""
        return {
            'name': self.algorithm_name,
            'type': 'K-Shortest Path with First Fit',
            'description': '使用K最短路径算法寻找候选路径，采用首次适应策略分配频谱',
            'parameters': {
                'k_value': self.k
            },
            'characteristics': [
                '快速分配',
                '多路径选择',
                '简单高效',
                '适合实时应用'
            ],
            'complexity': 'O(K * |E| * log|V| + K * S)',
            'best_use_case': '需要快速响应的网络环境'
        }


# 便捷函数
def create_ksp_ff_algorithm(k: int = 3) -> KSPFirstFitAllocation:
    """
    创建KSP_FF算法实例的便捷函数
    
    Args:
        k: K最短路径的K值
        
    Returns:
        KSPFirstFitAllocation实例
    """
    config = AlgorithmConfig(
        algorithm_name="KSP_FF",
        k_value=k
    )
    return KSPFirstFitAllocation(config)


# 使用示例和测试
if __name__ == "__main__":
    print("🔧 KSP_FF算法模块测试")
    print("=" * 50)
    
    # 创建算法实例
    algorithm = create_ksp_ff_algorithm(k=3)
    
    # 显示算法信息
    info = algorithm.get_algorithm_info()
    print(f"算法名称: {info['name']}")
    print(f"算法类型: {info['type']}")
    print(f"算法描述: {info['description']}")
    print(f"K值: {info['parameters']['k_value']}")
    print(f"复杂度: {info['complexity']}")
    print(f"最佳使用场景: {info['best_use_case']}")
    
    print(f"\n算法特点:")
    for characteristic in info['characteristics']:
        print(f"  - {characteristic}")
    
    # 测试K值配置
    print(f"\n🔧 测试K值配置...")
    print(f"当前K值: {algorithm.k}")
    algorithm.configure_k_value(5)
    print(f"更新后K值: {algorithm.k}")
    
    # 显示统计信息
    stats = algorithm.get_statistics()
    print(f"\n📊 算法统计信息:")
    print(f"总尝试次数: {stats['total_attempts']}")
    print(f"成功分配次数: {stats['successful_allocations']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    
    print("\n✅ KSP_FF算法模块测试完成！")
