#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机适应算法模块 - Random Fit算法
该算法随机选择可用的路径和频谱块进行分配，
提供基准性能对比和负载分散效果。
"""

import networkx as nx
import numpy as np
import time
from typing import List, Tuple, Optional
from algorithm_base import AllocationAlgorithm, AlgorithmConfig, Request, register_algorithm


@register_algorithm("Random_Fit")
class RandomFitAllocation(AllocationAlgorithm):
    """Random_Fit: 随机适应算法
    
    该算法随机选择可用的路径和频谱块进行分配。
    虽然不是最优策略，但可以提供基准性能对比，
    并且在某些情况下能够实现较好的负载分散效果。
    
    算法特点：
    - 随机选择：避免算法偏向性
    - 负载分散：随机性有助于负载均衡
    - 简单快速：算法复杂度低
    - 基准对比：提供性能基准
    """

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        """
        初始化Random_Fit算法
        
        Args:
            config: 算法配置参数，如果为None则使用默认配置
        """
        if config is None:
            config = AlgorithmConfig(
                algorithm_name="Random_Fit",
                max_paths=5,
                k_value=3
            )
        else:
            config.algorithm_name = "Random_Fit"
        
        super().__init__(config)
        self.max_paths = config.max_paths
        self.k = config.k_value
        
        # 设置随机种子（可选）
        self.random_seed = getattr(config, 'random_seed', None)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        print(f"🔧 Random_Fit算法初始化完成，最大路径数={self.max_paths}")

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """
        使用随机选择算法分配频谱

        算法流程：
        1. 构建可用链路图
        2. 计算多条可用路径（最多max_paths条）
        3. 随机选择一条路径
        4. 在该路径上随机选择一个可用的频谱块

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
            
            # 为每条边添加随机权重，增加路径多样性
            for u, v in G.edges():
                G[u][v]['weight'] = np.random.uniform(0.1, 1.0)
            
            # 计算多条路径
            try:
                paths = list(nx.shortest_simple_paths(
                    G, request.source, request.destination, weight='weight'
                ))[:self.max_paths]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

            if not paths:
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

            # 收集所有可用的(路径, 频谱块)组合
            available_allocations = []
            
            for path_index, path in enumerate(paths):
                possible_blocks = self._find_available_continuous_slots(path, request.spectrum_slots, network)
                
                for block in possible_blocks:
                    available_allocations.append((path_index, path, block))

            if not available_allocations:
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

            # 随机选择一个可用的分配
            selected_index = np.random.randint(0, len(available_allocations))
            path_index, selected_path, selected_block = available_allocations[selected_index]
            
            processing_time = time.perf_counter() - start_time
            self._update_statistics(True, processing_time)
            
            # 记录随机选择信息
            self._record_random_selection(
                path_index, len(paths), len(available_allocations), selected_index
            )
            
            return selected_path, selected_block

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self._update_statistics(False, processing_time)
            print(f"❌ Random_Fit算法执行错误: {e}")
            return None
    
    def _record_random_selection(self, path_index: int, total_paths: int, 
                                total_allocations: int, selected_index: int) -> None:
        """记录随机选择信息"""
        if 'random_selection_stats' not in self.allocation_stats:
            self.allocation_stats['random_selection_stats'] = {
                'total_selections': 0,
                'average_available_allocations': 0.0,
                'path_selection_distribution': {},
                'allocation_index_distribution': {}
            }
        
        stats = self.allocation_stats['random_selection_stats']
        stats['total_selections'] += 1
        
        # 更新平均可用分配数
        current_avg = stats['average_available_allocations']
        n = stats['total_selections']
        stats['average_available_allocations'] = (current_avg * (n - 1) + total_allocations) / n
        
        # 更新路径选择分布
        if path_index not in stats['path_selection_distribution']:
            stats['path_selection_distribution'][path_index] = 0
        stats['path_selection_distribution'][path_index] += 1
        
        # 更新分配索引分布（用于分析随机性）
        allocation_bucket = min(selected_index // 10, 9)  # 分成10个桶
        if allocation_bucket not in stats['allocation_index_distribution']:
            stats['allocation_index_distribution'][allocation_bucket] = 0
        stats['allocation_index_distribution'][allocation_bucket] += 1
    
    def get_detailed_statistics(self) -> dict:
        """获取详细的算法统计信息"""
        base_stats = self.get_statistics()
        
        # 添加Random_Fit特有的统计信息
        if 'random_selection_stats' in self.allocation_stats:
            random_stats = self.allocation_stats['random_selection_stats']
            base_stats['algorithm_specific'] = {
                'max_paths': self.max_paths,
                'k_value': self.k,
                'random_seed': self.random_seed,
                'average_available_allocations': random_stats['average_available_allocations'],
                'path_selection_distribution': random_stats['path_selection_distribution'],
                'allocation_index_distribution': random_stats['allocation_index_distribution'],
                'randomness_quality': self._calculate_randomness_quality(random_stats)
            }
        
        return base_stats
    
    def _calculate_randomness_quality(self, random_stats: dict) -> dict:
        """计算随机性质量指标"""
        path_dist = random_stats['path_selection_distribution']
        alloc_dist = random_stats['allocation_index_distribution']
        
        # 计算路径选择的均匀性
        if path_dist:
            path_values = list(path_dist.values())
            path_mean = sum(path_values) / len(path_values)
            path_variance = sum((v - path_mean) ** 2 for v in path_values) / len(path_values)
            path_uniformity = 1.0 / (1.0 + path_variance / max(1, path_mean))
        else:
            path_uniformity = 0.0
        
        # 计算分配索引的均匀性
        if alloc_dist:
            alloc_values = list(alloc_dist.values())
            alloc_mean = sum(alloc_values) / len(alloc_values)
            alloc_variance = sum((v - alloc_mean) ** 2 for v in alloc_values) / len(alloc_values)
            alloc_uniformity = 1.0 / (1.0 + alloc_variance / max(1, alloc_mean))
        else:
            alloc_uniformity = 0.0
        
        return {
            'path_selection_uniformity': path_uniformity,
            'allocation_index_uniformity': alloc_uniformity,
            'overall_randomness_score': (path_uniformity + alloc_uniformity) / 2
        }
    
    def configure_max_paths(self, max_paths: int) -> None:
        """配置最大路径数"""
        if max_paths <= 0:
            raise ValueError("最大路径数必须大于0")
        
        old_max_paths = self.max_paths
        self.max_paths = max_paths
        self.config.max_paths = max_paths
        
        print(f"🔧 Random_Fit算法最大路径数已更新: {old_max_paths} -> {max_paths}")
    
    def set_random_seed(self, seed: int) -> None:
        """设置随机种子"""
        self.random_seed = seed
        np.random.seed(seed)
        print(f"🎲 Random_Fit算法随机种子已设置: {seed}")
    
    def get_algorithm_info(self) -> dict:
        """获取算法信息"""
        return {
            'name': self.algorithm_name,
            'type': 'Random Selection Algorithm',
            'description': '随机选择可用的路径和频谱块进行分配',
            'parameters': {
                'max_paths': self.max_paths,
                'k_value': self.k,
                'random_seed': self.random_seed
            },
            'characteristics': [
                '随机选择',
                '负载分散',
                '简单快速',
                '基准对比',
                '避免算法偏向性'
            ],
            'complexity': 'O(K * |E| * log|V| + R)',
            'best_use_case': '基准性能测试和负载均衡场景'
        }


# 便捷函数
def create_random_fit_algorithm(max_paths: int = 5, random_seed: Optional[int] = None) -> RandomFitAllocation:
    """
    创建Random_Fit算法实例的便捷函数
    
    Args:
        max_paths: 最大路径数
        random_seed: 随机种子
        
    Returns:
        RandomFitAllocation实例
    """
    config = AlgorithmConfig(
        algorithm_name="Random_Fit",
        max_paths=max_paths,
        k_value=max_paths
    )
    
    if random_seed is not None:
        config.random_seed = random_seed
    
    return RandomFitAllocation(config)


# 使用示例和测试
if __name__ == "__main__":
    print("🔧 Random_Fit算法模块测试")
    print("=" * 50)
    
    # 创建算法实例
    algorithm = create_random_fit_algorithm(max_paths=5, random_seed=42)
    
    # 显示算法信息
    info = algorithm.get_algorithm_info()
    print(f"算法名称: {info['name']}")
    print(f"算法类型: {info['type']}")
    print(f"算法描述: {info['description']}")
    print(f"参数: {info['parameters']}")
    print(f"复杂度: {info['complexity']}")
    print(f"最佳使用场景: {info['best_use_case']}")
    
    print(f"\n算法特点:")
    for characteristic in info['characteristics']:
        print(f"  - {characteristic}")
    
    # 测试配置
    print(f"\n🔧 测试算法配置...")
    print(f"当前最大路径数: {algorithm.max_paths}")
    algorithm.configure_max_paths(8)
    print(f"更新后最大路径数: {algorithm.max_paths}")
    
    # 测试随机种子
    print(f"\n🎲 测试随机种子设置...")
    algorithm.set_random_seed(123)
    
    # 显示统计信息
    stats = algorithm.get_statistics()
    print(f"\n📊 算法统计信息:")
    print(f"总尝试次数: {stats['total_attempts']}")
    print(f"成功分配次数: {stats['successful_allocations']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    
    print("\n✅ Random_Fit算法模块测试完成！")
