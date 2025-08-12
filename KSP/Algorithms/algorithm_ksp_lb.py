#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KSP负载均衡算法模块 - K最短路径负载均衡算法
该算法在K最短路径的基础上，优先选择负载较轻的路径，
实现网络负载的均衡分布。
"""

import networkx as nx
import time
from typing import List, Tuple, Optional
from algorithm_base import AllocationAlgorithm, AlgorithmConfig, Request, register_algorithm


@register_algorithm("KSP_LB")
class KSPLoadBalanceAllocation(AllocationAlgorithm):
    """KSP_LB: K最短路径负载均衡算法（严格按照参考规范实现）

    该算法严格按照参考规范实现，使用平均链路利用率作为主要选择标准，
    跳数作为次要选择标准，实现网络负载均衡。

    参考规范步骤：
    1. 使用KSP算法计算从s到d的前k条最短路径
    2. 对每条路径计算平均链路利用率 Li = total_utilization / 链路数量
    3. 选择具有最小Li的路径
    4. 若多条路径具有相同的最小Li，则选择跳数最少的路径
    5. 使用首次适应策略分配频谱资源

    算法特点：
    - 严格规范遵循：完全按照参考算法规范实现
    - 简单高效：使用简单的平均链路利用率计算
    - 跳数优化：相同利用率时优先选择跳数少的路径
    - 首次适应：使用首次适应策略分配频谱
    """

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        """
        初始化KSP_LB算法

        Args:
            config: 算法配置参数，如果为None则使用默认配置
        """
        if config is None:
            config = AlgorithmConfig(
                algorithm_name="KSP_LB",
                k_value=3,
                edge_weight_offset=0.01
            )
        else:
            config.algorithm_name = "KSP_LB"

        super().__init__(config)
        self.k = config.k_value

        print(f"🔧 KSP_LB算法初始化完成，K={self.k}")
        print(f"   按照参考规范：使用平均链路利用率和跳数选择路径")

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """
        使用K最短路径负载均衡算法分配频谱（严格按照参考规范）

        算法流程（按照参考规范步骤1-14）：
        1. 构建可用链路图，使用负载利用率作为边权重
        2. 计算K条最短路径
        3. 对每条路径计算平均链路利用率 Li = total_utilization / 链路数量
        4. 选择具有最小Li的路径，相同Li时选择跳数最少的路径
        5. 使用首次适应策略分配频谱资源

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
            
            # 构建负载感知的可用链路图
            G = self._build_load_aware_graph(request, network)
            
            # 计算K条最短路径
            try:
                paths = list(nx.shortest_simple_paths(
                    G, request.source, request.destination, weight='weight'
                ))[:self.k]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

            # 评估每条路径的负载均衡情况
            best_allocation = None
            best_Li = float('inf')
            best_hop_count = float('inf')

            for path_index, path in enumerate(paths):
                possible_blocks = self._find_available_continuous_slots(path, request.spectrum_slots, network)

                if not possible_blocks:
                    continue

                # 计算路径的平均链路利用率 Li
                Li = self._calculate_load_balance_score(path, network)
                hop_count = len(path) - 1  # 跳数 = 节点数 - 1

                # 按照参考规范选择路径：
                # 1. 优先选择具有最小 Li 的路径
                # 2. 若多条路径具有相同的最小 Li，则选择跳数最少的路径
                should_select = False
                if Li < best_Li:
                    should_select = True
                elif Li == best_Li and hop_count < best_hop_count:
                    should_select = True

                if should_select:
                    best_Li = Li
                    best_hop_count = hop_count
                    # 在最优路径上选择首个可用频谱块（首次适应策略）
                    best_allocation = (path, possible_blocks[0])

                    # 记录选择信息
                    self._record_load_balance_selection(
                        path_index, len(paths), len(possible_blocks), Li
                    )

            if best_allocation:
                processing_time = time.perf_counter() - start_time
                self._update_statistics(True, processing_time)
                return best_allocation
            else:
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self._update_statistics(False, processing_time)
            print(f"❌ KSP_LB算法执行错误: {e}")
            return None
    
    def _build_load_aware_graph(self, request: Request, network: 'NetworkSimulator') -> nx.Graph:
        """构建负载感知的可用链路图"""
        G = nx.Graph()
        G.add_nodes_from(network.graph.nodes())

        for i, j, data in network.graph.edges(data=True):
            available_bandwidth = data['capacity'] - data['used']
            
            # 检查基本资源要求
            if (available_bandwidth >= request.bandwidth and
                len(data['available_spectrum']) >= request.spectrum_slots):
                
                # 使用负载利用率作为权重，优先选择负载较轻的链路
                utilization = data['used'] / data['capacity'] if data['capacity'] > 0 else 0
                G.add_edge(i, j, weight=utilization + self.config.edge_weight_offset)

        return G
    
    def _calculate_load_balance_score(self, path: List[int], network: 'NetworkSimulator') -> float:
        """
        计算路径的平均链路利用率 Li

        按照参考规范：Li = total_utilization / 路径中的链路数量

        Args:
            path: 网络路径
            network: 网络模拟器实例

        Returns:
            平均链路利用率 Li（越小越好）
        """
        if len(path) <= 1:
            return 0.0

        # 计算路径上所有链路的总利用率
        total_utilization = 0.0
        link_count = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = network.graph[u][v]
            link_utilization = edge_data['used'] / edge_data['capacity'] if edge_data['capacity'] > 0 else 0
            total_utilization += link_utilization
            link_count += 1

        if link_count == 0:
            return 0.0

        # 按照参考规范计算平均链路利用率 Li
        Li = total_utilization / link_count

        return Li
    
    def _record_load_balance_selection(self, path_index: int, total_paths: int,
                                     available_blocks: int, Li: float) -> None:
        """记录负载均衡选择信息"""
        if 'load_balance_stats' not in self.allocation_stats:
            self.allocation_stats['load_balance_stats'] = {
                'total_evaluations': 0,
                'average_Li': 0.0,
                'min_Li': float('inf'),
                'max_Li': 0.0,
                'path_selection_distribution': {}
            }

        stats = self.allocation_stats['load_balance_stats']
        stats['total_evaluations'] += 1

        # 更新平均链路利用率 Li 统计
        current_avg = stats['average_Li']
        n = stats['total_evaluations']
        stats['average_Li'] = (current_avg * (n - 1) + Li) / n
        stats['min_Li'] = min(stats['min_Li'], Li)
        stats['max_Li'] = max(stats['max_Li'], Li)

        # 更新路径选择分布
        if path_index not in stats['path_selection_distribution']:
            stats['path_selection_distribution'][path_index] = 0
        stats['path_selection_distribution'][path_index] += 1
    
    def get_detailed_statistics(self) -> dict:
        """获取详细的算法统计信息"""
        base_stats = self.get_statistics()

        # 添加KSP_LB特有的统计信息
        if 'load_balance_stats' in self.allocation_stats:
            lb_stats = self.allocation_stats['load_balance_stats']
            base_stats['algorithm_specific'] = {
                'k_value': self.k,
                'algorithm_type': 'Reference Specification Compliant',
                'average_Li': lb_stats['average_Li'],
                'min_Li': lb_stats['min_Li'],
                'max_Li': lb_stats['max_Li'],
                'path_selection_distribution': lb_stats['path_selection_distribution']
            }

        return base_stats
    

    def configure_k_value(self, k: int) -> None:
        """动态配置K值"""
        if k <= 0:
            raise ValueError("K值必须大于0")
        
        old_k = self.k
        self.k = k
        self.config.k_value = k
        
        print(f"🔧 KSP_LB算法K值已更新: {old_k} -> {k}")
    
    def get_algorithm_info(self) -> dict:
        """获取算法信息"""
        return {
            'name': self.algorithm_name,
            'type': 'K-Shortest Path with Load Balancing (Reference Specification)',
            'description': '严格按照参考规范实现的KSP_LB算法：使用平均链路利用率选择路径，跳数作为次要标准',
            'parameters': {
                'k_value': self.k,
                'edge_weight_offset': self.config.edge_weight_offset
            },
            'characteristics': [
                '平均链路利用率选择',
                '跳数次要标准',
                '首次适应频谱分配',
                '严格规范遵循',
                '简单高效'
            ],
            'complexity': 'O(K * |E| * log|V| + K * L)',
            'best_use_case': '需要严格按照KSP_LB参考规范执行的场景'
        }


# 便捷函数
def create_ksp_lb_algorithm(k: int = 3) -> KSPLoadBalanceAllocation:
    """
    创建KSP_LB算法实例的便捷函数

    Args:
        k: K最短路径的K值

    Returns:
        KSPLoadBalanceAllocation实例
    """
    config = AlgorithmConfig(
        algorithm_name="KSP_LB",
        k_value=k
    )
    return KSPLoadBalanceAllocation(config)


# 使用示例和测试
if __name__ == "__main__":
    print("🔧 KSP_LB算法模块测试")
    print("=" * 50)
    
    # 创建算法实例
    algorithm = create_ksp_lb_algorithm(k=3)

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

    # 测试K值配置
    print(f"\n🔧 测试K值配置...")
    print(f"当前K值: {algorithm.k}")
    algorithm.configure_k_value(5)
    print(f"更新后K值: {algorithm.k}")

    print(f"\n📋 算法严格按照参考规范实现:")
    print(f"  - 步骤9: Li = total_utilization / 链路数量")
    print(f"  - 步骤11: 选择具有最小Li的路径")
    print(f"  - 步骤12: 相同Li时选择跳数最少的路径")
    print(f"  - 步骤13: 使用首次适应策略分配频谱")
    
    # 显示统计信息
    stats = algorithm.get_statistics()
    print(f"\n📊 算法统计信息:")
    print(f"总尝试次数: {stats['total_attempts']}")
    print(f"成功分配次数: {stats['successful_allocations']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    
    print("\n✅ KSP_LB算法模块测试完成！")
