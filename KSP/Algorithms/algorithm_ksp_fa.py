#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KSP碎片感知算法模块 - K最短路径碎片感知算法
标准FA算法实现：通过最小化频谱分配时对现有连续可用频谱块的"切割"次数，减少带宽碎片化。

核心思想：
1. 路径候选集生成：计算K条最短路径
2. "切割"成本计算：对每条路径模拟频谱分配，统计对连续可用频谱块的切割次数
3. 最优路径选择：选择F_cut最小的路径，若相同则选最短路径
4. 频谱分配：使用首次适配规则分配频谱
"""

import networkx as nx
import time
import sys
import os
from typing import List, Tuple, Optional, Set

# 添加父目录到路径以导入模块
from algorithm_base import AllocationAlgorithm, AlgorithmConfig, Request, register_algorithm

@register_algorithm("KSP_FA")
class KSPFragmentationAwareAllocation(AllocationAlgorithm):
    """KSP_FA: K最短路径碎片感知算法（标准FA算法实现）

    核心目标：通过最小化频谱分配时对现有连续可用频谱块的"切割"次数，减少带宽碎片化。

    算法特点：
    - 切割感知：统计分配时对连续频谱块的切割次数（F_cut）
    - 路径优选：选择F_cut最小的路径进行分配
    - 首次适配：在选定路径上使用首次适配规则分配频谱
    - 最短路径优先：F_cut相同时优先选择最短路径
    """

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        """
        初始化KSP_FA算法（标准FA算法实现）

        Args:
            config: 算法配置参数，如果为None则使用默认配置
        """
        if config is None:
            config = AlgorithmConfig(
                algorithm_name="KSP_FA",
                k_value=3
            )
        else:
            config.algorithm_name = "KSP_FA"

        super().__init__(config)
        self.k = config.k_value

        print(f"🔧 标准FA算法初始化完成，K={self.k}")
        print(f"   核心目标：最小化连续频谱块切割次数（F_cut）")

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """
        标准FA算法实现：通过最小化切割次数选择最优路径和频谱分配

        算法流程（标准FA算法）：
        1. 路径候选集生成：计算K条最短路径
        2. "切割"成本计算：对每条路径模拟频谱分配，统计F_cut
        3. 最优路径选择：选择F_cut最小的路径，若相同则选最短路径
        4. 频谱分配：在选定路径上使用首次适配规则分配频谱

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

            # 步骤1：路径候选集生成 - 构建可用链路图并计算K条最短路径
            G = self._build_available_graph(request, network)

            # 为每条边添加跳数权重（用于路径长度计算）
            for u, v in G.edges():
                G[u][v]['weight'] = 1

            # 计算K条最短路径作为候选路径集
            try:
                candidate_paths = list(nx.shortest_simple_paths(
                    G, request.source, request.destination, weight='weight'
                ))[:self.k]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

            # 步骤2&3：对每条候选路径计算F_cut并选择最优路径
            best_path = None
            best_spectrum_block = None
            min_f_cut = float('inf')
            min_path_length = float('inf')

            for path in candidate_paths:
                # 检查路径上是否有足够的连续频谱槽
                possible_blocks = self._find_available_continuous_slots(path, request.spectrum_slots, network)

                if not possible_blocks:
                    continue

                # 计算该路径的F_cut（切割成本）
                f_cut = self._calculate_f_cut(path, request.spectrum_slots, network)
                path_length = len(path) - 1  # 路径跳数

                # 选择F_cut最小的路径；若F_cut相同，选择最短路径
                if (f_cut < min_f_cut) or (f_cut == min_f_cut and path_length < min_path_length):
                    min_f_cut = f_cut
                    min_path_length = path_length
                    best_path = path
                    # 步骤4：在选定路径上使用首次适配规则选择频谱块
                    best_spectrum_block = possible_blocks[0]  # 首次适配：选择第一个可用块

                    # 记录选择信息
                    self._record_fa_selection(path, f_cut, path_length, len(possible_blocks))

            if best_path and best_spectrum_block:
                processing_time = time.perf_counter() - start_time
                self._update_statistics(True, processing_time)
                return best_path, best_spectrum_block
            else:
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self._update_statistics(False, processing_time)
            print(f"❌ 标准FA算法执行错误: {e}")
            return None
    
    def _calculate_f_cut(self, path: List[int], slots_needed: int, network: 'NetworkSimulator') -> int:
        """
        计算路径的F_cut（切割成本）- 标准FA算法核心函数

        F_cut定义：分配频谱时对路径上各链路现有连续可用频谱块的"切割"次数总和

        例：若某链路有连续可用块[7-12]，分配时隙11会将其分割为[7-10]和[12-12]，则F_cut+1

        Args:
            path: 网络路径
            slots_needed: 需要的频谱槽数量
            network: 网络模拟器实例

        Returns:
            F_cut值（切割次数，越小越好）
        """
        total_f_cut = 0

        # 获取路径上所有链路的可用频谱交集
        common_spectrum = None
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_spectrum = network.get_edge_available_spectrum(u, v)
            if common_spectrum is None:
                common_spectrum = edge_spectrum.copy()
            else:
                common_spectrum &= edge_spectrum

        if not common_spectrum or len(common_spectrum) < slots_needed:
            return float('inf')  # 无法分配

        # 找到第一个可用的连续频谱块（首次适配）
        available_list = sorted(list(common_spectrum))
        allocation_start = None

        for i in range(len(available_list) - slots_needed + 1):
            if available_list[i + slots_needed - 1] == available_list[i] + slots_needed - 1:
                allocation_start = available_list[i]
                break

        if allocation_start is None:
            return float('inf')  # 无法找到连续块

        allocation_slots = set(range(allocation_start, allocation_start + slots_needed))

        # 计算每条链路的切割次数
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_spectrum = network.get_edge_available_spectrum(u, v)

            # 计算该链路上的切割次数
            link_f_cut = self._calculate_link_f_cut(edge_spectrum, allocation_slots)
            total_f_cut += link_f_cut

        return total_f_cut

    def _calculate_link_f_cut(self, available_spectrum: Set[int], allocation_slots: Set[int]) -> int:
        """
        计算单条链路的F_cut（切割次数）

        Args:
            available_spectrum: 链路可用频谱槽集合
            allocation_slots: 要分配的频谱槽集合

        Returns:
            该链路的切割次数
        """
        if not allocation_slots.issubset(available_spectrum):
            return 0  # 无法在此链路分配，不产生切割

        # 找出分配前的连续块
        available_list = sorted(list(available_spectrum))
        continuous_blocks_before = self._find_continuous_blocks(available_list)

        # 模拟分配后的可用频谱
        remaining_spectrum = available_spectrum - allocation_slots
        remaining_list = sorted(list(remaining_spectrum))
        continuous_blocks_after = self._find_continuous_blocks(remaining_list)

        # 计算切割次数：分配后的块数 - 分配前被影响的块数 + 被完全占用的块数
        f_cut = 0

        # 检查每个分配前的连续块是否被切割
        for block_start, block_end in continuous_blocks_before:
            block_slots = set(range(block_start, block_end + 1))

            # 如果分配的频谱与该块有交集但不完全包含该块，则产生切割
            if allocation_slots & block_slots and not allocation_slots >= block_slots:
                # 检查是否将一个块切割成多个块
                remaining_in_block = block_slots - allocation_slots
                if remaining_in_block:
                    remaining_blocks_in_this_block = self._find_continuous_blocks(sorted(list(remaining_in_block)))
                    # 切割次数 = 切割后的块数 - 1
                    f_cut += max(0, len(remaining_blocks_in_this_block) - 1)

        return f_cut

    def _find_continuous_blocks(self, spectrum_list: List[int]) -> List[Tuple[int, int]]:
        """
        找出频谱列表中的所有连续块

        Args:
            spectrum_list: 排序后的频谱槽列表

        Returns:
            连续块列表，每个元素为(start, end)元组
        """
        if not spectrum_list:
            return []

        blocks = []
        start = spectrum_list[0]
        end = spectrum_list[0]

        for i in range(1, len(spectrum_list)):
            if spectrum_list[i] == end + 1:
                end = spectrum_list[i]
            else:
                blocks.append((start, end))
                start = spectrum_list[i]
                end = spectrum_list[i]

        blocks.append((start, end))
        return blocks
    
    def _record_fa_selection(self, path: List[int], f_cut: int, path_length: int, available_blocks: int) -> None:
        """记录FA算法选择信息"""
        if 'fa_selection_stats' not in self.allocation_stats:
            self.allocation_stats['fa_selection_stats'] = {
                'total_selections': 0,
                'average_f_cut': 0.0,
                'min_f_cut': float('inf'),
                'max_f_cut': 0,
                'average_path_length': 0.0,
                'f_cut_distribution': {},
                'path_length_distribution': {}
            }

        stats = self.allocation_stats['fa_selection_stats']
        stats['total_selections'] += 1

        # 更新F_cut统计
        current_avg_f_cut = stats['average_f_cut']
        n = stats['total_selections']
        stats['average_f_cut'] = (current_avg_f_cut * (n - 1) + f_cut) / n
        stats['min_f_cut'] = min(stats['min_f_cut'], f_cut)
        stats['max_f_cut'] = max(stats['max_f_cut'], f_cut)

        # 更新路径长度统计
        current_avg_length = stats['average_path_length']
        stats['average_path_length'] = (current_avg_length * (n - 1) + path_length) / n

        # 更新分布统计
        if f_cut not in stats['f_cut_distribution']:
            stats['f_cut_distribution'][f_cut] = 0
        stats['f_cut_distribution'][f_cut] += 1

        if path_length not in stats['path_length_distribution']:
            stats['path_length_distribution'][path_length] = 0
        stats['path_length_distribution'][path_length] += 1
    
    def get_detailed_statistics(self) -> dict:
        """获取详细的算法统计信息"""
        base_stats = self.get_statistics()

        # 添加标准FA算法特有的统计信息
        if 'fa_selection_stats' in self.allocation_stats:
            fa_stats = self.allocation_stats['fa_selection_stats']
            base_stats['algorithm_specific'] = {
                'k_value': self.k,
                'algorithm_type': 'Standard FA Algorithm',
                'average_f_cut': fa_stats['average_f_cut'],
                'min_f_cut': fa_stats['min_f_cut'],
                'max_f_cut': fa_stats['max_f_cut'],
                'average_path_length': fa_stats['average_path_length'],
                'f_cut_distribution': fa_stats['f_cut_distribution'],
                'path_length_distribution': fa_stats['path_length_distribution']
            }

        return base_stats

    def configure_k_value(self, k: int) -> None:
        """动态配置K值"""
        if k <= 0:
            raise ValueError("K值必须大于0")

        old_k = self.k
        self.k = k
        self.config.k_value = k

        print(f"🔧 标准FA算法K值已更新: {old_k} -> {k}")

    def get_algorithm_info(self) -> dict:
        """获取算法信息"""
        return {
            'name': self.algorithm_name,
            'type': 'Standard Fragmentation-Aware (FA) Algorithm',
            'description': '标准FA算法：通过最小化连续频谱块切割次数(F_cut)减少带宽碎片化',
            'parameters': {
                'k_value': self.k
            },
            'characteristics': [
                '切割感知 (F_cut最小化)',
                '路径候选集生成',
                '首次适配频谱分配',
                '最短路径优先',
                '标准FA算法实现'
            ],
            'complexity': 'O(K * |E| * log|V| + K * L * S)',
            'best_use_case': '需要减少频谱碎片化的光网络环境',
            'algorithm_details': {
                'core_objective': '最小化连续可用频谱块的切割次数',
                'path_selection': 'F_cut最小的路径，相同时选最短路径',
                'spectrum_allocation': '首次适配规则',
                'fragmentation_metric': 'F_cut (切割次数)'
            }
        }


# 便捷函数
def create_ksp_fa_algorithm(k: int = 3) -> KSPFragmentationAwareAllocation:
    """
    创建标准FA算法实例的便捷函数

    Args:
        k: K最短路径的K值

    Returns:
        KSPFragmentationAwareAllocation实例（标准FA算法实现）
    """
    config = AlgorithmConfig(
        algorithm_name="KSP_FA",
        k_value=k
    )
    return KSPFragmentationAwareAllocation(config)


# 使用示例和测试
if __name__ == "__main__":
    print("🔧 标准FA算法模块测试")
    print("=" * 50)

    # 创建算法实例
    algorithm = create_ksp_fa_algorithm(k=3)

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

    print(f"\n算法详细信息:")
    details = info['algorithm_details']
    print(f"  核心目标: {details['core_objective']}")
    print(f"  路径选择: {details['path_selection']}")
    print(f"  频谱分配: {details['spectrum_allocation']}")
    print(f"  碎片化指标: {details['fragmentation_metric']}")

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

    print("\n✅ 标准FA算法模块测试完成！")
