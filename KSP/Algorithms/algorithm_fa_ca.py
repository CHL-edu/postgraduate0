#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准FA+CA融合算法模块 - 碎片感知与拥塞避免融合算法
实现标准的FA（碎片感知）+ CA（拥塞避免）融合算法

核心目标：综合考虑碎片成本（F_cut）、对齐成本（M_align）、路径跳数和频谱资源利用率，进行全局优化

算法流程：
1. 路径预选：筛选至少有一个可用频谱块的候选路径
2. 综合成本计算：A = (H × S + F_cut + M_align) / C
3. 最优路径选择：选择综合成本A最小的路径
4. 频谱分配：使用首次适应规则
"""

import networkx as nx
import time
import math
import numpy as np
import sys
import os
from typing import List, Tuple, Optional, Dict, Set

# 添加父目录到路径以导入模块
from algorithm_base import AllocationAlgorithm, AlgorithmConfig, Request, register_algorithm

@register_algorithm("FA_CA")
class FragmentationAwareCongestionAvoidanceAllocation(AllocationAlgorithm):
    """FA_CA: 标准碎片感知拥塞避免融合算法

    实现标准的FA+CA融合算法，综合考虑：
    - F_cut: 连续频谱块切割次数（碎片感知）
    - M_align: 相邻链路频谱失配增加量（拥塞避免）
    - H: 路径跳数
    - S: 距离自适应调制所需槽数
    - C: 路径总可用槽数

    综合成本公式：A = (H × S + F_cut + M_align) / C

    算法特点：
    - 标准FA算法：最小化频谱切割次数
    - 标准CA算法：最小化相邻链路失配
    - 融合优化：综合成本函数全局优化
    - 首次适应：标准频谱分配策略
    """

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        """
        初始化标准FA+CA融合算法

        Args:
            config: 算法配置参数，如果为None则使用默认配置
        """
        if config is None:
            config = AlgorithmConfig(
                algorithm_name="FA_CA",
                k_value=3  # K最短路径数量
            )
        else:
            config.algorithm_name = "FA_CA"

        super().__init__(config)
        self.k = config.k_value

        # FA+CA融合算法特有参数（移除物理层限制，专注逻辑层）
        self.distance_adaptive = False  # 禁用距离自适应调制（移除物理层限制）
        self.base_modulation_slots = 1  # 固定调制槽数（逻辑层）
        self.max_modulation_slots = 1   # 固定调制槽数（逻辑层）

        print(f"🔧 标准FA+CA融合算法初始化完成（已移除物理层限制）")
        print(f"   K值: {self.k}")
        print(f"   距离自适应调制: 已禁用（专注逻辑层资源分配）")
        print(f"   调制槽数: 固定为1（移除物理层复杂性）")

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """
        使用标准FA+CA融合算法分配频谱

        标准算法流程：
        1. 路径预选：筛选至少有一个可用频谱块的K条候选路径
        2. 综合成本计算：对每条路径计算 A = (H × S + F_cut + M_align) / C
        3. 最优路径选择：选择综合成本A最小的路径
        4. 频谱分配：使用首次适应规则分配频谱

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

            # 步骤1：路径预选 - 生成K条候选路径并筛选可行路径
            viable_paths = self._generate_viable_candidate_paths(request, network)

            if not viable_paths:
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

            # 步骤2&3：综合成本计算和最优路径选择
            best_path, best_spectrum_block = self._select_optimal_path_and_spectrum(
                viable_paths, request, network
            )

            if best_path is None or best_spectrum_block is None:
                processing_time = time.perf_counter() - start_time
                self._update_statistics(False, processing_time)
                return None

            processing_time = time.perf_counter() - start_time
            self._update_statistics(True, processing_time)

            # 记录FA+CA融合算法的选择信息
            self._record_fa_ca_decision(best_path, best_spectrum_block, len(viable_paths))

            return best_path, best_spectrum_block

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            self._update_statistics(False, processing_time)
            print(f"❌ 标准FA+CA融合算法执行错误: {e}")
            return None

    def _generate_viable_candidate_paths(self, request: Request, network: 'NetworkSimulator') -> List[List[int]]:
        """
        生成可行的候选路径（标准FA+CA算法步骤1）

        筛选至少有一个可用频谱块的K条候选路径

        Args:
            request: 网络请求对象
            network: 网络模拟器实例

        Returns:
            可行路径列表
        """
        try:
            # 构建基础图（只考虑基本连通性）
            G = self._build_available_graph(request, network)

            # 为每条边添加跳数权重
            for u, v in G.edges():
                G[u][v]['weight'] = 1

            # 计算K条最短路径
            try:
                candidate_paths = list(nx.shortest_simple_paths(
                    G, request.source, request.destination, weight='weight'
                ))[:self.k]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []

            # 筛选可行路径：至少有一个可用频谱块
            viable_paths = []
            for path in candidate_paths:
                possible_blocks = self._find_available_continuous_slots(path, request.spectrum_slots, network)
                if possible_blocks:  # 至少有一个可用频谱块
                    viable_paths.append(path)

            return viable_paths

        except Exception as e:
            print(f"路径预选错误: {e}")
            return []

    def _select_optimal_path_and_spectrum(self, viable_paths: List[List[int]],
                                        request: Request,
                                        network: 'NetworkSimulator') -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """
        选择最优路径和频谱分配（标准FA+CA算法步骤2&3）

        计算综合成本 A = (H × S + F_cut + M_align) / C 并选择最优路径

        Args:
            viable_paths: 可行路径列表
            request: 网络请求对象
            network: 网络模拟器实例

        Returns:
            (最优路径, 最优频谱块) 或 (None, None)
        """
        best_path = None
        best_spectrum_block = None
        min_comprehensive_cost = float('inf')

        for path in viable_paths:
            # 获取该路径的可用频谱块
            possible_blocks = self._find_available_continuous_slots(path, request.spectrum_slots, network)

            if not possible_blocks:
                continue

            # 计算路径的综合成本
            comprehensive_cost = self._calculate_comprehensive_cost(path, request, network)

            # 选择综合成本最小的路径
            if comprehensive_cost < min_comprehensive_cost:
                min_comprehensive_cost = comprehensive_cost
                best_path = path
                # 使用首次适应规则选择频谱块
                best_spectrum_block = possible_blocks[0]  # 首次适应

        return best_path, best_spectrum_block

    def _calculate_comprehensive_cost(self, path: List[int], request: Request, network: 'NetworkSimulator') -> float:
        """
        计算标准FA+CA融合算法的综合成本

        综合成本公式：A = (H × S + F_cut + M_align) / C
        其中：
        - H: 路径跳数
        - S: 距离自适应调制所需的槽数
        - F_cut: 碎片切割次数（FA算法）
        - M_align: 失配增加量（CA算法）
        - C: 当前路径上的总可用槽数

        Args:
            path: 网络路径
            request: 网络请求对象
            network: 网络模拟器实例

        Returns:
            综合成本A
        """
        try:
            # H: 路径跳数
            H = len(path) - 1

            # S: 距离自适应调制所需的槽数
            S = self._calculate_modulation_slots(path, request)

            # F_cut: 碎片切割次数（FA算法核心）
            F_cut = self._calculate_f_cut(path, request.spectrum_slots, network)

            # M_align: 失配增加量（CA算法核心）
            M_align = self._calculate_m_align(path, request.spectrum_slots, network)

            # C: 当前路径上的总可用槽数
            C = self._calculate_total_available_slots(path, network)

            # 避免除零错误
            if C == 0:
                return float('inf')

            # 计算综合成本
            comprehensive_cost = (H * S + F_cut + M_align) / C

            return comprehensive_cost

        except Exception as e:
            print(f"综合成本计算错误: {e}")
            return float('inf')

    def _calculate_modulation_slots(self, path: List[int], request: Request) -> int:
        """
        计算逻辑层调制所需的槽数（S）- 已移除物理层限制

        专注于逻辑层资源分配，不考虑物理距离和调制格式

        Args:
            path: 网络路径
            request: 网络请求对象

        Returns:
            固定的调制槽数S（逻辑层）
        """
        # 移除物理层距离自适应调制，使用固定值专注逻辑层分配
        return self.base_modulation_slots

    def _calculate_f_cut(self, path: List[int], slots_needed: int, network: 'NetworkSimulator') -> int:
        """
        计算碎片切割次数F_cut（标准FA算法核心）

        F_cut定义：分配频谱时对路径上各链路现有连续可用频谱块的"切割"次数总和

        例：若某链路有连续可用块[7-12]，分配时隙11会将其分割为[7-10]和[12-12]，则F_cut+1

        Args:
            path: 网络路径
            slots_needed: 需要的频谱槽数量
            network: 网络模拟器实例

        Returns:
            F_cut值（切割次数）
        """
        try:
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
                return 0  # 无法分配，不产生切割

            # 找到第一个可用的连续频谱块（首次适配）
            available_list = sorted(list(common_spectrum))
            allocation_start = None

            for i in range(len(available_list) - slots_needed + 1):
                if available_list[i + slots_needed - 1] == available_list[i] + slots_needed - 1:
                    allocation_start = available_list[i]
                    break

            if allocation_start is None:
                return 0  # 无法找到连续块

            allocation_slots = set(range(allocation_start, allocation_start + slots_needed))

            # 计算每条链路的切割次数
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_spectrum = network.get_edge_available_spectrum(u, v)

                # 计算该链路上的切割次数
                link_f_cut = self._calculate_link_f_cut(edge_spectrum, allocation_slots)
                total_f_cut += link_f_cut

            return total_f_cut

        except Exception as e:
            print(f"F_cut计算错误: {e}")
            return 0

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

        # 计算切割次数
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

    def _calculate_m_align(self, path: List[int], slots_needed: int, network: 'NetworkSimulator') -> int:
        """
        计算失配增加量M_align（标准CA算法核心）

        M_align定义：分配频谱时所有相邻链路对之间可用频谱失配的增加量

        例：如果链路AD分配槽8，并减少与相邻链路BA的公共可用频谱1个槽，则M_align += 1

        Args:
            path: 网络路径
            slots_needed: 需要的频谱槽数量
            network: 网络模拟器实例

        Returns:
            M_align值（失配增加量）
        """
        try:
            total_m_align = 0

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
                return 0  # 无法分配，不产生失配

            # 找到第一个可用的连续频谱块（首次适配）
            available_list = sorted(list(common_spectrum))
            allocation_start = None

            for i in range(len(available_list) - slots_needed + 1):
                if available_list[i + slots_needed - 1] == available_list[i] + slots_needed - 1:
                    allocation_start = available_list[i]
                    break

            if allocation_start is None:
                return 0  # 无法找到连续块

            allocation_slots = set(range(allocation_start, allocation_start + slots_needed))

            # 获取网络中所有链路（用于查找相邻链路）
            all_links = self._get_all_network_links(network)

            # 计算路径上每条链路与其相邻链路的失配增加
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_link = (u, v)

                # 找到该链路的所有相邻链路
                adjacent_links = self._get_adjacent_links(path_link, all_links)

                # 计算与每个相邻链路的失配增加
                for adj_link in adjacent_links:
                    m_align_increase = self._calculate_link_pair_misalignment_increase(
                        path_link, adj_link, allocation_slots, network
                    )
                    total_m_align += m_align_increase

            return total_m_align

        except Exception as e:
            print(f"M_align计算错误: {e}")
            return 0

    def _get_all_network_links(self, network: 'NetworkSimulator') -> List[Tuple[int, int]]:
        """
        获取网络中所有链路

        Args:
            network: 网络模拟器实例

        Returns:
            所有链路的列表
        """
        try:
            # 从网络拓扑中获取所有边
            topology = network.topology
            links = []
            for u, v in topology.edges():
                links.append((u, v))
                links.append((v, u))  # 双向链路
            return links
        except Exception:
            return []

    def _get_adjacent_links(self, target_link: Tuple[int, int],
                          all_links: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        获取指定链路的所有相邻链路

        相邻链路定义：与目标链路共享至少一个节点的其他链路

        Args:
            target_link: 目标链路 (u, v)
            all_links: 所有链路列表

        Returns:
            相邻链路列表
        """
        u, v = target_link
        adjacent_links = []

        for link in all_links:
            link_u, link_v = link
            # 检查是否共享节点且不是同一条链路
            if link != target_link and link != (v, u):
                if link_u == u or link_u == v or link_v == u or link_v == v:
                    adjacent_links.append(link)

        return adjacent_links

    def _calculate_link_pair_misalignment_increase(self, link1: Tuple[int, int], link2: Tuple[int, int],
                                                 allocation_slots: Set[int], network: 'NetworkSimulator') -> int:
        """
        计算两条相邻链路之间的失配增加量

        Args:
            link1: 第一条链路
            link2: 第二条链路
            allocation_slots: 要分配的频谱槽集合
            network: 网络模拟器实例

        Returns:
            失配增加量
        """
        try:
            # 获取两条链路的可用频谱
            u1, v1 = link1
            u2, v2 = link2

            spectrum1_before = network.get_edge_available_spectrum(u1, v1)
            spectrum2_before = network.get_edge_available_spectrum(u2, v2)

            # 计算分配前的公共可用频谱
            common_before = spectrum1_before & spectrum2_before

            # 模拟分配后的可用频谱
            spectrum1_after = spectrum1_before - allocation_slots
            spectrum2_after = spectrum2_before  # 相邻链路不直接受影响

            # 计算分配后的公共可用频谱
            common_after = spectrum1_after & spectrum2_after

            # 失配增加量 = 分配前公共频谱 - 分配后公共频谱
            misalignment_increase = len(common_before) - len(common_after)

            return max(0, misalignment_increase)

        except Exception:
            return 0

    def _calculate_total_available_slots(self, path: List[int], network: 'NetworkSimulator') -> int:
        """
        计算路径上的总可用槽数C

        Args:
            path: 网络路径
            network: 网络模拟器实例

        Returns:
            总可用槽数C
        """
        try:
            total_available = 0

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_spectrum = network.get_edge_available_spectrum(u, v)
                total_available += len(edge_spectrum)

            return total_available

        except Exception:
            return 1  # 避免除零错误

    def _build_available_graph(self, request: Request, network: 'NSFNetWithLogging') -> nx.Graph:
        """
        构建基础可用图（只考虑基本连通性）

        Args:
            request: 网络请求对象
            network: NSFNet网络实例

        Returns:
            基础可用图
        """
        G = nx.Graph()
        G.add_nodes_from(network.graph.nodes())

        # 只添加满足基本资源要求的边
        for u, v, data in network.graph.edges(data=True):
            available_spectrum = network.get_edge_available_spectrum(u, v)

            # 检查是否有足够的可用频谱槽
            if len(available_spectrum) >= request.spectrum_slots:
                G.add_edge(u, v)

        return G

    def _record_fa_ca_decision(self, path: List[int], spectrum_block: List[int], num_viable_paths: int) -> None:
        """记录FA+CA融合算法的决策信息"""
        if 'fa_ca_decision_stats' not in self.allocation_stats:
            self.allocation_stats['fa_ca_decision_stats'] = {
                'total_decisions': 0,
                'average_path_length': 0.0,
                'average_viable_paths': 0.0,
                'path_length_distribution': {},
                'spectrum_block_size_distribution': {},
                'comprehensive_cost_history': []
            }

        stats = self.allocation_stats['fa_ca_decision_stats']
        stats['total_decisions'] += 1

        # 更新路径长度统计
        path_length = len(path) - 1
        current_avg_length = stats['average_path_length']
        n = stats['total_decisions']
        stats['average_path_length'] = (current_avg_length * (n - 1) + path_length) / n

        # 更新可行路径数统计
        current_avg_viable = stats['average_viable_paths']
        stats['average_viable_paths'] = (current_avg_viable * (n - 1) + num_viable_paths) / n

        # 更新分布统计
        if path_length not in stats['path_length_distribution']:
            stats['path_length_distribution'][path_length] = 0
        stats['path_length_distribution'][path_length] += 1

        block_size = len(spectrum_block)
        if block_size not in stats['spectrum_block_size_distribution']:
            stats['spectrum_block_size_distribution'][block_size] = 0
        stats['spectrum_block_size_distribution'][block_size] += 1

    def get_detailed_statistics(self) -> dict:
        """获取详细的算法统计信息"""
        base_stats = self.get_statistics()

        # 添加标准FA+CA融合算法特有的统计信息
        if 'fa_ca_decision_stats' in self.allocation_stats:
            fa_ca_stats = self.allocation_stats['fa_ca_decision_stats']
            base_stats['algorithm_specific'] = {
                'k_value': self.k,
                'algorithm_type': 'Standard FA+CA Fusion Algorithm',
                'distance_adaptive': self.distance_adaptive,
                'base_modulation_slots': self.base_modulation_slots,
                'max_modulation_slots': self.max_modulation_slots,
                'average_path_length': fa_ca_stats['average_path_length'],
                'average_viable_paths': fa_ca_stats['average_viable_paths'],
                'path_length_distribution': fa_ca_stats['path_length_distribution'],
                'spectrum_block_size_distribution': fa_ca_stats['spectrum_block_size_distribution']
            }

        return base_stats

    def configure_k_value(self, k: int) -> None:
        """动态配置K值"""
        if k <= 0:
            raise ValueError("K值必须大于0")

        old_k = self.k
        self.k = k
        self.config.k_value = k

        print(f"🔧 标准FA+CA融合算法K值已更新: {old_k} -> {k}")

    def configure_modulation_parameters(self, base_slots: int, max_slots: int, distance_adaptive: bool = False) -> None:
        """配置逻辑层调制参数（已移除物理层限制）"""
        if base_slots <= 0:
            raise ValueError("调制参数必须为正数")

        self.base_modulation_slots = base_slots
        self.max_modulation_slots = base_slots  # 固定为相同值，移除物理层复杂性
        self.distance_adaptive = False  # 强制禁用物理层距离自适应

        print(f"🔧 逻辑层调制参数已更新: 固定槽数={base_slots}（已移除物理层限制）")

    def get_algorithm_info(self) -> dict:
        """获取算法信息"""
        return {
            'name': self.algorithm_name,
            'type': 'Standard FA+CA Fusion Algorithm',
            'description': '标准FA+CA融合算法：综合考虑碎片成本(F_cut)、对齐成本(M_align)、路径跳数和频谱资源利用率（专注逻辑层）',
            'parameters': {
                'k_value': self.k,
                'distance_adaptive': self.distance_adaptive,
                'base_modulation_slots': self.base_modulation_slots,
                'max_modulation_slots': self.max_modulation_slots
            },
            'characteristics': [
                '标准FA算法 (F_cut最小化)',
                '标准CA算法 (M_align最小化)',
                '融合优化 (综合成本函数)',
                '逻辑层资源分配（已移除物理层限制）',
                '首次适应频谱分配',
                'K最短路径候选'
            ],
            'complexity': 'O(K × P × (S + L))',
            'best_use_case': '需要同时优化频谱碎片化和链路拥塞的光网络环境',
            'algorithm_details': {
                'comprehensive_cost_formula': 'A = (H × S + F_cut + M_align) / C',
                'components': {
                    'H': '路径跳数',
                    'S': '距离自适应调制所需槽数',
                    'F_cut': '碎片切割次数（FA算法）',
                    'M_align': '失配增加量（CA算法）',
                    'C': '路径总可用槽数'
                },
                'path_selection': '综合成本A最小的路径',
                'spectrum_allocation': '首次适应规则',
                'fragmentation_metric': 'F_cut (连续块切割次数)',
                'congestion_metric': 'M_align (相邻链路失配增加)'
            }
        }


# 便捷函数
def create_fa_ca_algorithm(k: int = 3) -> FragmentationAwareCongestionAvoidanceAllocation:
    """
    创建标准FA+CA融合算法实例的便捷函数

    Args:
        k: K最短路径的K值

    Returns:
        FragmentationAwareCongestionAvoidanceAllocation实例
    """
    config = AlgorithmConfig(
        algorithm_name="FA_CA",
        k_value=k
    )
    algorithm = FragmentationAwareCongestionAvoidanceAllocation(config)

    # 设置FA+CA特有参数
    algorithm.distance_adaptive = True
    algorithm.base_modulation_slots = 1
    algorithm.max_modulation_slots = 4

    return algorithm


# 使用示例和测试
if __name__ == "__main__":
    print("🔧 标准FA+CA融合算法模块测试")
    print("=" * 50)

    # 创建算法实例
    algorithm = create_fa_ca_algorithm(k=3)

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
    print(f"  综合成本公式: {details['comprehensive_cost_formula']}")
    print(f"  成本组件:")
    for component, description in details['components'].items():
        print(f"    {component}: {description}")

    # 测试参数配置
    print(f"\n🔧 测试参数配置...")
    print(f"当前K值: {algorithm.k}")
    algorithm.configure_k_value(5)
    print(f"更新后K值: {algorithm.k}")

    algorithm.configure_modulation_parameters(2, 6, True)

    # 显示统计信息
    stats = algorithm.get_statistics()
    print(f"\n📊 算法统计信息:")
    print(f"总尝试次数: {stats['total_attempts']}")
    print(f"成功分配次数: {stats['successful_allocations']}")
    print(f"成功率: {stats['success_rate']:.2%}")

    print("\n✅ 标准FA+CA融合算法模块测试完成！")