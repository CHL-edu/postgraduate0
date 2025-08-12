#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法基类和接口定义模块
为Algorithms目录下的算法提供兼容的基类和配置类
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Any, Dict
import networkx as nx


@dataclass
class Request:
    """网络请求类 - 与NSFNet_with_logging.py中的Request类兼容"""
    id: int
    source: int
    destination: int
    bandwidth: float  # 带宽需求（Gbps）
    arrival_time: float  # 到达时间（时间模式）或时隙编号（时隙模式）
    duration: float  # 持续时间（时间模式）或持续时隙数（时隙模式）
    spectrum_slots: int = 1  # 需要的频谱槽数量
    status: str = "pending"  # pending, active, completed, blocked
    path: List[int] = field(default_factory=list)
    wavelengths: Set[int] = field(default_factory=set)  # 为光网络预留
    allocated_spectrum: List[int] = field(default_factory=list)  # 分配的频谱槽索引


@dataclass
class AlgorithmConfig:
    """算法配置类"""
    algorithm_name: str = "Unknown"
    k_value: int = 3
    max_paths: int = 5
    edge_weight_offset: float = 0.01
    random_seed: Optional[int] = None
    
    # 扩展配置参数
    enable_statistics: bool = True
    enable_detailed_logging: bool = False
    timeout_seconds: float = 10.0


class AllocationAlgorithm(ABC):
    """分配算法抽象基类 - 与NSFNet_with_logging.py兼容"""

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        """初始化分配算法"""
        if config is None:
            config = AlgorithmConfig()
        
        self.config = config
        self.algorithm_name = config.algorithm_name
        
        # 统计信息
        self.allocation_stats = {
            'total_attempts': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }

    @abstractmethod
    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """分配请求到网络路径和频谱

        Args:
            request: 网络请求
            network: NSFNet网络实例

        Returns:
            Optional[Tuple[List[int], List[int]]]: (路径, 分配的频谱槽列表) 或 None
        """
        pass

    def get_algorithm_name(self) -> str:
        """获取算法名称"""
        return self.algorithm_name
    
    def _update_statistics(self, success: bool, processing_time: float) -> None:
        """更新算法统计信息"""
        self.allocation_stats['total_attempts'] += 1
        self.allocation_stats['total_processing_time'] += processing_time
        
        if success:
            self.allocation_stats['successful_allocations'] += 1
        else:
            self.allocation_stats['failed_allocations'] += 1
        
        # 计算平均处理时间
        if self.allocation_stats['total_attempts'] > 0:
            self.allocation_stats['average_processing_time'] = (
                self.allocation_stats['total_processing_time'] / 
                self.allocation_stats['total_attempts']
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取算法统计信息"""
        stats = self.allocation_stats.copy()
        stats['success_rate'] = (
            stats['successful_allocations'] / stats['total_attempts'] 
            if stats['total_attempts'] > 0 else 0.0
        )
        return stats
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.allocation_stats = {
            'total_attempts': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
    
    def _find_available_continuous_slots(self, path: List[int], slots_needed: int, 
                                       network: 'NSFNetWithLogging') -> List[List[int]]:
        """在路径上查找可用的连续频谱槽"""
        if len(path) <= 1:
            return []
        
        # 获取路径上所有边的公共可用频谱
        common_spectrum = None
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if network.graph.has_edge(u, v):
                edge_spectrum = network.graph[u][v]['available_spectrum']
                if common_spectrum is None:
                    common_spectrum = edge_spectrum.copy()
                else:
                    common_spectrum &= edge_spectrum
            else:
                return []  # 路径不存在
        
        if not common_spectrum or len(common_spectrum) < slots_needed:
            return []
        
        # 查找连续的频谱块
        available_list = sorted(list(common_spectrum))
        possible_blocks = []
        
        for i in range(len(available_list) - slots_needed + 1):
            start_slot = available_list[i]
            # 检查连续性
            if available_list[i + slots_needed - 1] == start_slot + slots_needed - 1:
                block = list(range(start_slot, start_slot + slots_needed))
                possible_blocks.append(block)
        
        return possible_blocks

    def _build_available_graph(self, request: Request, network: 'NSFNetWithLogging') -> nx.Graph:
        """构建满足资源要求的可用链路图"""
        G = nx.Graph()
        G.add_nodes_from(network.graph.nodes())

        for u, v, data in network.graph.edges(data=True):
            available_bandwidth = data['capacity'] - data['used']
            available_spectrum = data['available_spectrum']

            # 检查带宽和频谱槽要求
            if (available_bandwidth >= request.bandwidth and
                len(available_spectrum) >= request.spectrum_slots):
                G.add_edge(u, v)

        return G


# 算法注册装饰器
_registered_algorithms = {}

def register_algorithm(name: str):
    """算法注册装饰器"""
    def decorator(cls):
        _registered_algorithms[name] = cls
        return cls
    return decorator


def get_registered_algorithms() -> Dict[str, type]:
    """获取所有注册的算法"""
    return _registered_algorithms.copy()


def create_algorithm(name: str, config: Optional[AlgorithmConfig] = None) -> Optional[AllocationAlgorithm]:
    """创建算法实例"""
    if name in _registered_algorithms:
        return _registered_algorithms[name](config)
    return None
