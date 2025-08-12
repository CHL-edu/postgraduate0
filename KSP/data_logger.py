#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFNet 详细数据记录系统
为每种算法在每个时隙记录完整的仿真数据
"""

import os
import csv
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class RequestDecision:
    """请求分配决策数据结构"""
    request_id: int
    source_node: int
    target_node: int
    bandwidth_demand: float
    spectrum_slots_demand: int
    arrival_time: float
    duration: float
    selected_path: List[int]
    allocated_spectrum_start: int
    allocated_spectrum_end: int
    allocation_status: str  # 'success' or 'blocked'
    failure_reason: str = ""
    algorithm_used: str = ""
    processing_time: float = 0.0


@dataclass
class TimeSlotMetrics:
    """时隙性能指标数据结构"""
    timeslot: int
    simulation_time: float
    total_requests: int
    successful_requests: int
    blocked_requests: int
    blocking_rate: float
    spectrum_utilization: float
    throughput: float
    fragmentation_rate: float
    average_delay: float
    average_hop_count: float
    active_requests_count: int
    total_bandwidth_demand: float
    successful_bandwidth: float
    computational_overhead: float  # 计算开销（秒）


@dataclass
class NetworkState:
    """网络状态监控数据结构"""
    timeslot: int
    simulation_time: float
    link_utilizations: Dict[str, float]
    active_requests: List[int]
    total_network_load: float
    spectrum_fragments_per_link: Dict[str, int]
    average_link_utilization: float
    max_link_utilization: float
    min_link_utilization: float


class DataLogger:
    """详细数据记录系统"""
    
    def __init__(self, base_log_dir: str = r"E:\PythonProject\KSP\log", enable_logging: bool = True):
        """
        初始化数据记录器
        
        Args:
            base_log_dir: 日志基础目录
            enable_logging: 是否启用数据记录
        """
        self.base_log_dir = base_log_dir
        self.enable_logging = enable_logging
        self.current_algorithm = ""
        self.current_timeslot = 0
        
        if self.enable_logging:
            self._create_directory_structure()
    
    def _create_directory_structure(self):
        """创建日志目录结构"""
        directories = [
            "spectrum_state",
            "request_decisions", 
            "performance_metrics",
            "network_state"
        ]
        
        algorithms = [
            "KSP_FF",
            "Random_Fit",
            "FA_CA",
            "KSP_LB",
            "KSP_FA",
            "Custom_Algorithm",
            "Custom_Algorithm_Random"
        ]
        
        # 创建基础目录
        os.makedirs(self.base_log_dir, exist_ok=True)
        
        # 为每个数据类型创建目录
        for directory in directories:
            dir_path = os.path.join(self.base_log_dir, directory)
            os.makedirs(dir_path, exist_ok=True)
            
            # 为每个算法创建子目录
            for algorithm in algorithms:
                alg_dir = os.path.join(dir_path, algorithm)
                os.makedirs(alg_dir, exist_ok=True)
        
        print(f"数据记录目录结构已创建: {self.base_log_dir}")
    
    def set_current_algorithm(self, algorithm_name: str):
        """设置当前算法名称"""
        # 标准化算法名称
        name_mapping = {
            "KSP_FF(K=3)": "KSP_FF",
            "Random Fit": "Random_Fit",
            "FA_CA": "FA_CA", 
            "KSP_LB": "KSP_LB",
            "KSP_FA": "KSP_FA",
            "Custom Algorithm": "Custom_Algorithm"
        }
        
        self.current_algorithm = name_mapping.get(algorithm_name, algorithm_name.replace(" ", "_"))
        self.current_timeslot = 0
    
    def log_request_decision(self, decision: RequestDecision):
        """记录请求分配决策数据"""
        if not self.enable_logging:
            return

        file_path = os.path.join(
            self.base_log_dir,
            "request_decisions",
            self.current_algorithm,
            f"timeslot_{self.current_timeslot}.csv"
        )

        # 检查文件是否存在，如果不存在则写入表头
        file_exists = os.path.exists(file_path)

        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(decision).keys()))

            if not file_exists:
                writer.writeheader()

            # 修复数据序列化问题：转换numpy类型和控制精度
            cleaned_data = self._clean_decision_data(asdict(decision))
            writer.writerow(cleaned_data)

    def _clean_decision_data(self, data: Dict) -> Dict:
        """清理请求决策数据，修复序列化问题"""
        cleaned = data.copy()

        # 修复路径数据的numpy类型序列化问题
        if 'selected_path' in cleaned and cleaned['selected_path']:
            if isinstance(cleaned['selected_path'], list):
                # 转换numpy类型为标准Python类型
                cleaned['selected_path'] = str([int(node) for node in cleaned['selected_path']])
            else:
                cleaned['selected_path'] = str(cleaned['selected_path'])

        # 控制浮点数精度，提高可读性
        float_fields = ['bandwidth_demand', 'arrival_time', 'duration', 'processing_time']
        for field in float_fields:
            if field in cleaned and isinstance(cleaned[field], (int, float)):
                cleaned[field] = round(float(cleaned[field]), 6)

        return cleaned
    
    def log_spectrum_matrix(self, spectrum_matrix: np.ndarray, edge_labels: List[str]):
        """
        记录二维频谱占用矩阵
        
        Args:
            spectrum_matrix: 频谱占用矩阵 (edges x spectrum_slots)
            edge_labels: 边标签列表 (如 ["0-1", "1-2", ...])
        """
        if not self.enable_logging:
            return
            
        file_path = os.path.join(
            self.base_log_dir,
            "spectrum_state", 
            self.current_algorithm,
            f"timeslot_{self.current_timeslot}.csv"
        )
        
        # 创建DataFrame并保存
        df = pd.DataFrame(spectrum_matrix.T, columns=edge_labels)
        df.index.name = 'spectrum_slot'
        df.to_csv(file_path)
    
    def log_performance_metrics(self, metrics: TimeSlotMetrics):
        """记录时隙性能指标"""
        if not self.enable_logging:
            return
            
        file_path = os.path.join(
            self.base_log_dir,
            "performance_metrics",
            self.current_algorithm, 
            "metrics.csv"
        )
        
        # 检查文件是否存在，如果不存在则写入表头
        file_exists = os.path.exists(file_path)
        
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(metrics).keys()))
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(asdict(metrics))
    
    def log_network_state(self, state: NetworkState):
        """记录网络状态监控数据"""
        if not self.enable_logging:
            return
            
        file_path = os.path.join(
            self.base_log_dir,
            "network_state",
            self.current_algorithm,
            f"timeslot_{self.current_timeslot}.json"
        )
        
        # 将数据转换为可序列化的格式
        state_dict = asdict(state)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    def increment_timeslot(self):
        """增加时隙计数器"""
        self.current_timeslot += 1

    def get_current_timeslot(self) -> int:
        """获取当前时隙"""
        return self.current_timeslot

    def ensure_timeslot_consistency(self):
        """确保所有数据类型的时隙文件一致性"""
        if not self.enable_logging:
            return

        data_types = ["spectrum_state", "request_decisions", "network_state"]

        for data_type in data_types:
            dir_path = os.path.join(self.base_log_dir, data_type, self.current_algorithm)
            if not os.path.exists(dir_path):
                continue

            # 检查当前时隙的文件是否存在
            if data_type == "network_state":
                file_path = os.path.join(dir_path, f"timeslot_{self.current_timeslot}.json")
                if not os.path.exists(file_path):
                    # 创建空的网络状态文件
                    empty_state = NetworkState(
                        timeslot=self.current_timeslot,
                        simulation_time=0.0,
                        link_utilizations={},
                        active_requests=[],
                        total_network_load=0.0,
                        spectrum_fragments_per_link={},
                        average_link_utilization=0.0,
                        max_link_utilization=0.0,
                        min_link_utilization=0.0
                    )
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(asdict(empty_state), f, indent=2, ensure_ascii=False)

            elif data_type == "spectrum_state":
                file_path = os.path.join(dir_path, f"timeslot_{self.current_timeslot}.csv")
                if not os.path.exists(file_path):
                    # 创建空的频谱状态文件
                    self._create_empty_spectrum_file(file_path)

            elif data_type == "request_decisions":
                file_path = os.path.join(dir_path, f"timeslot_{self.current_timeslot}.csv")
                if not os.path.exists(file_path):
                    # 创建空的请求决策文件
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            'request_id', 'source_node', 'target_node', 'bandwidth_demand',
                            'spectrum_slots_demand', 'arrival_time', 'duration', 'selected_path',
                            'allocated_spectrum_start', 'allocated_spectrum_end', 'allocation_status',
                            'failure_reason', 'algorithm_used', 'processing_time'
                        ])

    def _create_empty_spectrum_file(self, file_path: str):
        """创建空的频谱状态文件"""
        # 创建标准的NSFNet拓扑并生成正确排序的边标签
        import networkx as nx

        # 创建NSFNet拓扑
        G = nx.Graph()
        edges = [
            (0, 1), (0, 3), (1, 2), (1, 3), (2, 5), (3, 4), (3, 6), (4, 5), (4, 6), (5, 8),
            (6, 7), (7, 8), (7, 10), (8, 13), (9, 10), (9, 11), (9, 12), (10, 11), (11, 12),
            (12, 13), (6, 9)
        ]
        G.add_edges_from(edges)

        # 使用统一的边标签生成方法
        edge_labels = SpectrumMatrixBuilder.get_edge_labels(G)

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['spectrum_slot'] + edge_labels)
            # 从配置读取频谱槽数量
            from config_singleton import get_config
            config = get_config()
            log_config = config.get_logging_config()
            num_spectrum_slots = log_config['default_spectrum_slots_for_empty_files']
            # 写入全0数据
            for slot in range(num_spectrum_slots):
                row = [slot] + [0] * len(edge_labels)
                writer.writerow(row)
    
    def create_summary_report(self):
        """创建汇总报告"""
        if not self.enable_logging:
            return
            
        summary_path = os.path.join(self.base_log_dir, "summary_report.json")
        
        summary = {
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_timeslots": self.current_timeslot,
            "algorithms_tested": [],
            "data_structure": {
                "spectrum_state": "二维频谱占用矩阵 (CSV格式)",
                "request_decisions": "请求分配决策数据 (CSV格式)", 
                "performance_metrics": "时隙性能指标 (CSV格式)",
                "network_state": "网络状态监控数据 (JSON格式)"
            }
        }
        
        # 扫描已记录的算法
        spectrum_dir = os.path.join(self.base_log_dir, "spectrum_state")
        if os.path.exists(spectrum_dir):
            summary["algorithms_tested"] = [d for d in os.listdir(spectrum_dir) 
                                          if os.path.isdir(os.path.join(spectrum_dir, d))]
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"汇总报告已生成: {summary_path}")


class SpectrumMatrixBuilder:
    """频谱矩阵构建器"""
    
    @staticmethod
    def build_spectrum_matrix(network_graph, edge_labels: List[str]) -> np.ndarray:
        """
        构建二维频谱占用矩阵
        
        Args:
            network_graph: NetworkX图对象
            edge_labels: 边标签列表
            
        Returns:
            频谱占用矩阵 (edges x spectrum_slots)
        """
        if not edge_labels:
            return np.array([])
        
        # 获取频谱槽数量
        first_edge_data = list(network_graph.edges(data=True))[0][2]
        # 从配置读取默认频谱槽数量
        from config_singleton import get_config
        config = get_config()
        log_config = config.get_logging_config()
        default_slots = log_config['default_spectrum_slots_for_empty_files']
        num_spectrum_slots = first_edge_data.get('total_spectrum', default_slots)
        
        # 初始化矩阵
        matrix = np.zeros((len(edge_labels), num_spectrum_slots), dtype=int)
        
        # 填充矩阵
        for i, edge_label in enumerate(edge_labels):
            # 解析边标签 (如 "0-1")
            nodes = edge_label.split('-')
            if len(nodes) == 2:
                u, v = int(nodes[0]), int(nodes[1])
                
                if network_graph.has_edge(u, v):
                    edge_data = network_graph[u][v]
                    used_spectrum = edge_data.get('used_spectrum', set())
                    
                    # 标记已使用的频谱槽
                    for slot in used_spectrum:
                        if 0 <= slot < num_spectrum_slots:
                            matrix[i, slot] = 1
        
        return matrix
    
    @staticmethod
    def get_edge_labels(network_graph) -> List[str]:
        """获取网络边标签列表，按源节点编号从小到大排序"""
        edge_labels = []
        for u, v in network_graph.edges():
            edge_labels.append(f"{u}-{v}")

        # 按源节点编号进行数值排序，而非字符串排序
        def sort_key(edge_label):
            nodes = edge_label.split('-')
            if len(nodes) == 2:
                try:
                    source_node = int(nodes[0])
                    target_node = int(nodes[1])
                    # 先按源节点排序，再按目标节点排序
                    return (source_node, target_node)
                except ValueError:
                    # 如果转换失败，回退到字符串排序
                    return (edge_label, edge_label)
            return (edge_label, edge_label)

        return sorted(edge_labels, key=sort_key)
