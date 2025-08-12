#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFNet网络仿真模块 - 带详细数据记录版本
实现K最短路径算法和各种频谱分配策略，并记录详细的仿真数据
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Any

# 导入数据记录模块
from data_logger import (
    DataLogger, RequestDecision, TimeSlotMetrics, NetworkState,
    SpectrumMatrixBuilder
)

# 导入配置读取器和参数变化管理器
from config_singleton import get_config, OptimizedConfigReader
from parameter_change_manager import ParameterChangeManager

# 导入Algorithms目录下的算法实现
from Algorithms.algorithm_ksp_ff import KSPFirstFitAllocation as NewKSPFirstFitAllocation
from Algorithms.algorithm_ksp_lb import KSPLoadBalanceAllocation as NewLoadBalancedAllocation
from Algorithms.algorithm_ksp_fa import KSPFragmentationAwareAllocation as NewKSPFragmentationAwareAllocation
from Algorithms.algorithm_random_fit import RandomFitAllocation as NewKSPRandomFitAllocation
from Algorithms.algorithm_fa_ca import FragmentationAwareCongestionAvoidanceAllocation as NewFACongestionAwareAllocation

# 创建全局配置实例（使用单例模式）
config_reader = get_config()
parameter_manager = ParameterChangeManager(OptimizedConfigReader())

# 获取日志配置（backup_directory在logging段中）
logging_config = config_reader.get_logging_config()
backup_dir = logging_config['backup_directory']

# 确保备份目录存在
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)
    print(f"创建备份目录: {backup_dir}")

timestamp = time.strftime("%Y%m%d_%H%M%S")


@dataclass
class Request:
    """网络请求类"""
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
class TimeslotRequest:
    """时隙请求数据结构（专用于时隙模式）"""
    id: int
    source: int
    destination: int
    bandwidth: float
    spectrum_slots: int
    arrival_timeslot: int
    duration_timeslots: int


class AllocationAlgorithm(ABC):
    """分配算法抽象基类"""

    def __init__(self):
        """初始化分配算法"""
        self.algorithm_name = "Unknown Algorithm"  # 算法名称，用于可视化标题

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


# 适配器类，用于兼容新的算法实现
class AlgorithmAdapter(AllocationAlgorithm):
    """算法适配器，用于兼容新的算法实现"""

    def __init__(self, new_algorithm_instance):
        super().__init__()
        self.new_algorithm = new_algorithm_instance
        self.algorithm_name = new_algorithm_instance.get_algorithm_name()

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        return self.new_algorithm.allocate(request, network)

    def get_algorithm_name(self) -> str:
        return self.new_algorithm.get_algorithm_name()

    def _find_available_continuous_slots(self, path: List[int], slots_needed: int, network: 'NSFNetWithLogging') -> List[
        List[int]]:
        """
        一个辅助方法，用于在给定路径上查找所有可用的连续频谱块。
        """
        # 1. 查找路径上可用频谱的交集
        common_spectrum = None
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_spectrum = network.get_edge_available_spectrum(u, v)
            if common_spectrum is None:
                common_spectrum = edge_spectrum.copy()
            else:
                common_spectrum &= edge_spectrum

            # 如果可用频谱总数已经不够，提前退出
            if len(common_spectrum) < slots_needed:
                return []

        if not common_spectrum:
            return []

        # 2. 查找所有大小为 'slots_needed' 的连续块
        available_list = sorted(list(common_spectrum))
        possible_blocks = []
        # 只需迭代到可以容纳一个完整块的位置
        for i in range(len(available_list) - slots_needed + 1):
            start_slot = available_list[i]
            # 检查从当前位置开始的块是否连续
            if available_list[i + slots_needed - 1] == start_slot + slots_needed - 1:
                block = list(range(start_slot, start_slot + slots_needed))
                possible_blocks.append(block)
        return possible_blocks


class KSPFirstFitAllocation(AllocationAlgorithm):
    """KSP_FF: K最短路径首次适应算法

    该算法使用K最短路径算法寻找多条候选路径，然后在每条路径上
    使用首次适应策略分配频谱槽，选择第一个找到的可用频谱块。
    """

    def __init__(self, k: int = 3):
        """初始化KSP_FF算法 - 使用新的算法实现"""
        super().__init__()
        self.k = k
        self.algorithm_name = "KSP_FF"

        # 创建新算法实例
        from algorithm_base import AlgorithmConfig
        config = AlgorithmConfig(algorithm_name="KSP_FF", k_value=k)
        self.new_algorithm = NewKSPFirstFitAllocation(config)

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """使用新的KSP_FF算法实现"""
        return self.new_algorithm.allocate(request, network)


class KSPRandomFitAllocation(AllocationAlgorithm):
    """Random Fit: 真正的随机适应算法

    该算法完全基于随机选择，不依赖于K最短路径算法。
    在所有可能的路径中随机选择，并在选定路径上随机分配可用的频谱槽，
    最大化分配的随机性和多样性。
    """

    def __init__(self, max_paths: int = None):
        """初始化Random Fit算法 - 使用新的算法实现"""
        super().__init__()
        if max_paths is None:
            alg_config = config_reader.get_algorithm_config()
            self.max_paths = alg_config['random_fit_max_paths']
        else:
            self.max_paths = max_paths
        self.algorithm_name = "Random_Fit"

        # 创建新算法实例
        from algorithm_base import AlgorithmConfig
        config = AlgorithmConfig(algorithm_name="Random_Fit", max_paths=self.max_paths)
        self.new_algorithm = NewKSPRandomFitAllocation(config)

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """使用新的Random Fit算法实现"""
        return self.new_algorithm.allocate(request, network)


# 完整的频谱分配算法实现
# 包含6种不同的分配策略


class FACongestionAwareAllocation(AllocationAlgorithm):
    """FA-CA: 碎片感知拥塞避免算法

    该算法专注于减少频谱碎片化和避免网络拥塞，通过综合考虑
    频谱利用率、碎片化程度和链路负载来选择最优路径。
    """

    def __init__(self):
        """初始化FA_CA算法 - 使用新的算法实现"""
        super().__init__()
        self.algorithm_name = "FA_CA"

        # 创建新算法实例
        from algorithm_base import AlgorithmConfig
        config = AlgorithmConfig(algorithm_name="FA_CA", k_value=3)
        self.new_algorithm = NewFACongestionAwareAllocation(config)

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """使用新的FA_CA算法实现"""
        return self.new_algorithm.allocate(request, network)




class LoadBalancedAllocation(AllocationAlgorithm):
    """KSP_LB: K最短路径负载均衡算法

    该算法结合K最短路径和负载均衡策略，通过选择负载最轻的路径
    来平衡网络流量分布，避免某些链路过载而其他链路空闲。
    """

    def __init__(self, k: int = 3):
        """初始化KSP_LB算法 - 使用新的算法实现"""
        super().__init__()
        self.k = k
        self.algorithm_name = "KSP_LB"

        # 创建新算法实例
        from algorithm_base import AlgorithmConfig
        config = AlgorithmConfig(algorithm_name="KSP_LB", k_value=k)
        self.new_algorithm = NewLoadBalancedAllocation(config)

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """使用新的KSP_LB算法实现"""
        return self.new_algorithm.allocate(request, network)




class KSPFragmentationAwareAllocation(AllocationAlgorithm):
    """KSP_FA: K最短路径碎片感知算法

    该算法在K最短路径的基础上，专门考虑频谱碎片化问题，
    优先选择能够最小化频谱碎片化的路径和频谱分配方案。
    """

    def __init__(self, k: int = 3):
        """初始化KSP_FA算法 - 使用新的算法实现"""
        super().__init__()
        self.k = k
        self.algorithm_name = "KSP_FA"

        # 创建新算法实例
        from algorithm_base import AlgorithmConfig
        config = AlgorithmConfig(algorithm_name="KSP_FA", k_value=k)
        self.new_algorithm = NewKSPFragmentationAwareAllocation(config)

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """使用新的KSP_FA算法实现"""
        return self.new_algorithm.allocate(request, network)




class ExhaustiveOptimalAllocation(AllocationAlgorithm):
    """自定义穷举适合分配算法

    该算法按顺序尝试前面的5种算法（KSP_FF, KSP_FA, Random Fit, KSP_LB, FA-CA），
    选择第一个能够成功分配资源的算法，而非寻找最优解。这种策略能够
    快速找到可行解，提高分配成功率。
    """

    def __init__(self, k: int = 3):
        """
        初始化自定义穷举适合分配算法

        Args:
            k: K最短路径算法中的K值
        """
        super().__init__()
        self.k = k
        self.algorithm_name = "Custom_Algorithm"  # 统一算法名称格式

        # 按优先级顺序初始化所有可用的分配算法
        self.algorithms = [
            KSPFirstFitAllocation(k=k),           # 1. KSP_FF - 快速首次适应
            KSPFragmentationAwareAllocation(k=k), # 2. KSP_FA - 碎片感知
            KSPRandomFitAllocation(),             # 3. Random Fit - 随机适应
            LoadBalancedAllocation(k=k),          # 4. KSP_LB - 负载均衡
            FACongestionAwareAllocation()         # 5. FA-CA - 拥塞避免
        ]

        # 算法性能统计
        self.algorithm_stats = {alg.get_algorithm_name(): {'success': 0, 'total': 0, 'first_success': 0}
                                for alg in self.algorithms}

        # 记录每次分配使用的算法
        self.allocation_history = []

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """
        穷举适合分配：按顺序尝试算法，返回第一个成功的分配结果
        """
        allocation_attempt = {
            'request_id': request.id,
            'algorithms_tried': [],
            'successful_algorithm': None,
            'success': False
        }

        for i, algorithm in enumerate(self.algorithms):
            try:
                alg_name = algorithm.get_algorithm_name()
                self.algorithm_stats[alg_name]['total'] += 1
                allocation_attempt['algorithms_tried'].append(alg_name)

                result = algorithm.allocate(request, network)

                if result is not None:
                    # 找到第一个成功的分配
                    self.algorithm_stats[alg_name]['success'] += 1
                    if i == 0:  # 如果是第一个算法就成功
                        self.algorithm_stats[alg_name]['first_success'] += 1

                    allocation_attempt['successful_algorithm'] = alg_name
                    allocation_attempt['success'] = True
                    self.allocation_history.append(allocation_attempt)

                    return result

            except Exception as e:
                print(f"算法 {algorithm.get_algorithm_name()} 执行错误: {e}")
                continue

        # 所有算法都失败
        self.allocation_history.append(allocation_attempt)
        return None

    def get_algorithm_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取算法统计信息"""
        stats = {}
        for alg_name, stat in self.algorithm_stats.items():
            success_rate = stat['success'] / stat['total'] if stat['total'] > 0 else 0.0
            first_success_rate = stat['first_success'] / stat['total'] if stat['total'] > 0 else 0.0
            stats[alg_name] = {
                'success_count': stat['success'],
                'total_attempts': stat['total'],
                'success_rate': success_rate,
                'first_success_count': stat['first_success'],
                'first_success_rate': first_success_rate
            }
        return stats

    def get_allocation_summary(self) -> Dict[str, Any]:
        """获取分配总结信息"""
        total_requests = len(self.allocation_history)
        successful_requests = sum(1 for attempt in self.allocation_history if attempt['success'])

        # 统计每个算法作为成功算法的次数
        algorithm_success_count = {}
        for attempt in self.allocation_history:
            if attempt['success'] and attempt['successful_algorithm']:
                alg = attempt['successful_algorithm']
                algorithm_success_count[alg] = algorithm_success_count.get(alg, 0) + 1

        # 计算平均尝试算法数量
        total_attempts = sum(len(attempt['algorithms_tried']) for attempt in self.allocation_history)
        avg_attempts = total_attempts / total_requests if total_requests > 0 else 0

        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0.0,
            'algorithm_success_distribution': algorithm_success_count,
            'average_algorithms_tried': avg_attempts,
            'detailed_algorithm_stats': self.get_algorithm_statistics()
        }


class CustomAlgorithmRandom(AllocationAlgorithm):
    """Custom_Algorithm_Random: 随机算法选择器

    该算法在每次分配请求时，从其他5种算法（KSP_FF, KSP_FA, KSP_LB, FA_CA, Random_Fit）
    中随机选择一种来执行分配，增加算法选择的随机性和多样性。
    """

    def __init__(self, k: int = 3):
        """
        初始化随机算法选择器

        Args:
            k: K最短路径算法中的K值
        """
        super().__init__()
        self.k = k
        self.algorithm_name = "Custom_Algorithm_Random"

        # 初始化所有可用的分配算法
        self.algorithms = [
            KSPFirstFitAllocation(k=k),           # 1. KSP_FF - 快速首次适应
            KSPFragmentationAwareAllocation(k=k), # 2. KSP_FA - 碎片感知
            LoadBalancedAllocation(k=k),          # 3. KSP_LB - 负载均衡
            FACongestionAwareAllocation(),        # 4. FA-CA - 拥塞避免
            KSPRandomFitAllocation()              # 5. Random_Fit - 随机适应
        ]

        # 算法使用统计
        self.algorithm_usage_stats = {alg.get_algorithm_name(): {'used': 0, 'success': 0}
                                     for alg in self.algorithms}

        # 记录每次分配使用的算法
        self.allocation_history = []

    def allocate(self, request: Request, network: 'NSFNetWithLogging') -> Optional[Tuple[List[int], List[int]]]:
        """
        随机选择一种算法进行分配

        Args:
            request: 网络请求对象
            network: NSFNet网络实例

        Returns:
            成功时返回(路径, 频谱槽列表)，失败时返回None
        """
        # 随机选择一个算法
        selected_algorithm = np.random.choice(self.algorithms)
        selected_alg_name = selected_algorithm.get_algorithm_name()

        # 更新使用统计
        self.algorithm_usage_stats[selected_alg_name]['used'] += 1

        # 记录分配尝试
        allocation_attempt = {
            'request_id': request.id,
            'selected_algorithm': selected_alg_name,
            'success': False
        }

        try:
            # 使用选定的算法进行分配
            result = selected_algorithm.allocate(request, network)

            if result is not None:
                # 分配成功
                self.algorithm_usage_stats[selected_alg_name]['success'] += 1
                allocation_attempt['success'] = True
                self.allocation_history.append(allocation_attempt)
                return result
            else:
                # 分配失败
                self.allocation_history.append(allocation_attempt)
                return None

        except Exception as e:
            print(f"随机选择的算法 {selected_alg_name} 执行错误: {e}")
            allocation_attempt['error'] = str(e)
            self.allocation_history.append(allocation_attempt)
            return None

    def get_algorithm_usage_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取算法使用统计信息"""
        stats = {}
        for alg_name, stat in self.algorithm_usage_stats.items():
            usage_rate = stat['used'] / len(self.allocation_history) if self.allocation_history else 0.0
            success_rate = stat['success'] / stat['used'] if stat['used'] > 0 else 0.0
            stats[alg_name] = {
                'used_count': stat['used'],
                'success_count': stat['success'],
                'usage_rate': usage_rate,
                'success_rate': success_rate
            }
        return stats

    def get_allocation_summary(self) -> Dict[str, Any]:
        """获取分配总结信息"""
        total_requests = len(self.allocation_history)
        successful_requests = sum(1 for attempt in self.allocation_history if attempt['success'])

        # 统计每个算法的使用分布
        algorithm_distribution = {}
        for attempt in self.allocation_history:
            alg = attempt['selected_algorithm']
            algorithm_distribution[alg] = algorithm_distribution.get(alg, 0) + 1

        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0.0,
            'algorithm_usage_distribution': algorithm_distribution,
            'detailed_algorithm_stats': self.get_algorithm_usage_statistics()
        }


class TimeslotRequestGenerator:
    """基于时隙的请求生成器"""

    def __init__(self, total_timeslots: int = None, requests_per_timeslot: float = None,
                 random_seed: int = None, use_config: bool = True):
        """
        初始化时隙请求生成器

        Args:
            total_timeslots: 总时隙数（None时从配置读取）
            requests_per_timeslot: 每时隙平均请求数（None时从配置读取）
            random_seed: 随机种子（None时从配置读取）
            use_config: 是否使用配置文件
        """
        self.use_config = use_config

        if use_config:
            simulation_config = config_reader.get_simulation_config()
            traffic_config = config_reader.get_traffic_config()
            self.total_timeslots = total_timeslots or simulation_config['total_timeslots']
            self.requests_per_timeslot = requests_per_timeslot or traffic_config['requests_per_timeslot']
            self.random_seed = random_seed or simulation_config.get('random_seed', 42)
        else:
            self.total_timeslots = total_timeslots or 100
            self.requests_per_timeslot = requests_per_timeslot or 20.0
            self.random_seed = random_seed or 42

        self.requests = []
        self._generate_requests()

    def _generate_requests(self):
        """生成所有时隙的请求（支持动态参数变化）"""
        np.random.seed(self.random_seed)

        nodes = list(range(14))  # NSFNet节点0-13
        request_id = 0

        print(f"🎲 使用随机种子 {self.random_seed}")
        print(f"📊 生成 {self.total_timeslots} 个时隙的请求")

        if parameter_manager.is_change_enabled():
            print(f"🔄 启用参数动态变化功能")
        else:
            print(f"📊 平均每时隙 {self.requests_per_timeslot} 个请求（固定参数）")

        for timeslot in range(self.total_timeslots):
            # 获取当前时隙的动态配置
            if parameter_manager.is_change_enabled():
                req_config = parameter_manager.get_dynamic_request_generation_config(timeslot, self.total_timeslots)
                timeslot_config = parameter_manager.get_dynamic_timeslot_config(timeslot, self.total_timeslots)
                current_requests_per_timeslot = timeslot_config['requests_per_timeslot']

                # 记录参数变化
                parameter_manager.record_parameters(timeslot, self.total_timeslots)
            else:
                req_config = config_reader.get_request_generation_config()
                current_requests_per_timeslot = self.requests_per_timeslot

            # 每个时隙的请求数量
            num_requests = np.random.poisson(current_requests_per_timeslot)

            for _ in range(num_requests):
                request_id += 1

                # 生成请求参数
                source = np.random.choice(nodes)
                destination = np.random.choice([n for n in nodes if n != source])

                # 带宽需求（使用动态配置）
                if np.random.rand() < req_config['small_bandwidth_probability']:
                    bandwidth = np.random.uniform(req_config['small_bandwidth_min'], req_config['small_bandwidth_max'])
                else:
                    bandwidth = np.random.uniform(req_config['large_bandwidth_min'], req_config['large_bandwidth_max'])

                spectrum_slots = max(1, int(np.ceil(bandwidth / req_config['spectrum_slot_bandwidth'])))

                # 持续时间（使用动态配置）
                duration_mean = req_config['duration_mean']
                duration_timeslots = max(1, int(np.random.exponential(duration_mean)))

                request = TimeslotRequest(
                    id=request_id,
                    source=source,
                    destination=destination,
                    bandwidth=bandwidth,
                    spectrum_slots=spectrum_slots,
                    arrival_timeslot=timeslot,
                    duration_timeslots=duration_timeslots
                )

                self.requests.append(request)

        print(f"✅ 生成了 {len(self.requests)} 个请求")

    def get_requests(self) -> List[TimeslotRequest]:
        """获取生成的请求列表"""
        return self.requests.copy()


class NSFNetWithLogging:
    """NSFNet网络模拟器，支持频谱分配和详细数据记录（纯时隙模式）"""

    def __init__(self, num_spectrum_slots: int = None, enable_logging: bool = True,
                 log_dir: str = None, total_timeslots: int = None,
                 pre_generated_requests: List[TimeslotRequest] = None):
        """
        初始化NSFNet网络模拟器（纯时隙模式）

        Args:
            num_spectrum_slots: 每条链路的频谱槽数量
            enable_logging: 是否启用详细数据记录
            log_dir: 日志目录
            total_timeslots: 总时隙数
            pre_generated_requests: 预生成的请求列表
        """
        # 获取配置
        sim_config = config_reader.get_simulation_config()
        log_config = config_reader.get_logging_config()
        alg_config = config_reader.get_algorithm_config()

        # 使用配置值或传入参数
        self.default_capacity = sim_config['default_capacity']
        self.num_spectrum_slots = num_spectrum_slots if num_spectrum_slots is not None else sim_config['num_spectrum_slots']

        self.graph = self._create_nsfnet_topology()
        self.current_timeslot = 0
        self.active_requests = []
        self.completed_requests = []
        self.blocked_requests = []

        # 计算开销跟踪开关和变量（默认禁用以提高性能）
        self.enable_computational_overhead = False  # 计算开销记录开关
        self.current_timeslot_computational_overhead = 0.0

        # 使用配置的K值初始化默认算法
        self.allocation_algorithm = KSPFirstFitAllocation(k=alg_config['ksp_k_value'])

        # 数据记录器
        log_directory = log_dir if log_dir is not None else log_config['log_directory']
        self.data_logger = DataLogger(base_log_dir=log_directory, enable_logging=enable_logging)
        self.edge_labels = SpectrumMatrixBuilder.get_edge_labels(self.graph)

        # 时隙模式属性
        self.total_timeslots = total_timeslots if total_timeslots is not None else sim_config['total_timeslots']
        self.pre_generated_requests = pre_generated_requests or []
        self.requests_by_timeslot = self._organize_requests_by_timeslot() if pre_generated_requests else {}
        print(f"🎯 时隙模式，总时隙数: {self.total_timeslots}")

        self.stats = {
            'total_requests': 0,
            'accepted_requests': 0,
            'blocked_requests': 0,
            'completed_requests': 0,
            'blocking_probability': 0.0,
            'average_utilization': 0.0,
            'spectrum_utilization': 0.0,
            'average_hop_count': 0.0,
            'average_spectrum_slots': 0.0,
            'fragmentation_rate': 0.0
        }

    def _create_nsfnet_topology(self) -> nx.Graph:
        """创建NSFNet拓扑结构"""
        G = nx.Graph()
        nodes = [
            (0, {'name': 'Seattle', 'pos': (0, 3)}), (1, {'name': 'San Francisco', 'pos': (0, 2)}),
            (2, {'name': 'Los Angeles', 'pos': (0, 1)}), (3, {'name': 'Salt Lake City', 'pos': (1, 2.5)}),
            (4, {'name': 'Boulder', 'pos': (2, 2)}), (5, {'name': 'Houston', 'pos': (3, 0.5)}),
            (6, {'name': 'Lincoln', 'pos': (3, 2.5)}), (7, {'name': 'Champaign', 'pos': (4, 2.5)}),
            (8, {'name': 'Atlanta', 'pos': (5, 1)}), (9, {'name': 'Pittsburgh', 'pos': (5, 3)}),
            (10, {'name': 'Ann Arbor', 'pos': (4, 3.5)}), (11, {'name': 'Ithaca', 'pos': (6, 3.5)}),
            (12, {'name': 'Princeton', 'pos': (6, 2.5)}), (13, {'name': 'Washington DC', 'pos': (6, 2)})
        ]
        G.add_nodes_from(nodes)
        edges = [
            (0, 1), (0, 3), (1, 2), (1, 3), (2, 5), (3, 4), (3, 6), (4, 5), (4, 6), (5, 8),
            (6, 7), (7, 8), (7, 10), (8, 13), (9, 10), (9, 11), (9, 12), (10, 11), (11, 12),
            (12, 13), (6, 9)
        ]
        for u, v in edges:
            G.add_edge(u, v,
                       capacity=self.default_capacity,
                       used=0.0,
                       total_spectrum=self.num_spectrum_slots,
                       available_spectrum=set(range(self.num_spectrum_slots)),
                       used_spectrum=set())
        return G

    def get_edge_available_spectrum(self, u: int, v: int) -> Set[int]:
        if self.graph.has_edge(u, v):
            return self.graph[u][v]['available_spectrum']
        elif self.graph.has_edge(v, u):
            return self.graph[v][u]['available_spectrum']
        return set()

    def _organize_requests_by_timeslot(self) -> Dict[int, List[TimeslotRequest]]:
        """按时隙组织预生成的请求"""
        requests_by_timeslot = {}
        for timeslot in range(self.total_timeslots):
            requests_by_timeslot[timeslot] = []

        for req in self.pre_generated_requests:
            if req.arrival_timeslot < self.total_timeslots:
                requests_by_timeslot[req.arrival_timeslot].append(req)

        return requests_by_timeslot

    def set_pre_generated_requests(self, requests: List[TimeslotRequest]):
        """设置预生成的请求列表（时隙模式）"""
        self.pre_generated_requests = requests
        self.requests_by_timeslot = self._organize_requests_by_timeslot()
        print(f"📋 已设置 {len(requests)} 个预生成请求")



    def allocate_request(self, request: Request) -> bool:
        """分配请求并记录决策数据"""
        result = self.allocation_algorithm.allocate(request, self)
        processing_time = 0.0  # 使用时隙级别计算开销，不记录单请求时间

        return self._process_allocation_result(request, result, processing_time)



    def _process_allocation_result(self, request: Request, result, processing_time: float) -> bool:
        """处理分配结果的通用方法"""

        if result:
            path, spectrum_slots = result
            request.path = path
            request.allocated_spectrum = spectrum_slots
            request.status = "active"

            # 更新网络状态
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = self.graph[u][v]
                edge['used'] += request.bandwidth
                for slot in spectrum_slots:
                    edge['available_spectrum'].discard(slot)
                    edge['used_spectrum'].add(slot)

            self.active_requests.append(request)
            self.stats['accepted_requests'] += 1

            # 记录成功的分配决策
            decision = RequestDecision(
                request_id=request.id,
                source_node=request.source,
                target_node=request.destination,
                bandwidth_demand=request.bandwidth,
                spectrum_slots_demand=request.spectrum_slots,
                arrival_time=request.arrival_time,
                duration=request.duration,
                selected_path=path,
                allocated_spectrum_start=min(spectrum_slots),
                allocated_spectrum_end=max(spectrum_slots),
                allocation_status="success",
                failure_reason="",
                algorithm_used=self.allocation_algorithm.get_algorithm_name(),
                processing_time=processing_time
            )
            self.data_logger.log_request_decision(decision)

            return True
        else:
            request.status = "blocked"
            self.blocked_requests.append(request)
            self.stats['blocked_requests'] += 1

            # 记录失败的分配决策
            decision = RequestDecision(
                request_id=request.id,
                source_node=request.source,
                target_node=request.destination,
                bandwidth_demand=request.bandwidth,
                spectrum_slots_demand=request.spectrum_slots,
                arrival_time=request.arrival_time,
                duration=request.duration,
                selected_path=[],
                allocated_spectrum_start=-1,
                allocated_spectrum_end=-1,
                allocation_status="blocked",
                failure_reason="No available path or spectrum",
                algorithm_used=self.allocation_algorithm.get_algorithm_name(),
                processing_time=processing_time
            )
            self.data_logger.log_request_decision(decision)

            return False

    def release_request(self, request: Request):
        if request.path and request.allocated_spectrum:
            for i in range(len(request.path) - 1):
                u, v = request.path[i], request.path[i + 1]
                edge = self.graph[u][v]
                edge['used'] = max(0, edge['used'] - request.bandwidth)
                for slot in request.allocated_spectrum:
                    edge['available_spectrum'].add(slot)
                    edge['used_spectrum'].discard(slot)
        request.status = "completed"
        if request in self.active_requests:
            self.active_requests.remove(request)
        self.completed_requests.append(request)
        self.stats['completed_requests'] += 1



    def _update_stats(self):
        """更新网络统计信息"""
        # 修复：确保统计数据的一致性
        total_requests = len(self.active_requests) + len(self.completed_requests) + len(self.blocked_requests)
        accepted_requests = len(self.active_requests) + len(self.completed_requests)
        blocked_requests = len(self.blocked_requests)

        # 更新基本统计
        self.stats['total_requests'] = total_requests
        self.stats['accepted_requests'] = accepted_requests
        self.stats['blocked_requests'] = blocked_requests
        self.stats['completed_requests'] = len(self.completed_requests)

        # 计算阻塞率
        if total_requests > 0:
            self.stats['blocking_probability'] = blocked_requests / total_requests
        else:
            self.stats['blocking_probability'] = 0.0

        # 修复：计算平均边利用率而不是总带宽利用率
        edge_utilizations = []
        for _, _, data in self.graph.edges(data=True):
            if data.get('capacity', 0) > 0:
                utilization = data.get('used', 0) / data['capacity']
                edge_utilizations.append(utilization)

        if edge_utilizations:
            self.stats['average_utilization'] = sum(edge_utilizations) / len(edge_utilizations)
        else:
            self.stats['average_utilization'] = 0.0

        # 修复：计算频谱利用率
        total_spectrum_slots = sum(data.get('total_spectrum', 0) for _, _, data in self.graph.edges(data=True))
        used_spectrum_slots = sum(len(data.get('used_spectrum', [])) for _, _, data in self.graph.edges(data=True))
        if total_spectrum_slots > 0:
            self.stats['spectrum_utilization'] = used_spectrum_slots / total_spectrum_slots
        else:
            self.stats['spectrum_utilization'] = 0.0

        # 修复：计算平均跳数和频谱槽数
        if accepted_requests > 0:
            accepted_reqs = [r for r in self.active_requests + self.completed_requests if hasattr(r, 'path') and r.path]
            if accepted_reqs:
                total_hops = sum(len(r.path) - 1 for r in accepted_reqs)
                self.stats['average_hop_count'] = total_hops / len(accepted_reqs)

                total_spectrum_used = sum(len(getattr(r, 'allocated_spectrum', [])) for r in accepted_reqs)
                self.stats['average_spectrum_slots'] = total_spectrum_used / len(accepted_reqs)
            else:
                self.stats['average_hop_count'] = 0.0
                self.stats['average_spectrum_slots'] = 0.0
        else:
            self.stats['average_hop_count'] = 0.0
            self.stats['average_spectrum_slots'] = 0.0

        # 计算碎片化率
        self.stats['fragmentation_rate'] = self._calculate_fragmentation_rate()

    def _calculate_fragmentation_rate(self) -> float:
        """
        计算频谱碎片化率 - 改进版本
        考虑整个频谱的使用模式，而不仅仅是可用频谱

        Returns:
            碎片化率 (0-1之间，越高表示碎片化越严重)
        """
        try:
            total_fragmentation = 0.0
            total_edges = 0
            total_utilization = 0.0

            # 遍历所有边计算碎片化
            for u, v, data in self.graph.edges(data=True):
                total_edges += 1
                available_spectrum = data.get('available_spectrum', set())
                used_spectrum = data.get('used_spectrum', set())
                total_spectrum = data.get('total_spectrum', 0)

                if total_spectrum == 0:
                    continue

                # 计算频谱利用率
                utilization = len(used_spectrum) / total_spectrum
                total_utilization += utilization

                # 如果没有使用任何频谱，碎片化率为0
                if len(used_spectrum) == 0:
                    continue

                # 计算可用频谱的碎片化程度
                if len(available_spectrum) > 0:
                    available_list = sorted(list(available_spectrum))

                    # 计算可用频谱的连续块数量
                    available_fragments = 1
                    for i in range(1, len(available_list)):
                        if available_list[i] != available_list[i-1] + 1:
                            available_fragments += 1

                    # 计算碎片化指标：连续块数量相对于可用频谱数量的比例
                    if len(available_list) > 1:
                        edge_fragmentation = (available_fragments - 1) / (len(available_list) - 1)
                    else:
                        edge_fragmentation = 0.0

                    # 权重：利用率越高，碎片化的影响越大
                    weighted_fragmentation = edge_fragmentation * utilization
                    total_fragmentation += weighted_fragmentation

            if total_edges == 0:
                return 0.0

            # 计算平均碎片化率
            avg_fragmentation = total_fragmentation / total_edges
            avg_utilization = total_utilization / total_edges

            # 如果平均利用率很低，碎片化影响较小
            weights = config_reader.get_algorithm_weights_config()
            if avg_utilization < weights['fragmentation_low_utilization_threshold']:
                return avg_fragmentation * weights['fragmentation_low_utilization_factor']

            return min(1.0, max(0.0, avg_fragmentation))

        except Exception as e:
            print(f"计算碎片化率失败: {e}")
            return 0.0

    def _log_timeslot_data(self):
        """记录当前时隙的所有数据"""
        if not self.data_logger.enable_logging:
            return

        # 1. 记录频谱占用矩阵
        spectrum_matrix = SpectrumMatrixBuilder.build_spectrum_matrix(self.graph, self.edge_labels)
        self.data_logger.log_spectrum_matrix(spectrum_matrix, self.edge_labels)

        # 2. 记录时隙性能指标
        total_bandwidth_demand = sum(r.bandwidth for r in self.active_requests + self.completed_requests + self.blocked_requests)
        successful_bandwidth = sum(r.bandwidth for r in self.active_requests + self.completed_requests)

        metrics = TimeSlotMetrics(
            timeslot=self.data_logger.get_current_timeslot(),
            simulation_time=float(self.current_timeslot),
            total_requests=self.stats['total_requests'],
            successful_requests=self.stats['accepted_requests'],
            blocked_requests=self.stats['blocked_requests'],
            blocking_rate=self.stats['blocking_probability'],
            spectrum_utilization=self.stats['spectrum_utilization'],
            throughput=successful_bandwidth / max(total_bandwidth_demand, 1),
            fragmentation_rate=self.stats['fragmentation_rate'],
            average_delay=2.0,  # 简化的延迟计算
            average_hop_count=self.stats['average_hop_count'],
            active_requests_count=len(self.active_requests),
            total_bandwidth_demand=total_bandwidth_demand,
            successful_bandwidth=successful_bandwidth,
            computational_overhead=self.current_timeslot_computational_overhead
        )
        self.data_logger.log_performance_metrics(metrics)

        # 3. 记录网络状态
        link_utilizations = {}
        spectrum_fragments = {}

        for u, v, data in self.graph.edges(data=True):
            edge_label = f"{u}-{v}"
            utilization = data['used'] / data['capacity'] if data['capacity'] > 0 else 0
            link_utilizations[edge_label] = utilization

            # 计算频谱碎片数
            available_spectrum = data.get('available_spectrum', set())
            if available_spectrum:
                available_list = sorted(list(available_spectrum))
                fragments = 1
                for i in range(1, len(available_list)):
                    if available_list[i] != available_list[i-1] + 1:
                        fragments += 1
                spectrum_fragments[edge_label] = fragments
            else:
                spectrum_fragments[edge_label] = 0

        utilization_values = list(link_utilizations.values())
        network_state = NetworkState(
            timeslot=self.data_logger.get_current_timeslot(),
            simulation_time=float(self.current_timeslot),
            link_utilizations=link_utilizations,
            active_requests=[r.id for r in self.active_requests],
            total_network_load=sum(r.bandwidth for r in self.active_requests),
            spectrum_fragments_per_link=spectrum_fragments,
            average_link_utilization=sum(utilization_values) / len(utilization_values) if utilization_values else 0,
            max_link_utilization=max(utilization_values) if utilization_values else 0,
            min_link_utilization=min(utilization_values) if utilization_values else 0
        )
        self.data_logger.log_network_state(network_state)

    def set_allocation_algorithm(self, algorithm: AllocationAlgorithm):
        """设置分配算法并更新数据记录器"""
        self.allocation_algorithm = algorithm
        self.data_logger.set_current_algorithm(algorithm.get_algorithm_name())

    def run_simulation(self, progress_interval: Optional[int] = 10):
        """运行时隙仿真"""
        self.run_timeslot_simulation(progress_interval)

    def run_timeslot_simulation(self, progress_interval: Optional[int] = 10):
        """运行基于时隙的仿真"""
        print(f"🎯 运行时隙模式仿真，总共 {self.total_timeslots} 个时隙")
        print(f"Algorithm: {self.allocation_algorithm.get_algorithm_name()}")
        print(f"Logging enabled: {self.data_logger.enable_logging}")

        # 确保初始时隙的数据一致性
        if self.data_logger.enable_logging:
            self.data_logger.ensure_timeslot_consistency()

        for timeslot in range(self.total_timeslots):
            self.current_timeslot = timeslot
            self._process_timeslot(timeslot)

            # 显示进度
            if progress_interval is not None and progress_interval > 0 and (timeslot + 1) % progress_interval == 0:
                print(f"⏳ 已完成时隙 {timeslot}, "
                      f"总请求: {self.stats['total_requests']}, "
                      f"阻塞率: {self.stats['blocking_probability']:.4f}")

        print("Timeslot simulation completed!")

        # 生成汇总报告
        if self.data_logger.enable_logging:
            self.data_logger.create_summary_report()

    def _process_timeslot(self, timeslot: int):
        """处理单个时隙（时隙模式）"""
        # 设置数据记录器的当前时隙
        self.data_logger.current_timeslot = timeslot

        # 记录整个时隙的算法处理开始时间（如果启用计算开销记录）
        if self.enable_computational_overhead:
            timeslot_algorithm_start = time.perf_counter()
        else:
            timeslot_algorithm_start = 0.0

        # 1. 释放到期的请求
        requests_to_release = []
        for request in self.active_requests:
            if request.arrival_time + request.duration <= timeslot:
                requests_to_release.append(request)

        for request in requests_to_release:
            self.release_request(request)

        # 2. 处理新到达的请求（算法计算部分）
        if timeslot in self.requests_by_timeslot:
            for ts_req in self.requests_by_timeslot[timeslot]:
                # 转换为标准Request对象
                request = Request(
                    id=ts_req.id,
                    source=ts_req.source,
                    destination=ts_req.destination,
                    bandwidth=ts_req.bandwidth,
                    spectrum_slots=ts_req.spectrum_slots,
                    arrival_time=float(ts_req.arrival_timeslot),
                    duration=float(ts_req.duration_timeslots)
                )

                self.stats['total_requests'] += 1
                self.allocate_request(request)

        # 记录整个时隙的算法处理结束时间（如果启用计算开销记录）
        if self.enable_computational_overhead:
            timeslot_algorithm_end = time.perf_counter()
            self.current_timeslot_computational_overhead = timeslot_algorithm_end - timeslot_algorithm_start
        else:
            self.current_timeslot_computational_overhead = 0.0

        # 3. 更新统计信息
        self._update_stats()

        # 4. 记录时隙数据
        self._log_timeslot_data()
        self.data_logger.ensure_timeslot_consistency()

    def get_current_state(self) -> Dict:
        state = {
            'current_timeslot': self.current_timeslot, 'stats': self.stats.copy(), 'active_requests': len(self.active_requests),
            'link_utilization': {}, 'node_info': {}
        }
        for u, v, data in self.graph.edges(data=True):
            key = f"{u}-{v}"
            state['link_utilization'][key] = {
                'capacity': data['capacity'], 'used': data['used'],
                'utilization': data['used'] / data['capacity'] if data['capacity'] > 0 else 0
            }
        for node, data in self.graph.nodes(data=True):
            state['node_info'][node] = {
                'name': data.get('name', f'Node_{node}'),
                'active_requests': sum(1 for r in self.active_requests if r.path and node in r.path)
            }
        return state

    def set_computational_overhead_tracking(self, enabled: bool):
        """设置计算开销跟踪开关

        Args:
            enabled: True启用计算开销记录，False禁用以提高性能
        """
        self.enable_computational_overhead = enabled
        if enabled:
            print("✅ 计算开销记录已启用")
        else:
            print("⚡ 计算开销记录已禁用（提高性能）")


def ask_user_for_computational_overhead() -> bool:
    """询问用户是否启用计算开销记录"""
    print("\n" + "="*60)
    print("⚙️  计算开销记录设置")
    print("="*60)
    print("计算开销记录可以分析算法的性能，但会影响仿真速度：")
    print("  ✅ 启用：获得详细的算法性能数据（影响仿真速度10-60%）")
    print("  ⚡ 禁用：最佳仿真性能，无计算开销数据")
    print()

    while True:
        try:
            choice = input("是否启用计算开销记录？(y/n) [默认: y]: ").strip().lower()

            if choice == '' or choice == 'y' or choice == 'yes':
                print("✅ 已选择：启用计算开销记录（详细分析）")
                return True
            elif choice == 'n' or choice == 'no':
                print("⚡ 已选择：禁用计算开销记录（最佳性能）")
                return False
            else:
                print("❌ 请输入 y/yes 或 n/no")

        except KeyboardInterrupt:
            print("\n⚡ 使用默认设置：禁用计算开销记录")
            return False
        except Exception as e:
            print(f"❌ 输入错误: {e}")
            print("⚡ 使用默认设置：禁用计算开销记录")
            return False


def run_timeslot_based_test():
    """运行基于时隙的一致性测试"""
    print("=== NSFNet 时隙模式算法一致性测试 ===")

    # 获取配置
    test_config = config_reader.get_test_parameters_config()
    simulation_config = config_reader.get_simulation_config()
    traffic_config = config_reader.get_traffic_config()
    num_spectrum_slots = test_config['test_num_spectrum_slots']

    # 使用仿真和流量配置
    total_timeslots = simulation_config['total_timeslots']
    requests_per_timeslot = traffic_config['requests_per_timeslot']
    random_seed = simulation_config.get('random_seed', 42)  # 使用默认值42

    alg_config = config_reader.get_algorithm_config()
    ksp_k = alg_config['ksp_k_value']
    random_k = alg_config['random_fit_k_value']

    # 1. 生成统一的请求序列
    print(f"\n🎯 第1步：生成统一请求序列")
    print(f"📊 总时隙数: {total_timeslots}")
    print(f"📊 每时隙平均请求数: {requests_per_timeslot}")
    print(f"🎲 随机种子: {random_seed}")

    request_generator = TimeslotRequestGenerator(use_config=True)
    pre_generated_requests = request_generator.get_requests()

    # 2. 定义算法
    algorithms = [
        ("KSP_FA", KSPFragmentationAwareAllocation(k=ksp_k)),
        ("KSP_FF", KSPFirstFitAllocation(k=ksp_k)),
        ("Random_Fit", KSPRandomFitAllocation()),
        ("FA_CA", FACongestionAwareAllocation()),
        ("KSP_LB", LoadBalancedAllocation(k=ksp_k)),
        ("Custom_Algorithm", ExhaustiveOptimalAllocation(k=ksp_k)),
        ("Custom_Algorithm_Random", CustomAlgorithmRandom(k=ksp_k))
    ]

    print(f"\n🔬 将测试 {len(algorithms)} 种频谱分配算法:")
    for i, (name, _) in enumerate(algorithms, 1):
        print(f"  {i}. {name}")

    # 询问用户是否启用计算开销记录
    enable_overhead = ask_user_for_computational_overhead()

    # 3. 为每个算法运行仿真
    algorithm_results = {}

    for i, (alg_name, algorithm) in enumerate(algorithms, 1):
        print(f"\n{'='*60}")
        print(f"🧪 测试算法 {i}/{len(algorithms)}: {alg_name}")
        print(f"{'='*60}")

        # 创建网络实例（时隙模式）
        nsfnet = NSFNetWithLogging(
            num_spectrum_slots=num_spectrum_slots,
            enable_logging=True,
            total_timeslots=total_timeslots,
            pre_generated_requests=pre_generated_requests
        )

        # 设置算法
        nsfnet.set_allocation_algorithm(algorithm)

        # 根据用户选择设置计算开销记录
        nsfnet.set_computational_overhead_tracking(enable_overhead)

        # 运行时隙仿真
        print(f"⏳ 运行时隙仿真 (每时隙平均请求={requests_per_timeslot}, 总时隙={total_timeslots})...")
        nsfnet.run_simulation()

        # 收集结果
        stats = nsfnet.stats
        algorithm_results[alg_name] = {
            'total_requests': stats['total_requests'],
            'accepted_requests': stats['accepted_requests'],
            'blocked_requests': stats['blocked_requests'],
            'blocking_probability': stats['blocking_probability'],
            'spectrum_utilization': stats['spectrum_utilization'],
            'fragmentation_rate': stats['fragmentation_rate']
        }

        print(f"✅ {alg_name} 完成 - 阻塞率: {stats['blocking_probability']:.4f}")

    # 保存参数变化历史（如果启用了参数变化）
    if parameter_manager.is_change_enabled():
        parameter_manager.save_parameter_history()
        print(f"\n📊 参数变化功能已启用，变化历史已保存")

    return algorithm_results, len(pre_generated_requests), total_timeslots





if __name__ == "__main__":
    # 运行时隙模式测试
    algorithm_results, total_requests, total_timeslots = run_timeslot_based_test()

    # 显示结果对比
    print(f"\n📊 时隙模式算法性能对比:")
    print("=" * 80)
    print(f"{'算法':<15} {'总请求':<8} {'成功':<8} {'阻塞':<8} {'阻塞率':<10} {'频谱利用率':<12}")
    print("-" * 80)

    for alg_name, results in algorithm_results.items():
        print(f"{alg_name:<15} {results['total_requests']:<8} "
              f"{results['accepted_requests']:<8} {results['blocked_requests']:<8} "
              f"{results['blocking_probability']:<10.4f} {results['spectrum_utilization']:<12.4f}")

    # 找出最佳算法
    best_blocking = min(algorithm_results.items(), key=lambda x: x[1]['blocking_probability'])
    best_utilization = max(algorithm_results.items(), key=lambda x: x[1]['spectrum_utilization'])

    print(f"\n🏆 最佳性能算法:")
    print(f"  最低阻塞率: {best_blocking[0]} ({best_blocking[1]['blocking_probability']:.4f})")
    print(f"  最高频谱利用率: {best_utilization[0]} ({best_utilization[1]['spectrum_utilization']:.4f})")

    print(f"\n✅ 时隙模式测试完成!")
    print(f"📊 完整时隙数: {total_timeslots} 个时隙 (时隙0 到 时隙{total_timeslots-1})")
    print(f"📋 所有算法处理相同的 {total_requests} 个请求")
    print(f"🎯 请求序列完全一致，算法比较具有科学性")

    print("\n📁 记录的数据文件结构:")
    print("log/")
    print("├── spectrum_state/        # 频谱占用矩阵")
    print("├── request_decisions/     # 请求分配决策")
    print("├── performance_metrics/   # 性能指标")
    print("├── network_state/         # 网络状态")
    print("└── summary_report.json    # 汇总报告")

    print(f"\n✅ 所有算法的数据已保存到 'log' 目录中")
    print(f"💡 使用 data_analysis.py 算法性能制图")
    print(f"💡 使用 parameter_analyze.py 参数变化制图")