#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件读取器
读取和管理仿真配置参数
"""

import configparser
import os
from typing import Dict, Any


class ConfigReader:
    """配置文件读取器"""
    
    def __init__(self, config_file: str = "config.ini"):
        """
        初始化配置读取器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file, encoding='utf-8')
            # 只在调试模式下打印加载信息，避免重复输出
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"配置文件已加载: {self.config_file}")
        else:
            print(f"配置文件不存在: {self.config_file}，使用默认配置")
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        self.config['simulation'] = {
            'lambda_rate': '1.0',
            'simulation_duration': '30.0',
            'time_step': '0.1',
            'num_spectrum_slots': '320',
            'default_capacity': '100.0'
        }

        self.config['logging'] = {
            'enable_logging': 'true',
            'log_directory': r'E:\PythonProject\KSP\log',
            'timeslot_duration': '1.0',
            'log_request_decisions': 'true',
            'log_spectrum_matrix': 'true',
            'log_performance_metrics': 'true',
            'log_network_state': 'true',
            'default_spectrum_slots_for_empty_files': '320'
        }

        self.config['algorithms'] = {
            'ksp_k_value': '3',
            'random_fit_k_value': '5',
            'random_fit_max_paths': '10',
            'default_path_weight': '1'
        }

        self.config['output'] = {
            'backup_directory': r'E:\PythonProject\KSP\log',
            'enable_visualization': 'true',
            'save_plots': 'true',
            'plot_dpi': '300'
        }

        self.config['performance'] = {
            'progress_interval': '10',
            'memory_optimization': 'false',
            'async_logging': 'false'
        }

        self.config['request_generation'] = {
            'small_bandwidth_probability': '0.8',
            'small_bandwidth_min': '5.0',
            'small_bandwidth_max': '25.0',
            'large_bandwidth_min': '80.0',
            'large_bandwidth_max': '120.0',
            'spectrum_slot_bandwidth': '12.5',
            'duration_mean': '1.0'
        }

        self.config['time_management'] = {
            'default_time_step': '0.1',
            'default_timeslot_duration': '1.0'
        }

        self.config['algorithm_weights'] = {
            'fa_ca_spectrum_weight': '0.4',
            'fa_ca_load_weight': '0.3',
            'fa_ca_fragmentation_weight': '0.3',
            'ksp_lb_std_weight': '0.7',
            'ksp_lb_mean_weight': '0.3',
            'ksp_fa_block_count_weight': '0.7',
            'ksp_fa_size_uniformity_weight': '0.3',
            'edge_weight_offset': '0.01',
            'fragmentation_low_utilization_threshold': '0.1',
            'fragmentation_low_utilization_factor': '0.5'
        }

        self.config['test_parameters'] = {
            'test_lambda_rate': '8.0',
            'test_simulation_duration': '10.0',
            'test_num_spectrum_slots': '320'
        }

        self.config['parameter_change'] = {
            'test_changePara': 'false',
            'change_mode': 'sine_wave',
            'change_period': '50',
            'change_amplitude': '0.3',
            'change_small_bandwidth_probability': 'true',
            'change_duration_mean': 'true',
            'change_bandwidth_ranges': 'true',
            'change_requests_per_timeslot': 'true',
            'change_spectrum_slots': 'true',
            'phase_offset': '0.0',
            'min_value_ratio': '0.5',
            'max_value_ratio': '2.0'
        }
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """获取仿真配置"""
        return {
            'lambda_rate': self.config.getfloat('simulation', 'lambda_rate'),
            'simulation_duration': self.config.getfloat('simulation', 'simulation_duration'),
            'time_step': self.config.getfloat('simulation', 'time_step'),
            'num_spectrum_slots': self.config.getint('simulation', 'num_spectrum_slots'),
            'default_capacity': self.config.getfloat('simulation', 'default_capacity')
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return {
            'enable_logging': self.config.getboolean('logging', 'enable_logging'),
            'log_directory': self.config.get('logging', 'log_directory'),
            'timeslot_duration': self.config.getfloat('logging', 'timeslot_duration'),
            'log_request_decisions': self.config.getboolean('logging', 'log_request_decisions'),
            'log_spectrum_matrix': self.config.getboolean('logging', 'log_spectrum_matrix'),
            'log_performance_metrics': self.config.getboolean('logging', 'log_performance_metrics'),
            'log_network_state': self.config.getboolean('logging', 'log_network_state'),
            'default_spectrum_slots_for_empty_files': self.config.getint('logging', 'default_spectrum_slots_for_empty_files')
        }
    
    def get_algorithm_config(self) -> Dict[str, Any]:
        """获取算法配置"""
        return {
            'ksp_k_value': self.config.getint('algorithms', 'ksp_k_value'),
            'random_fit_k_value': self.config.getint('algorithms', 'random_fit_k_value'),
            'random_fit_max_paths': self.config.getint('algorithms', 'random_fit_max_paths'),
            'default_path_weight': self.config.getint('algorithms', 'default_path_weight')
        }
    
    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return {
            'backup_directory': self.config.get('output', 'backup_directory'),
            'enable_visualization': self.config.getboolean('output', 'enable_visualization'),
            'save_plots': self.config.getboolean('output', 'save_plots'),
            'plot_dpi': self.config.getint('output', 'plot_dpi')
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        return {
            'progress_interval': self.config.getint('performance', 'progress_interval'),
            'memory_optimization': self.config.getboolean('performance', 'memory_optimization'),
            'async_logging': self.config.getboolean('performance', 'async_logging')
        }

    def get_request_generation_config(self) -> Dict[str, Any]:
        """获取请求生成配置"""
        return {
            'small_bandwidth_probability': self.config.getfloat('request_generation', 'small_bandwidth_probability'),
            'small_bandwidth_min': self.config.getfloat('request_generation', 'small_bandwidth_min'),
            'small_bandwidth_max': self.config.getfloat('request_generation', 'small_bandwidth_max'),
            'large_bandwidth_min': self.config.getfloat('request_generation', 'large_bandwidth_min'),
            'large_bandwidth_max': self.config.getfloat('request_generation', 'large_bandwidth_max'),
            'spectrum_slot_bandwidth': self.config.getfloat('request_generation', 'spectrum_slot_bandwidth'),
            'duration_mean': self.config.getfloat('request_generation', 'duration_mean')
        }

    def get_time_management_config(self) -> Dict[str, Any]:
        """获取时间管理配置"""
        return {
            'default_time_step': self.config.getfloat('time_management', 'default_time_step'),
            'default_timeslot_duration': self.config.getfloat('time_management', 'default_timeslot_duration')
        }

    def get_algorithm_weights_config(self) -> Dict[str, Any]:
        """获取算法权重配置"""
        return {
            'fa_ca_spectrum_weight': self.config.getfloat('algorithm_weights', 'fa_ca_spectrum_weight'),
            'fa_ca_load_weight': self.config.getfloat('algorithm_weights', 'fa_ca_load_weight'),
            'fa_ca_fragmentation_weight': self.config.getfloat('algorithm_weights', 'fa_ca_fragmentation_weight'),
            'ksp_lb_std_weight': self.config.getfloat('algorithm_weights', 'ksp_lb_std_weight'),
            'ksp_lb_mean_weight': self.config.getfloat('algorithm_weights', 'ksp_lb_mean_weight'),
            'ksp_fa_block_count_weight': self.config.getfloat('algorithm_weights', 'ksp_fa_block_count_weight'),
            'ksp_fa_size_uniformity_weight': self.config.getfloat('algorithm_weights', 'ksp_fa_size_uniformity_weight'),
            'edge_weight_offset': self.config.getfloat('algorithm_weights', 'edge_weight_offset'),
            'fragmentation_low_utilization_threshold': self.config.getfloat('algorithm_weights', 'fragmentation_low_utilization_threshold'),
            'fragmentation_low_utilization_factor': self.config.getfloat('algorithm_weights', 'fragmentation_low_utilization_factor')
        }

    def get_test_parameters_config(self) -> Dict[str, Any]:
        """获取测试参数配置（纯时隙模式）"""
        return {
            'test_num_spectrum_slots': self.config.getint('test_parameters', 'test_num_spectrum_slots')
        }

    def get_timeslot_mode_config(self) -> Dict[str, Any]:
        """获取时隙模式专用配置"""
        return {
            'total_timeslots': self.config.getint('timeslot_mode', 'total_timeslots'),
            'requests_per_timeslot': self.config.getfloat('timeslot_mode', 'requests_per_timeslot'),
            'duration_timeslots_mean': self.config.getfloat('timeslot_mode', 'duration_timeslots_mean'),
            'random_seed': self.config.getint('timeslot_mode', 'random_seed')
        }

    def get_parameter_change_config(self) -> Dict[str, Any]:
        """获取参数变化配置"""
        return {
            'test_changePara': self.config.getboolean('parameter_change', 'test_changePara'),
            'change_mode': self.config.get('parameter_change', 'change_mode'),
            'change_period': self.config.getint('parameter_change', 'change_period'),
            'change_amplitude': self.config.getfloat('parameter_change', 'change_amplitude'),
            'change_small_bandwidth_probability': self.config.getboolean('parameter_change', 'change_small_bandwidth_probability'),
            'change_duration_mean': self.config.getboolean('parameter_change', 'change_duration_mean'),
            'change_bandwidth_ranges': self.config.getboolean('parameter_change', 'change_bandwidth_ranges'),
            'change_requests_per_timeslot': self.config.getboolean('parameter_change', 'change_requests_per_timeslot'),
            'change_spectrum_slots': self.config.getboolean('parameter_change', 'change_spectrum_slots'),
            'phase_offset': self.config.getfloat('parameter_change', 'phase_offset'),
            'min_value_ratio': self.config.getfloat('parameter_change', 'min_value_ratio'),
            'max_value_ratio': self.config.getfloat('parameter_change', 'max_value_ratio'),
            'bandwidth_probability_min': self.config.getfloat('parameter_change', 'bandwidth_probability_min'),
            'bandwidth_probability_max': self.config.getfloat('parameter_change', 'bandwidth_probability_max'),
            'requests_per_timeslot_min': self.config.getfloat('parameter_change', 'requests_per_timeslot_min'),
            'requests_per_timeslot_max': self.config.getfloat('parameter_change', 'requests_per_timeslot_max'),
            'spectrum_slots_min': self.config.getint('parameter_change', 'spectrum_slots_min'),
            'spectrum_slots_max': self.config.getint('parameter_change', 'spectrum_slots_max')
        }

    def get_visualization_config(self) -> Dict[str, Any]:
        """获取可视化配置"""
        return {
            'figure_width': self.config.getint('visualization', 'figure_width'),
            'figure_height': self.config.getint('visualization', 'figure_height'),
            'spring_layout_k': self.config.getfloat('visualization', 'spring_layout_k'),
            'spring_layout_iterations': self.config.getint('visualization', 'spring_layout_iterations'),
            'node_size': self.config.getint('visualization', 'node_size'),
            'node_color': self.config.get('visualization', 'node_color'),
            'node_alpha': self.config.getfloat('visualization', 'node_alpha'),
            'edge_width': self.config.getfloat('visualization', 'edge_width'),
            'edge_color': self.config.get('visualization', 'edge_color'),
            'edge_alpha': self.config.getfloat('visualization', 'edge_alpha'),
            'font_size': self.config.getint('visualization', 'font_size'),
            'title_font_size': self.config.getint('visualization', 'title_font_size')
        }

    def get_all_config(self) -> Dict[str, Dict[str, Any]]:
        """获取所有配置"""
        return {
            'simulation': self.get_simulation_config(),
            'logging': self.get_logging_config(),
            'algorithms': self.get_algorithm_config(),
            'output': self.get_output_config(),
            'performance': self.get_performance_config(),
            'request_generation': self.get_request_generation_config(),
            'time_management': self.get_time_management_config(),
            'algorithm_weights': self.get_algorithm_weights_config(),
            'test_parameters': self.get_test_parameters_config(),
            'timeslot_mode': self.get_timeslot_mode_config(),
            'parameter_change': self.get_parameter_change_config(),
            'visualization': self.get_visualization_config()
        }
    
    def save_config(self, config_file: str = None):
        """保存配置到文件"""
        if config_file is None:
            config_file = self.config_file
        
        with open(config_file, 'w', encoding='utf-8') as f:
            self.config.write(f)
        
        print(f"配置已保存到: {config_file}")
    
    def update_config(self, section: str, key: str, value: str):
        """更新配置项"""
        if section not in self.config:
            self.config.add_section(section)
        
        self.config.set(section, key, str(value))
        print(f"配置已更新: [{section}] {key} = {value}")
    
    def print_config(self):
        """打印当前配置"""
        print("\n=== 当前配置 ===")
        for section_name in self.config.sections():
            print(f"\n[{section_name}]")
            for key, value in self.config.items(section_name):
                print(f"{key} = {value}")


# 为了性能优化，提供一个使用单例模式的ConfigReader
def create_optimized_config_reader():
    """创建优化的配置读取器（使用单例模式）"""
    try:
        from config_singleton import OptimizedConfigReader
        return OptimizedConfigReader()
    except ImportError:
        # 如果单例模块不可用，回退到原始实现
        return ConfigReader()


# 重写ConfigReader类以使用单例模式（可选优化）
class OriginalConfigReader(ConfigReader):
    """原始的配置读取器实现"""
    pass


# 默认使用优化版本
try:
    from config_singleton import OptimizedConfigReader
    ConfigReader = OptimizedConfigReader
except ImportError:
    # 如果单例模块不可用，继续使用原始实现
    pass


if __name__ == "__main__":
    # 测试配置读取器
    config_reader = ConfigReader()
    config_reader._debug_mode = True  # 启用调试模式

    print("\n=== 配置测试 ===")
    sim_config = config_reader.get_simulation_config()
    print(f"仿真配置: {sim_config}")

    log_config = config_reader.get_logging_config()
    print(f"日志配置: {log_config}")

    alg_config = config_reader.get_algorithm_config()
    print(f"算法配置: {alg_config}")
