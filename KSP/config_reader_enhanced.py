#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的配置文件读取器
支持层次化配置结构和智能值解析，同时保持向后兼容性
"""

import configparser
import os
import ast
import re
from typing import Dict, Any, List, Union, Optional


class EnhancedConfigReader:
    """增强的配置文件读取器"""
    
    def __init__(self, config_file: str = "config_optimized.ini", fallback_file: str = "config.ini"):
        """
        初始化增强配置读取器
        
        Args:
            config_file: 主配置文件路径
            fallback_file: 回退配置文件路径（向后兼容）
        """
        self.config_file = config_file
        self.fallback_file = fallback_file
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.legacy_config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """加载配置文件，支持回退机制"""
        # 尝试加载优化后的配置文件
        if os.path.exists(self.config_file):
            self.config.read(self.config_file, encoding='utf-8')
            print(f"优化配置文件已加载: {self.config_file}")
        else:
            print(f"优化配置文件不存在: {self.config_file}")
        
        # 加载原始配置文件作为回退
        if os.path.exists(self.fallback_file):
            self.legacy_config.read(self.fallback_file, encoding='utf-8')
            print(f"回退配置文件已加载: {self.fallback_file}")
        else:
            print(f"回退配置文件不存在: {self.fallback_file}，使用默认配置")
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        self.config['defaults'] = {
            'spectrum_slots': '100',
            'capacity': '100.0',
            'time_step': '0.1',
            'timeslot_duration': '1.0',
            'random_seed': '42',
            'k_value': '3',
            'weight': '1',
            'probability': '1.0',
            'bandwidth_unit': '10'
        }
        
        self.config['simulation'] = {
            'mode': 'timeslot',
            'total_timeslots': '100',
            'lambda_rate': '10.0',
            'simulation_duration': '100.0'
        }
    
    def get_value(self, section: str, key: str, value_type: type = str, default: Any = None) -> Any:
        """
        智能获取配置值，支持类型转换和回退机制
        
        Args:
            section: 配置段名
            key: 配置键名
            value_type: 期望的值类型
            default: 默认值
            
        Returns:
            解析后的配置值
        """
        # 支持层次化键名（如 fa_ca.spectrum_weight）
        if '.' in key:
            section_key, sub_key = key.split('.', 1)
            full_key = f"{section_key}_{sub_key}"
        else:
            full_key = key
        
        # 首先尝试从优化配置中获取
        try:
            if self.config.has_option(section, key):
                raw_value = self.config.get(section, key)
            elif self.config.has_option(section, full_key):
                raw_value = self.config.get(section, full_key)
            else:
                raise configparser.NoOptionError(key, section)
        except (configparser.NoSectionError, configparser.NoOptionError):
            # 回退到原始配置
            try:
                if self.legacy_config.has_option(section, key):
                    raw_value = self.legacy_config.get(section, key)
                elif self.legacy_config.has_option(section, full_key):
                    raw_value = self.legacy_config.get(section, full_key)
                else:
                    if default is not None:
                        return default
                    raise configparser.NoOptionError(key, section)
            except (configparser.NoSectionError, configparser.NoOptionError):
                if default is not None:
                    return default
                raise
        
        return self._parse_value(raw_value, value_type)
    
    def _parse_value(self, raw_value: str, value_type: type) -> Any:
        """智能解析配置值"""
        if value_type == bool:
            return raw_value.lower() in ('true', '1', 'yes', 'on')
        elif value_type == int:
            return int(float(raw_value))  # 支持浮点数转整数
        elif value_type == float:
            return float(raw_value)
        elif value_type == list:
            # 支持逗号分隔的列表
            if ',' in raw_value:
                return [item.strip() for item in raw_value.split(',')]
            else:
                try:
                    return ast.literal_eval(raw_value)
                except:
                    return [raw_value]
        elif value_type == dict:
            try:
                return ast.literal_eval(raw_value)
            except:
                return {}
        else:
            return raw_value
    
    def get_section_dict(self, section: str, prefix: str = None) -> Dict[str, Any]:
        """
        获取整个配置段作为字典
        
        Args:
            section: 配置段名
            prefix: 键名前缀过滤
            
        Returns:
            配置字典
        """
        result = {}
        
        # 从优化配置获取
        if self.config.has_section(section):
            for key, value in self.config.items(section):
                if prefix is None or key.startswith(prefix):
                    result[key] = value
        
        # 从回退配置补充
        if self.legacy_config.has_section(section):
            for key, value in self.legacy_config.items(section):
                if (prefix is None or key.startswith(prefix)) and key not in result:
                    result[key] = value
        
        return result
    
    # 保持向后兼容的接口方法
    def get_simulation_config(self) -> Dict[str, Any]:
        """获取仿真配置（向后兼容）"""
        return {
            'lambda_rate': self.get_value('simulation', 'lambda_rate', float, 10.0),
            'simulation_duration': self.get_value('simulation', 'simulation_duration', float, 100.0),
            'time_step': self.get_value('simulation', 'time_step', float, 0.1),
            'num_spectrum_slots': self.get_value('network', 'num_spectrum_slots', int, 
                                               self.get_value('simulation', 'num_spectrum_slots', int, 100)),
            'default_capacity': self.get_value('network', 'default_capacity', float,
                                             self.get_value('simulation', 'default_capacity', float, 100.0)),
            'total_timeslots': self.get_value('simulation', 'total_timeslots', int, 100),
            'mode': self.get_value('simulation', 'mode', str, 'timeslot')
        }
    
    def get_traffic_config(self) -> Dict[str, Any]:
        """获取流量配置（合并request_generation和timeslot_mode）"""
        return {
            'requests_per_timeslot': self.get_value('traffic', 'requests_per_timeslot', float,
                                                   self.get_value('timeslot_mode', 'requests_per_timeslot', float, 20.0)),
            'small_bandwidth_probability': self.get_value('traffic', 'small_bandwidth_probability', float,
                                                         self.get_value('request_generation', 'small_bandwidth_probability', float, 1.0)),
            'small_bandwidth_min': self.get_value('traffic', 'small_bandwidth_min', float,
                                                 self.get_value('request_generation', 'small_bandwidth_min', float, 5.0)),
            'small_bandwidth_max': self.get_value('traffic', 'small_bandwidth_max', float,
                                                 self.get_value('request_generation', 'small_bandwidth_max', float, 25.0)),
            'large_bandwidth_min': self.get_value('traffic', 'large_bandwidth_min', float,
                                                 self.get_value('request_generation', 'large_bandwidth_min', float, 80.0)),
            'large_bandwidth_max': self.get_value('traffic', 'large_bandwidth_max', float,
                                                 self.get_value('request_generation', 'large_bandwidth_max', float, 120.0)),
            'spectrum_slot_bandwidth': self.get_value('traffic', 'spectrum_slot_bandwidth', int,
                                                     self.get_value('request_generation', 'spectrum_slot_bandwidth', int, 10)),
            'duration_mean': self.get_value('traffic', 'duration_mean', float,
                                           self.get_value('request_generation', 'duration_mean', float, 2.0)),
            'duration_timeslots_mean': self.get_value('traffic', 'duration_timeslots_mean', float,
                                                     self.get_value('timeslot_mode', 'duration_timeslots_mean', float, 2.0)),
            'random_seed': self.get_value('simulation', 'random_seed', int,
                                         self.get_value('timeslot_mode', 'random_seed', int, 42))
        }
    
    def get_algorithm_config(self) -> Dict[str, Any]:
        """获取算法配置（向后兼容）"""
        return {
            'ksp_k_value': self.get_value('algorithms', 'ksp_k_value', int, 3),
            'random_fit_k_value': self.get_value('algorithms', 'random_fit_k_value', int, 5),
            'random_fit_max_paths': self.get_value('algorithms', 'random_fit_max_paths', int, 10),
            'default_path_weight': self.get_value('algorithms', 'default_path_weight', int, 1),
            # 算法权重参数
            'fa_ca_spectrum_weight': self.get_value('algorithms', 'fa_ca.spectrum_weight', float,
                                                   self.get_value('algorithm_weights', 'fa_ca_spectrum_weight', float, 0.4)),
            'fa_ca_load_weight': self.get_value('algorithms', 'fa_ca.load_weight', float,
                                               self.get_value('algorithm_weights', 'fa_ca_load_weight', float, 0.3)),
            'fa_ca_fragmentation_weight': self.get_value('algorithms', 'fa_ca.fragmentation_weight', float,
                                                        self.get_value('algorithm_weights', 'fa_ca_fragmentation_weight', float, 0.3)),
            'ksp_lb_std_weight': self.get_value('algorithms', 'ksp_lb.std_weight', float,
                                               self.get_value('algorithm_weights', 'ksp_lb_std_weight', float, 0.7)),
            'ksp_lb_mean_weight': self.get_value('algorithms', 'ksp_lb.mean_weight', float,
                                                self.get_value('algorithm_weights', 'ksp_lb_mean_weight', float, 0.3)),
            'ksp_fa_block_count_weight': self.get_value('algorithms', 'ksp_fa.block_count_weight', float,
                                                       self.get_value('algorithm_weights', 'ksp_fa_block_count_weight', float, 0.7)),
            'ksp_fa_size_uniformity_weight': self.get_value('algorithms', 'ksp_fa.size_uniformity_weight', float,
                                                           self.get_value('algorithm_weights', 'ksp_fa_size_uniformity_weight', float, 0.3)),
            'edge_weight_offset': self.get_value('algorithms', 'edge_weight_offset', float,
                                                self.get_value('algorithm_weights', 'edge_weight_offset', float, 0.01)),
            'fragmentation_low_utilization_threshold': self.get_value('algorithms', 'fragmentation_low_utilization_threshold', float,
                                                                     self.get_value('algorithm_weights', 'fragmentation_low_utilization_threshold', float, 0.1)),
            'fragmentation_low_utilization_factor': self.get_value('algorithms', 'fragmentation_low_utilization_factor', float,
                                                                  self.get_value('algorithm_weights', 'fragmentation_low_utilization_factor', float, 0.5))
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置（向后兼容）"""
        return {
            'enable_logging': self.get_value('logging', 'enable_logging', bool, True),
            'log_directory': self.get_value('logging', 'log_directory', str, r'E:\PythonProject\KSP\log'),
            'timeslot_duration': self.get_value('logging', 'timeslot_duration', float, 1.0),
            'log_request_decisions': self.get_value('logging', 'log_request_decisions', bool, True),
            'log_spectrum_matrix': self.get_value('logging', 'log_spectrum_matrix', bool, True),
            'log_performance_metrics': self.get_value('logging', 'log_performance_metrics', bool, True),
            'log_network_state': self.get_value('logging', 'log_network_state', bool, True),
            'default_spectrum_slots_for_empty_files': self.get_value('logging', 'default_spectrum_slots_for_empty_files', int, 320),
            'backup_directory': self.get_value('logging', 'backup_directory', str,
                                              self.get_value('output', 'backup_directory', str, r'E:\PythonProject\KSP\log'))
        }

    def get_visualization_config(self) -> Dict[str, Any]:
        """获取可视化配置（向后兼容）"""
        return {
            'enable_visualization': self.get_value('visualization', 'enable_visualization', bool,
                                                  self.get_value('output', 'enable_visualization', bool, True)),
            'save_plots': self.get_value('visualization', 'save_plots', bool,
                                        self.get_value('output', 'save_plots', bool, True)),
            'plot_dpi': self.get_value('visualization', 'plot_dpi', int,
                                      self.get_value('output', 'plot_dpi', int, 300)),
            'figure_width': self.get_value('visualization', 'figure_width', int, 12),
            'figure_height': self.get_value('visualization', 'figure_height', int, 10),
            'spring_layout_k': self.get_value('visualization', 'spring_layout_k', float, 0.3),
            'spring_layout_iterations': self.get_value('visualization', 'spring_layout_iterations', int, 50),
            'node_size': self.get_value('visualization', 'node_size', int, 800),
            'node_color': self.get_value('visualization', 'node_color', str, 'lightblue'),
            'node_alpha': self.get_value('visualization', 'node_alpha', float, 0.9),
            'edge_width': self.get_value('visualization', 'edge_width', float, 1.5),
            'edge_color': self.get_value('visualization', 'edge_color', str, 'gray'),
            'edge_alpha': self.get_value('visualization', 'edge_alpha', float, 0.7),
            'font_size': self.get_value('visualization', 'font_size', int, 12),
            'title_font_size': self.get_value('visualization', 'title_font_size', int, 15)
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置（向后兼容）"""
        return {
            'progress_interval': self.get_value('performance', 'progress_interval', int, 10),
            'memory_optimization': self.get_value('performance', 'memory_optimization', bool, False),
            'async_logging': self.get_value('performance', 'async_logging', bool, False)
        }

    def get_dynamics_config(self) -> Dict[str, Any]:
        """获取动态参数变化配置（合并parameter_change）"""
        return {
            'test_changePara': self.get_value('dynamics', 'enable_parameter_change', bool,
                                             self.get_value('parameter_change', 'test_changePara', bool, True)),
            'change_mode': self.get_value('dynamics', 'change_mode', str,
                                         self.get_value('parameter_change', 'change_mode', str, 'triangle_wave')),
            'change_period': self.get_value('dynamics', 'change_period', int,
                                           self.get_value('parameter_change', 'change_period', int, 50)),
            'change_amplitude': self.get_value('dynamics', 'change_amplitude', float,
                                              self.get_value('parameter_change', 'change_amplitude', float, 0.3)),
            'change_small_bandwidth_probability': self.get_value('dynamics', 'change_small_bandwidth_probability', bool,
                                                                self.get_value('parameter_change', 'change_small_bandwidth_probability', bool, False)),
            'change_duration_mean': self.get_value('dynamics', 'change_duration_mean', bool,
                                                  self.get_value('parameter_change', 'change_duration_mean', bool, False)),
            'change_bandwidth_ranges': self.get_value('dynamics', 'change_bandwidth_ranges', bool,
                                                     self.get_value('parameter_change', 'change_bandwidth_ranges', bool, False)),
            'change_requests_per_timeslot': self.get_value('dynamics', 'change_requests_per_timeslot', bool,
                                                          self.get_value('parameter_change', 'change_requests_per_timeslot', bool, True)),
            'change_spectrum_slots': self.get_value('dynamics', 'change_spectrum_slots', bool,
                                                   self.get_value('parameter_change', 'change_spectrum_slots', bool, False)),
            'phase_offset': self.get_value('dynamics', 'phase_offset', float,
                                          self.get_value('parameter_change', 'phase_offset', float, 0.0)),
            'min_value_ratio': self.get_value('dynamics', 'min_value_ratio', float,
                                             self.get_value('parameter_change', 'min_value_ratio', float, 0.5)),
            'max_value_ratio': self.get_value('dynamics', 'max_value_ratio', float,
                                             self.get_value('parameter_change', 'max_value_ratio', float, 2.0)),
            'bandwidth_probability_min': self.get_value('dynamics', 'bandwidth_probability_min', float,
                                                       self.get_value('parameter_change', 'bandwidth_probability_min', float, 0.1)),
            'bandwidth_probability_max': self.get_value('dynamics', 'bandwidth_probability_max', float,
                                                       self.get_value('parameter_change', 'bandwidth_probability_max', float, 0.9)),
            'requests_per_timeslot_min': self.get_value('dynamics', 'requests_per_timeslot_min', float,
                                                       self.get_value('parameter_change', 'requests_per_timeslot_min', float, 1.0)),
            'requests_per_timeslot_max': self.get_value('dynamics', 'requests_per_timeslot_max', float,
                                                       self.get_value('parameter_change', 'requests_per_timeslot_max', float, 50.0)),
            'spectrum_slots_min': self.get_value('dynamics', 'spectrum_slots_min', int,
                                                self.get_value('parameter_change', 'spectrum_slots_min', int, 50)),
            'spectrum_slots_max': self.get_value('dynamics', 'spectrum_slots_max', int,
                                                self.get_value('parameter_change', 'spectrum_slots_max', int, 200))
        }

    def get_test_parameters_config(self) -> Dict[str, Any]:
        """获取测试参数配置（向后兼容）"""
        return {
            'test_num_spectrum_slots': self.get_value('test_parameters', 'test_num_spectrum_slots', int, 100)
        }

    def get_time_management_config(self) -> Dict[str, Any]:
        """获取时间管理配置（向后兼容）"""
        return {
            'default_time_step': self.get_value('time_management', 'default_time_step', float, 0.1),
            'default_timeslot_duration': self.get_value('time_management', 'default_timeslot_duration', float, 1.0)
        }

    # 新增的便捷方法
    def get_network_config(self) -> Dict[str, Any]:
        """获取网络配置"""
        return {
            'topology_type': self.get_value('network', 'topology_type', str, 'NSFNet'),
            'num_nodes': self.get_value('network', 'num_nodes', int, 14),
            'num_edges': self.get_value('network', 'num_edges', int, 21),
            'default_capacity': self.get_value('network', 'default_capacity', float, 100.0),
            'num_spectrum_slots': self.get_value('network', 'num_spectrum_slots', int, 100)
        }

    def get_defaults(self) -> Dict[str, Any]:
        """获取全局默认值"""
        return {
            'spectrum_slots': self.get_value('defaults', 'spectrum_slots', int, 100),
            'capacity': self.get_value('defaults', 'capacity', float, 100.0),
            'time_step': self.get_value('defaults', 'time_step', float, 0.1),
            'timeslot_duration': self.get_value('defaults', 'timeslot_duration', float, 1.0),
            'random_seed': self.get_value('defaults', 'random_seed', int, 42),
            'k_value': self.get_value('defaults', 'k_value', int, 3),
            'weight': self.get_value('defaults', 'weight', int, 1),
            'probability': self.get_value('defaults', 'probability', float, 1.0),
            'bandwidth_unit': self.get_value('defaults', 'bandwidth_unit', int, 10)
        }

    # 向后兼容的方法别名
    def get_request_generation_config(self) -> Dict[str, Any]:
        """获取请求生成配置（向后兼容别名）"""
        return self.get_traffic_config()

    def get_timeslot_mode_config(self) -> Dict[str, Any]:
        """获取时隙模式配置（向后兼容别名）"""
        return self.get_traffic_config()

    def get_parameter_change_config(self) -> Dict[str, Any]:
        """获取参数变化配置（向后兼容别名）"""
        return self.get_dynamics_config()

    def get_algorithm_weights_config(self) -> Dict[str, Any]:
        """获取算法权重配置（向后兼容别名）"""
        return self.get_algorithm_config()

    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置（向后兼容别名）"""
        return self.get_visualization_config()

    def get_all_config(self) -> Dict[str, Dict[str, Any]]:
        """获取所有配置（向后兼容）"""
        return {
            'simulation': self.get_simulation_config(),
            'traffic': self.get_traffic_config(),
            'algorithms': self.get_algorithm_config(),
            'logging': self.get_logging_config(),
            'visualization': self.get_visualization_config(),
            'performance': self.get_performance_config(),
            'dynamics': self.get_dynamics_config(),
            'test_parameters': self.get_test_parameters_config(),
            'time_management': self.get_time_management_config(),
            'network': self.get_network_config(),
            'defaults': self.get_defaults(),
            # 向后兼容的别名
            'request_generation': self.get_request_generation_config(),
            'timeslot_mode': self.get_timeslot_mode_config(),
            'parameter_change': self.get_parameter_change_config(),
            'algorithm_weights': self.get_algorithm_weights_config(),
            'output': self.get_output_config()
        }

    def print_config(self):
        """打印所有配置信息"""
        all_config = self.get_all_config()
        print("\n" + "="*60)
        print("配置文件信息汇总")
        print("="*60)

        for section_name, section_config in all_config.items():
            print(f"\n[{section_name.upper()}]")
            for key, value in section_config.items():
                print(f"  {key}: {value}")

        print("="*60)


# 向后兼容的别名
ConfigReader = EnhancedConfigReader


if __name__ == "__main__":
    # 测试增强配置读取器
    config = EnhancedConfigReader()
    config.print_config()
