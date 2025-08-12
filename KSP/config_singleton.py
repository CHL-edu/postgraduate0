#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置单例管理器
解决配置文件重复加载的性能问题
"""

import threading
from typing import Optional, Dict, Any
from config_reader_enhanced import EnhancedConfigReader


class ConfigSingleton:
    """配置单例管理器，确保配置文件只加载一次"""
    
    _instance: Optional['ConfigSingleton'] = None
    _lock = threading.Lock()
    _config_reader: Optional[EnhancedConfigReader] = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置单例"""
        if self._config_reader is None:
            with self._lock:
                if self._config_reader is None:
                    # 只在第一次初始化时加载配置
                    self._config_reader = EnhancedConfigReader()
                    print("🔧 配置单例已初始化（配置文件只加载一次）")
    
    @property
    def config_reader(self) -> EnhancedConfigReader:
        """获取配置读取器实例"""
        if self._config_reader is None:
            self.__init__()
        return self._config_reader
    
    # 代理所有配置读取方法
    def get_simulation_config(self) -> Dict[str, Any]:
        return self.config_reader.get_simulation_config()
    
    def get_traffic_config(self) -> Dict[str, Any]:
        return self.config_reader.get_traffic_config()
    
    def get_algorithm_config(self) -> Dict[str, Any]:
        return self.config_reader.get_algorithm_config()
    
    def get_logging_config(self) -> Dict[str, Any]:
        return self.config_reader.get_logging_config()
    
    def get_visualization_config(self) -> Dict[str, Any]:
        return self.config_reader.get_visualization_config()
    
    def get_performance_config(self) -> Dict[str, Any]:
        return self.config_reader.get_performance_config()
    
    def get_dynamics_config(self) -> Dict[str, Any]:
        return self.config_reader.get_dynamics_config()
    
    def get_test_parameters_config(self) -> Dict[str, Any]:
        return self.config_reader.get_test_parameters_config()
    
    def get_time_management_config(self) -> Dict[str, Any]:
        return self.config_reader.get_time_management_config()
    
    def get_network_config(self) -> Dict[str, Any]:
        return self.config_reader.get_network_config()
    
    def get_defaults(self) -> Dict[str, Any]:
        return self.config_reader.get_defaults()
    
    # 向后兼容的方法别名
    def get_request_generation_config(self) -> Dict[str, Any]:
        return self.config_reader.get_request_generation_config()
    
    def get_timeslot_mode_config(self) -> Dict[str, Any]:
        return self.config_reader.get_timeslot_mode_config()
    
    def get_parameter_change_config(self) -> Dict[str, Any]:
        return self.config_reader.get_parameter_change_config()
    
    def get_algorithm_weights_config(self) -> Dict[str, Any]:
        return self.config_reader.get_algorithm_weights_config()
    
    def get_output_config(self) -> Dict[str, Any]:
        return self.config_reader.get_output_config()
    
    def get_all_config(self) -> Dict[str, Dict[str, Any]]:
        return self.config_reader.get_all_config()
    
    def get_value(self, section: str, key: str, value_type: type = str, default: Any = None) -> Any:
        return self.config_reader.get_value(section, key, value_type, default)
    
    def print_config(self):
        return self.config_reader.print_config()


# 创建全局配置单例实例
_global_config = ConfigSingleton()


def get_config() -> ConfigSingleton:
    """获取全局配置单例实例"""
    return _global_config


# 为了向后兼容，提供一个模拟ConfigReader的类
class OptimizedConfigReader:
    """优化的配置读取器，使用单例模式避免重复加载"""
    
    def __init__(self, config_file: str = "config.ini"):
        """
        初始化优化配置读取器
        
        Args:
            config_file: 配置文件路径（为了兼容性保留，但实际使用单例）
        """
        # 不再重复加载配置文件，直接使用单例
        self._config_singleton = get_config()
    
    def __getattr__(self, name):
        """代理所有方法调用到配置单例"""
        return getattr(self._config_singleton, name)


if __name__ == "__main__":
    # 测试配置单例
    print("测试配置单例模式...")
    
    # 创建多个实例，应该只加载一次配置
    config1 = ConfigSingleton()
    config2 = ConfigSingleton()
    config3 = get_config()
    
    print(f"config1 is config2: {config1 is config2}")
    print(f"config2 is config3: {config2 is config3}")
    
    # 测试配置读取
    sim_config = config1.get_simulation_config()
    print(f"仿真配置: {len(sim_config)} 个参数")
    
    # 测试优化的配置读取器
    optimized_reader = OptimizedConfigReader()
    alg_config = optimized_reader.get_algorithm_config()
    print(f"算法配置: {len(alg_config)} 个参数")
