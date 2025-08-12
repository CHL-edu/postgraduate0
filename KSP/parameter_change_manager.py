#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数动态变化管理器
实现仿真参数的动态变化功能，支持多种变化模式
"""

import math
import numpy as np
from typing import Dict, Any, Optional
from config_reader import ConfigReader


class ParameterChangeManager:
    """参数动态变化管理器"""
    
    def __init__(self, config_reader: ConfigReader):
        """
        初始化参数变化管理器
        
        Args:
            config_reader: 配置读取器实例
        """
        self.config_reader = config_reader
        self.change_config = config_reader.get_parameter_change_config()
        self.base_request_config = config_reader.get_request_generation_config()
        self.base_test_config = config_reader.get_test_parameters_config()
        self.base_timeslot_config = config_reader.get_timeslot_mode_config()
        
        # 参数变化历史记录
        self.parameter_history = []
        
        print(f"📊 参数变化管理器初始化完成")
        print(f"   变化模式: {self.change_config['change_mode']}")
        print(f"   变化周期: {self.change_config['change_period']} 时隙")
        print(f"   变化幅度: ±{self.change_config['change_amplitude']*100:.1f}%")
    
    def is_change_enabled(self) -> bool:
        """检查是否启用参数变化"""
        return self.change_config['test_changePara']
    
    def get_dynamic_request_generation_config(self, timeslot: int, total_timeslots: int) -> Dict[str, Any]:
        """
        获取动态的请求生成配置
        
        Args:
            timeslot: 当前时隙
            total_timeslots: 总时隙数
            
        Returns:
            动态配置字典
        """
        if not self.is_change_enabled():
            return self.base_request_config.copy()
        
        config = self.base_request_config.copy()
        
        # 计算变化因子
        change_factor = self._calculate_change_factor(timeslot, total_timeslots)
        
        # 应用参数变化
        if self.change_config['change_small_bandwidth_probability']:
            config['small_bandwidth_probability'] = self._apply_change(
                self.base_request_config['small_bandwidth_probability'],
                change_factor,
                min_val=self.change_config['bandwidth_probability_min'],
                max_val=self.change_config['bandwidth_probability_max']
            )
        
        if self.change_config['change_duration_mean']:
            config['duration_mean'] = self._apply_change(
                self.base_request_config['duration_mean'],
                change_factor,
                min_val=0.5, max_val=5.0  # 持续时间范围限制
            )
        
        if self.change_config['change_bandwidth_ranges']:
            # 带宽范围变化
            config['small_bandwidth_min'] = self._apply_change(
                self.base_request_config['small_bandwidth_min'],
                change_factor,
                min_val=1.0, max_val=50.0
            )
            config['small_bandwidth_max'] = self._apply_change(
                self.base_request_config['small_bandwidth_max'],
                change_factor,
                min_val=10.0, max_val=100.0
            )
            config['large_bandwidth_min'] = self._apply_change(
                self.base_request_config['large_bandwidth_min'],
                change_factor,
                min_val=50.0, max_val=150.0
            )
            config['large_bandwidth_max'] = self._apply_change(
                self.base_request_config['large_bandwidth_max'],
                change_factor,
                min_val=80.0, max_val=200.0
            )
        
        return config
    
    def get_dynamic_timeslot_config(self, timeslot: int, total_timeslots: int) -> Dict[str, Any]:
        """
        获取动态的时隙模式配置
        
        Args:
            timeslot: 当前时隙
            total_timeslots: 总时隙数
            
        Returns:
            动态配置字典
        """
        if not self.is_change_enabled():
            return self.base_timeslot_config.copy()
        
        config = self.base_timeslot_config.copy()
        
        # 计算变化因子
        change_factor = self._calculate_change_factor(timeslot, total_timeslots)
        
        # 应用参数变化
        if self.change_config['change_requests_per_timeslot']:
            config['requests_per_timeslot'] = self._apply_change(
                self.base_timeslot_config['requests_per_timeslot'],
                change_factor,
                min_val=self.change_config['requests_per_timeslot_min'],
                max_val=self.change_config['requests_per_timeslot_max']
            )
        
        return config
    
    def get_dynamic_test_config(self, timeslot: int, total_timeslots: int) -> Dict[str, Any]:
        """
        获取动态的测试参数配置
        
        Args:
            timeslot: 当前时隙
            total_timeslots: 总时隙数
            
        Returns:
            动态配置字典
        """
        if not self.is_change_enabled():
            return self.base_test_config.copy()
        
        config = self.base_test_config.copy()
        
        # 计算变化因子
        change_factor = self._calculate_change_factor(timeslot, total_timeslots)
        
        # 应用参数变化
        if self.change_config['change_spectrum_slots']:
            config['test_num_spectrum_slots'] = int(self._apply_change(
                self.base_test_config['test_num_spectrum_slots'],
                change_factor,
                min_val=self.change_config['spectrum_slots_min'],
                max_val=self.change_config['spectrum_slots_max']
            ))
        
        return config
    
    def _calculate_change_factor(self, timeslot: int, total_timeslots: int) -> float:
        """
        计算变化因子（-1到1之间）
        
        Args:
            timeslot: 当前时隙
            total_timeslots: 总时隙数
            
        Returns:
            变化因子
        """
        period = self.change_config['change_period']
        mode = self.change_config['change_mode']
        phase_offset = self.change_config['phase_offset']
        
        # 计算相位
        phase = (2 * math.pi * timeslot / period) + phase_offset
        
        if mode == 'sine_wave':
            return math.sin(phase)
        elif mode == 'triangle_wave':
            # 三角波：-1到1的线性变化
            normalized_phase = (phase % (2 * math.pi)) / (2 * math.pi)
            if normalized_phase < 0.25:
                return 4 * normalized_phase
            elif normalized_phase < 0.75:
                return 2 - 4 * normalized_phase
            else:
                return 4 * normalized_phase - 4
        elif mode == 'step_change':
            # 阶跃变化：每个周期内保持恒定
            cycle_position = (timeslot % period) / period
            return 1 if cycle_position < 0.5 else -1
        elif mode == 'linear_trend':
            # 线性趋势：从-1到1线性变化
            return -1 + 2 * (timeslot / total_timeslots)
        else:
            # 默认使用正弦波
            return math.sin(phase)
    
    def _apply_change(self, base_value: float, change_factor: float, 
                     min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """
        应用变化到基础值
        
        Args:
            base_value: 基础值
            change_factor: 变化因子（-1到1）
            min_val: 最小值限制
            max_val: 最大值限制
            
        Returns:
            变化后的值
        """
        amplitude = self.change_config['change_amplitude']
        min_ratio = self.change_config['min_value_ratio']
        max_ratio = self.change_config['max_value_ratio']
        
        # 计算变化后的值
        change_ratio = 1 + amplitude * change_factor
        
        # 应用比例限制
        change_ratio = max(min_ratio, min(max_ratio, change_ratio))
        
        new_value = base_value * change_ratio
        
        # 应用绝对值限制
        if min_val is not None:
            new_value = max(min_val, new_value)
        if max_val is not None:
            new_value = min(max_val, new_value)
        
        return new_value
    
    def record_parameters(self, timeslot: int, total_timeslots: int):
        """
        记录当前时隙的参数值
        
        Args:
            timeslot: 当前时隙
            total_timeslots: 总时隙数
        """
        if not self.is_change_enabled():
            return
        
        change_factor = self._calculate_change_factor(timeslot, total_timeslots)
        request_config = self.get_dynamic_request_generation_config(timeslot, total_timeslots)
        timeslot_config = self.get_dynamic_timeslot_config(timeslot, total_timeslots)
        test_config = self.get_dynamic_test_config(timeslot, total_timeslots)
        
        record = {
            'timeslot': timeslot,
            'change_factor': change_factor,
            'small_bandwidth_probability': request_config['small_bandwidth_probability'],
            'duration_mean': request_config['duration_mean'],
            'requests_per_timeslot': timeslot_config['requests_per_timeslot'],
            'test_num_spectrum_slots': test_config['test_num_spectrum_slots']
        }
        
        self.parameter_history.append(record)
    
    def get_parameter_history(self) -> list:
        """获取参数变化历史"""
        return self.parameter_history.copy()
    
    def save_parameter_history(self, filename: str = "parameter_change_history.json"):
        """
        保存参数变化历史到文件
        
        Args:
            filename: 保存文件名
        """
        import json
        import os
        
        # 确保log目录存在
        log_dir = r"E:\PythonProject\KSP\log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        filepath = os.path.join(log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.parameter_history, f, indent=2, ensure_ascii=False)
        
        print(f"📊 参数变化历史已保存到: {filepath}")


if __name__ == "__main__":
    # 测试参数变化管理器
    from config_reader import ConfigReader
    
    config_reader = ConfigReader()
    manager = ParameterChangeManager(config_reader)
    
    print("\n=== 参数变化测试 ===")
    total_timeslots = 100
    
    for timeslot in [0, 25, 50, 75, 99]:
        print(f"\n时隙 {timeslot}:")
        change_factor = manager._calculate_change_factor(timeslot, total_timeslots)
        print(f"  变化因子: {change_factor:.3f}")
        
        if manager.is_change_enabled():
            req_config = manager.get_dynamic_request_generation_config(timeslot, total_timeslots)
            print(f"  小带宽概率: {req_config['small_bandwidth_probability']:.3f}")
            print(f"  持续时间均值: {req_config['duration_mean']:.3f}")
        else:
            print("  参数变化未启用")
