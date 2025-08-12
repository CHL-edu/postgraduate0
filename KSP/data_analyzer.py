#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFNet 数据分析工具
分析记录的仿真数据并生成报告
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import seaborn as sns


class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, log_dir: str = r"E:\PythonProject\KSP\log"):
        """
        初始化数据分析器（已适配优化后的参数）

        Args:
            log_dir: 日志目录路径（默认使用优化后的绝对路径）
        """
        self.log_dir = log_dir
        self.algorithms = []
        self.analysis_results = {}

        # 检查日志目录
        if os.path.exists(log_dir):
            self._discover_algorithms()
            self._check_data_integrity()
        else:
            print(f"❌ 日志目录不存在: {log_dir}")

    def _check_data_integrity(self):
        """检查数据完整性"""
        print(f"🔍 检查数据完整性...")
        for algorithm in self.algorithms:
            # 检查性能指标文件
            metrics_file = os.path.join(self.log_dir, "performance_metrics", algorithm, "metrics.csv")
            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    non_zero_rows = df[df['total_requests'] > 0] if 'total_requests' in df.columns else df
                    print(f"   {algorithm}: 性能数据 {len(df)} 行，有效数据 {len(non_zero_rows)} 行")
                except Exception as e:
                    print(f"   {algorithm}: 性能数据读取失败 - {e}")

            # 检查请求决策文件
            decision_dir = os.path.join(self.log_dir, "request_decisions", algorithm)
            if os.path.exists(decision_dir):
                files = [f for f in os.listdir(decision_dir) if f.endswith('.csv')]
                total_requests = 0
                for file in files[:5]:  # 检查前5个文件
                    try:
                        df = pd.read_csv(os.path.join(decision_dir, file))
                        total_requests += len(df)
                    except:
                        pass
                print(f"   {algorithm}: 请求文件 {len(files)} 个，前5个文件请求数 {total_requests}")
        print("✅ 数据完整性检查完成")

    def _get_enhanced_colors(self, algorithms: list, base_colors: dict) -> dict:
        """获取增强的颜色方案，防止重叠和提高对比度"""
        import matplotlib.colors as mcolors
        import numpy as np

        enhanced_colors = {}

        # 高对比度备选颜色池（避免相似颜色）
        high_contrast_colors = [
            '#1f77b4',  # 蓝色
            '#ff7f0e',  # 橙色
            '#00cc44',  # 亮绿色
            '#d62728',  # 红色
            '#9467bd',  # 紫色
            '#8B4513',  # 深棕色
            '#e377c2',  # 粉色
            '#17becf',  # 青色
            '#bcbd22',  # 橄榄色
            '#ff9896',  # 浅红色
            '#c5b0d5',  # 浅紫色
            '#c49c94',  # 浅棕色
            '#f7b6d3',  # 浅粉色
            '#c7c7c7',  # 灰色
            '#dbdb8d',  # 浅橄榄色
        ]

        # 为现有算法分配颜色
        for i, alg in enumerate(algorithms):
            if alg in base_colors:
                enhanced_colors[alg] = base_colors[alg]
            elif i < len(high_contrast_colors):
                enhanced_colors[alg] = high_contrast_colors[i]
            else:
                # 如果超出预定义颜色，使用色彩空间生成
                extra_colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))
                enhanced_colors[alg] = mcolors.to_hex(extra_colors[i])

        return enhanced_colors

    def _check_color_contrast(self, colors: dict) -> dict:
        """检查颜色对比度并优化相似颜色"""
        import matplotlib.colors as mcolors

        def rgb_distance(color1, color2):
            """计算两个RGB颜色的欧几里得距离"""
            rgb1 = mcolors.to_rgb(color1)
            rgb2 = mcolors.to_rgb(color2)
            return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5

        optimized_colors = colors.copy()
        algorithms = list(colors.keys())

        # 检查所有颜色对的对比度
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]
                color1, color2 = colors[alg1], colors[alg2]

                # 如果颜色太相似（距离小于0.3），调整第二个颜色
                if rgb_distance(color1, color2) < 0.3:
                    print(f"⚠️ 检测到相似颜色: {alg1} 和 {alg2}")

                    # 为第二个算法选择更对比的颜色
                    contrast_colors = ['#FF1493', '#00CED1', '#FFD700', '#32CD32', '#FF4500', '#8A2BE2']
                    for new_color in contrast_colors:
                        if all(rgb_distance(new_color, optimized_colors[other_alg]) > 0.3
                               for other_alg in algorithms if other_alg != alg2):
                            optimized_colors[alg2] = new_color
                            print(f"✅ 已优化 {alg2} 的颜色为: {new_color}")
                            break

        return optimized_colors

    def _calculate_summary_metrics(self, algorithm: str) -> pd.DataFrame:
        """从请求决策数据计算汇总性能指标"""
        try:
            decision_dir = os.path.join(self.log_dir, "request_decisions", algorithm)
            if not os.path.exists(decision_dir):
                return None

            # 收集所有请求数据
            all_requests = []
            files = [f for f in os.listdir(decision_dir) if f.endswith('.csv')]

            for file in files:
                try:
                    df = pd.read_csv(os.path.join(decision_dir, file))
                    if len(df) > 0:  # 只处理非空文件
                        all_requests.append(df)
                except:
                    continue

            if not all_requests:
                return None

            # 合并所有请求数据
            combined_df = pd.concat(all_requests, ignore_index=True)

            # 计算汇总指标
            total_requests = len(combined_df)
            successful_requests = len(combined_df[combined_df['allocation_status'] == 'success'])
            blocked_requests = total_requests - successful_requests
            blocking_rate = blocked_requests / total_requests if total_requests > 0 else 0

            # 计算平均跳数
            successful_df = combined_df[combined_df['allocation_status'] == 'success']
            avg_hop_count = 0
            if len(successful_df) > 0:
                hop_counts = []
                for path_str in successful_df['selected_path']:
                    try:
                        path = eval(path_str) if isinstance(path_str, str) else path_str
                        if isinstance(path, list) and len(path) > 1:
                            hop_counts.append(len(path) - 1)
                    except:
                        continue
                avg_hop_count = np.mean(hop_counts) if hop_counts else 0

            # 创建汇总数据框
            summary_data = {
                'algorithm': [algorithm],
                'total_requests': [total_requests],
                'successful_requests': [successful_requests],
                'blocked_requests': [blocked_requests],
                'blocking_rate': [blocking_rate],
                'average_hop_count': [avg_hop_count],
                'spectrum_utilization': [0.0],  # 需要从频谱数据计算
                'throughput': [successful_requests],
                'fragmentation_rate': [0.0]  # 需要从频谱数据计算
            }

            return pd.DataFrame(summary_data)

        except Exception as e:
            print(f"计算 {algorithm} 汇总指标失败: {e}")
            return None

    def _discover_algorithms(self):
        """发现已记录的算法"""
        spectrum_dir = os.path.join(self.log_dir, "spectrum_state")
        if os.path.exists(spectrum_dir):
            self.algorithms = [d for d in os.listdir(spectrum_dir) 
                             if os.path.isdir(os.path.join(spectrum_dir, d))]
            print(f"发现算法: {self.algorithms}")
        else:
            print("❌ 频谱状态目录不存在")
    
    def analyze_performance_metrics(self) -> Dict[str, pd.DataFrame]:
        """分析性能指标（适配优化后的数据格式）"""
        print("\n📈 分析性能指标...")
        performance_data = {}

        for algorithm in self.algorithms:
            metrics_file = os.path.join(self.log_dir, "performance_metrics", algorithm, "metrics.csv")

            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)

                    # 检查是否有有效数据（总请求数大于0）
                    if 'total_requests' in df.columns:
                        valid_df = df[df['total_requests'] > 0]
                        if len(valid_df) > 0:
                            # 有有效的时隙级数据
                            df = valid_df.copy()
                            if 'timeslot' in df.columns:
                                df = df.sort_values('timeslot').reset_index(drop=True)
                            performance_data[algorithm] = df

                            print(f"\n{algorithm} 性能统计（时隙级数据）:")
                            print(f"  有效时隙数: {len(df)}")
                            print(f"  平均阻塞率: {df['blocking_rate'].mean():.6f} ± {df['blocking_rate'].std():.6f}")
                            print(f"  平均频谱利用率: {df['spectrum_utilization'].mean():.4f} ± {df['spectrum_utilization'].std():.4f}")
                            if 'fragmentation_rate' in df.columns:
                                print(f"  平均碎片化率: {df['fragmentation_rate'].mean():.4f} ± {df['fragmentation_rate'].std():.4f}")
                            if 'average_hop_count' in df.columns:
                                print(f"  平均跳数: {df['average_hop_count'].mean():.2f} ± {df['average_hop_count'].std():.2f}")
                            if 'computational_overhead' in df.columns:
                                print(f"  平均计算开销: {df['computational_overhead'].mean():.6f} ± {df['computational_overhead'].std():.6f} 秒")
                            print(f"  平均吞吐量: {df['throughput'].mean():.2f} ± {df['throughput'].std():.2f}")
                        else:
                            # 没有有效的时隙级数据，尝试从请求数据计算汇总指标
                            summary_df = self._calculate_summary_metrics(algorithm)
                            if summary_df is not None:
                                performance_data[algorithm] = summary_df
                                print(f"\n{algorithm} 性能统计（汇总数据）:")
                                print(f"  总请求数: {summary_df['total_requests'].iloc[0]}")
                                print(f"  成功请求数: {summary_df['successful_requests'].iloc[0]}")
                                print(f"  阻塞率: {summary_df['blocking_rate'].iloc[0]:.6f}")
                                print(f"  平均跳数: {summary_df['average_hop_count'].iloc[0]:.2f}")
                            else:
                                print(f"⚠️ {algorithm}: 无法获取有效性能数据")
                    else:
                        # 旧格式数据，直接使用
                        performance_data[algorithm] = df
                        print(f"✅ {algorithm}: 加载了 {len(df)} 条性能记录（旧格式）")

                except Exception as e:
                    print(f"❌ 分析 {algorithm} 失败: {e}")
            else:
                print(f"❌ 文件不存在: {metrics_file}")

        return performance_data
    
    def analyze_spectrum_utilization(self) -> Dict[str, Dict]:
        """分析频谱利用率"""
        print("\n🌈 分析频谱利用率...")
        spectrum_analysis = {}
        
        for algorithm in self.algorithms:
            spectrum_dir = os.path.join(self.log_dir, "spectrum_state", algorithm)
            
            if os.path.exists(spectrum_dir):
                files = [f for f in os.listdir(spectrum_dir) if f.endswith('.csv')]
                
                if files:
                    # 分析第一个和最后一个时隙
                    first_file = os.path.join(spectrum_dir, "timeslot_0.csv")
                    last_file = os.path.join(spectrum_dir, f"timeslot_{len(files)-1}.csv")
                    
                    analysis = {
                        'total_timeslots': len(files),
                        'spectrum_evolution': []
                    }
                    
                    for i, file_path in enumerate([first_file, last_file]):
                        if os.path.exists(file_path):
                            try:
                                df = pd.read_csv(file_path, index_col=0)
                                
                                # 计算每条边的利用率
                                edge_utilizations = df.sum(axis=0) / len(df)
                                
                                analysis['spectrum_evolution'].append({
                                    'timeslot': 0 if i == 0 else len(files)-1,
                                    'overall_utilization': df.sum().sum() / (df.shape[0] * df.shape[1]),
                                    'max_edge_utilization': edge_utilizations.max(),
                                    'min_edge_utilization': edge_utilizations.min(),
                                    'avg_edge_utilization': edge_utilizations.mean(),
                                    'edge_utilization_std': edge_utilizations.std()
                                })
                                
                            except Exception as e:
                                print(f"❌ 分析频谱文件失败: {e}")
                    
                    spectrum_analysis[algorithm] = analysis
                    
                    print(f"\n{algorithm} 频谱分析:")
                    print(f"  总时隙数: {analysis['total_timeslots']}")
                    for evo in analysis['spectrum_evolution']:
                        print(f"  时隙 {evo['timeslot']}: 总体利用率 {evo['overall_utilization']:.3f}, "
                              f"边利用率 {evo['avg_edge_utilization']:.3f}±{evo['edge_utilization_std']:.3f}")
        
        return spectrum_analysis
    
    def analyze_request_decisions(self) -> Dict[str, Dict]:
        """分析请求分配决策"""
        print("\n📋 分析请求分配决策...")
        decision_analysis = {}
        
        for algorithm in self.algorithms:
            decision_dir = os.path.join(self.log_dir, "request_decisions", algorithm)
            
            if os.path.exists(decision_dir):
                files = [f for f in os.listdir(decision_dir) if f.endswith('.csv')]
                
                all_decisions = []
                for file in files:
                    file_path = os.path.join(decision_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        all_decisions.append(df)
                    except Exception as e:
                        print(f"❌ 读取决策文件失败: {e}")
                
                if all_decisions:
                    # 过滤掉空的DataFrame以避免FutureWarning
                    non_empty_decisions = [df for df in all_decisions if not df.empty]
                    if non_empty_decisions:
                        combined_df = pd.concat(non_empty_decisions, ignore_index=True)
                    else:
                        print(f"⚠️ {algorithm}: 所有决策文件都为空")
                        continue
                    
                    # 分析统计
                    total_requests = len(combined_df)
                    successful_requests = len(combined_df[combined_df['allocation_status'] == 'success'])
                    blocked_requests = len(combined_df[combined_df['allocation_status'] == 'blocked'])
                    
                    # 分析成功请求的特征
                    success_df = combined_df[combined_df['allocation_status'] == 'success']
                    
                    analysis = {
                        'total_requests': total_requests,
                        'successful_requests': successful_requests,
                        'blocked_requests': blocked_requests,
                        'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
                        'avg_bandwidth_demand': combined_df['bandwidth_demand'].mean(),
                        'avg_spectrum_slots_demand': combined_df['spectrum_slots_demand'].mean(),
                        'avg_processing_time': combined_df['processing_time'].mean(),
                    }
                    
                    if len(success_df) > 0:
                        analysis.update({
                            'avg_path_length': success_df['selected_path'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0).mean(),
                            'avg_spectrum_range': (success_df['allocated_spectrum_end'] - success_df['allocated_spectrum_start'] + 1).mean()
                        })
                    
                    decision_analysis[algorithm] = analysis
                    
                    print(f"\n{algorithm} 决策分析:")
                    print(f"  总请求数: {analysis['total_requests']}")
                    print(f"  成功率: {analysis['success_rate']:.3f}")
                    print(f"  平均带宽需求: {analysis['avg_bandwidth_demand']:.2f} Gbps")
                    print(f"  平均处理时间: {analysis['avg_processing_time']*1000:.2f} ms")
                    if 'avg_path_length' in analysis:
                        print(f"  平均路径长度: {analysis['avg_path_length']:.2f} 跳")
        
        return decision_analysis

    def _verify_algorithm_coverage(self, performance_data: Dict[str, pd.DataFrame]):
        """验证算法覆盖情况"""
        expected_algorithms = [
            'KSP_FF', 'Random_Fit', 'FA_CA', 'KSP_LB',
            'KSP_FA', 'Custom_Algorithm', 'Custom_Algorithm_Random'
        ]

        found_algorithms = list(performance_data.keys())
        print(f"📋 发现的算法: {found_algorithms}")

        missing_algorithms = [alg for alg in expected_algorithms if alg not in found_algorithms]
        extra_algorithms = [alg for alg in found_algorithms if alg not in expected_algorithms]

        if missing_algorithms:
            print(f"⚠️ 缺失的算法: {missing_algorithms}")

        if extra_algorithms:
            print(f"ℹ️ 额外的算法: {extra_algorithms}")

        if len(found_algorithms) == len(expected_algorithms) and not missing_algorithms:
            print("✅ 所有7种算法数据完整")
        else:
            print(f"📊 当前有 {len(found_algorithms)} 种算法的数据")

    def generate_comparison_report(self, performance_data: Dict[str, pd.DataFrame]):
        """生成算法对比报告"""
        print("\n📊 生成算法对比报告...")
        
        if len(performance_data) < 2:
            print("❌ 需要至少2个算法的数据进行对比")
            return
        
        # 创建对比表格
        comparison_metrics = ['blocking_rate', 'spectrum_utilization', 'fragmentation_rate', 'average_hop_count', 'computational_overhead', 'throughput']
        comparison_data = []
        
        for algorithm, df in performance_data.items():
            row = {'Algorithm': algorithm}
            for metric in comparison_metrics:
                if metric in df.columns:
                    row[f'{metric}_mean'] = df[metric].mean()
                    row[f'{metric}_std'] = df[metric].std()
                else:
                    row[f'{metric}_mean'] = 0
                    row[f'{metric}_std'] = 0
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存对比报告
        report_file = os.path.join(self.log_dir, "algorithm_comparison_report.csv")
        comparison_df.to_csv(report_file, index=False)
        print(f"✅ 对比报告已保存: {report_file}")
        
        # 显示对比结果
        print(f"\n📈 算法性能对比:")
        print("-" * 140)
        print(f"{'算法':<20} {'阻塞率':<15} {'频谱利用率':<15} {'碎片化率':<15} {'平均跳数':<15} {'计算开销(s)':<15} {'吞吐量':<15}")
        print("-" * 140)

        for _, row in comparison_df.iterrows():
            hop_count_str = f"{row['average_hop_count_mean']:.3f}±{row['average_hop_count_std']:.3f}" if 'average_hop_count_mean' in row else "N/A"
            comp_overhead_str = f"{row['computational_overhead_mean']:.6f}±{row['computational_overhead_std']:.6f}" if 'computational_overhead_mean' in row else "N/A"
            print(f"{row['Algorithm']:<20} "
                  f"{row['blocking_rate_mean']:.3f}±{row['blocking_rate_std']:.3f}  "
                  f"{row['spectrum_utilization_mean']:.3f}±{row['spectrum_utilization_std']:.3f}  "
                  f"{row['fragmentation_rate_mean']:.3f}±{row['fragmentation_rate_std']:.3f}  "
                  f"{hop_count_str:<15} "
                  f"{comp_overhead_str:<15} "
                  f"{row['throughput_mean']:.3f}±{row['throughput_std']:.3f}")
    
    def create_visualization_plots(self, performance_data: Dict[str, pd.DataFrame]):
        """创建可视化图表"""
        print("\n📊 创建可视化图表...")

        if not performance_data:
            print("❌ 没有性能数据可供可视化")
            return

        # 验证算法覆盖
        self._verify_algorithm_coverage(performance_data)

        # 设置图表样式
        plt.style.use('default')

        # 为7种算法定义高对比度颜色（优化绿色和棕色的区分度）
        algorithm_colors = {
            'KSP_FF': '#1f77b4',                    # 蓝色
            'Random_Fit': '#ff7f0e',               # 橙色
            'FA_CA': '#00cc44',                    # 亮绿色（增强对比度）
            'KSP_LB': '#d62728',                   # 红色
            'KSP_FA': '#9467bd',                   # 紫色
            'Custom_Algorithm': '#E8C3A2',         # 深棕色（增强对比度）
            'Custom_Algorithm_Random': '#e377c2'   # 粉色
        }

        # 定义透明度设置
        line_alpha = 0.85      # 折线图透明度
        bar_alpha = 0.75       # 柱状图透明度
        grid_alpha = 0.25      # 网格透明度
        fill_alpha = 0.15      # 填充区域透明度

        # 根据算法数量动态调整透明度
        num_algorithms = len(performance_data)
        if num_algorithms > 5:
            line_alpha = max(0.6, line_alpha - 0.05 * (num_algorithms - 5))
            bar_alpha = max(0.6, bar_alpha - 0.05 * (num_algorithms - 5))
            fill_alpha = max(0.1, fill_alpha - 0.02 * (num_algorithms - 5))

        # 定义6个性能指标
        metrics = ['blocking_rate', 'spectrum_utilization', 'fragmentation_rate', 'average_hop_count', 'computational_overhead', 'throughput']
        titles = ['Blocking Rate', 'Spectrum Utilization', 'Fragmentation Rate', 'Average Hop Count', 'Computational Overhead', 'Throughput']
        y_labels = ['Blocking Rate', 'Spectrum Utilization', 'Fragmentation Rate', 'Average Hop Count', 'Computational Overhead (seconds)', 'Throughput']

        # 检查数据类型：时隙级数据还是汇总数据
        has_timeslot_data = any('timeslot' in df.columns and len(df) > 1 for df in performance_data.values())

        # 获取增强的颜色方案
        enhanced_colors = self._get_enhanced_colors(list(performance_data.keys()), algorithm_colors)

        # 检查并优化颜色对比度
        enhanced_colors = self._check_color_contrast(enhanced_colors)

        print(f"🎨 最终颜色方案:")
        for alg, color in enhanced_colors.items():
            print(f"   {alg}: {color}")

        # 创建透明度配置字典
        alpha_config = {
            'line_alpha': line_alpha,
            'bar_alpha': bar_alpha,
            'grid_alpha': grid_alpha,
            'fill_alpha': fill_alpha
        }

        if has_timeslot_data:
            # 绘制时隙级性能趋势图
            self._plot_timeslot_trends(performance_data, enhanced_colors, metrics, titles, y_labels, alpha_config)
        else:
            # 绘制汇总性能对比图
            self._plot_summary_comparison(performance_data, enhanced_colors, alpha_config)

    def _plot_timeslot_trends(self, performance_data: Dict[str, pd.DataFrame], algorithm_colors: dict,
                             metrics: list, titles: list, y_labels: list, alpha_config: dict):
        """绘制时隙级性能趋势图"""
        print("📈 绘制时隙级性能趋势...")

        # 提取透明度配置
        line_alpha = alpha_config['line_alpha']
        grid_alpha = alpha_config['grid_alpha']
        fill_alpha = alpha_config['fill_alpha']

        # 为每个指标创建独立的图表
        for metric, title, y_label in zip(metrics, titles, y_labels):
            plt.figure(figsize=(12, 8))

            # 检查是否有算法包含该指标
            has_data = False
            for algorithm, df in performance_data.items():
                if metric in df.columns and 'timeslot' in df.columns:
                    has_data = True
                    # 确保数据按时隙顺序排序，避免折线首尾相接
                    df_sorted = df.sort_values('timeslot').reset_index(drop=True)

                    # 使用预定义的颜色，如果算法不在预定义列表中则使用默认颜色
                    color = algorithm_colors.get(algorithm, '#000000')

                    plt.plot(df_sorted['timeslot'], df_sorted[metric],
                            label=algorithm, linewidth=2.5, alpha=line_alpha, color=color)

                    # 移除填充区域，只保留线条以避免重叠
                    # 注释掉填充代码：
                    # if len(df_sorted) > 1:
                    #     plt.fill_between(df_sorted['timeslot'], df_sorted[metric],
                    #                    alpha=fill_alpha, color=color)

            if has_data:
                plt.title(f'NSFNet Algorithm Performance - {title}', fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('Time Slot', fontsize=12)
                plt.ylabel(y_label, fontsize=12)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                plt.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.8)

                # 为小数值指标设置科学计数法
                if metric in ['blocking_rate', 'fragmentation_rate', 'computational_overhead']:
                    plt.ylim(bottom=0)  # 确保纵坐标从0开始显示
                    # 使用科学计数法显示小数值
                    from matplotlib.ticker import ScalarFormatter
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    if metric == 'blocking_rate':
                        formatter.set_powerlimits((-3, -3))  # 阻塞率强制使用科学计数法
                    elif metric == 'computational_overhead':
                        formatter.set_powerlimits((-6, -3))  # 计算开销使用科学计数法
                    else:
                        formatter.set_powerlimits((-2, 2))   # 其他指标自动判断
                    plt.gca().yaxis.set_major_formatter(formatter)
                elif metric in ['spectrum_utilization', 'average_hop_count', 'throughput']:
                    plt.ylim(bottom=0)  # 确保纵坐标从0开始显示

                # 设置坐标轴样式
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['left'].set_linewidth(0.5)
                plt.gca().spines['bottom'].set_linewidth(0.5)

                plt.tight_layout()

                # 保存图表到日志目录
                plot_file = os.path.join(self.log_dir, f"performance_{metric}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                print(f"✅ {title} 图表已保存: {plot_file}")
            else:
                print(f"⚠️ 没有找到 {metric} 的数据")

            plt.close()

        # 生成图表创建总结
        self._generate_chart_summary(performance_data)
        print(f"✅ 所有性能图表创建完成！")

    def _plot_summary_comparison(self, performance_data: Dict[str, pd.DataFrame], algorithm_colors: dict, alpha_config: dict):
        """绘制汇总性能对比图"""
        print("📊 绘制汇总性能对比...")

        # 提取透明度配置
        bar_alpha = alpha_config['bar_alpha']
        grid_alpha = alpha_config['grid_alpha']

        # 提取汇总数据
        algorithms = []
        blocking_rates = []
        spectrum_utilizations = []
        hop_counts = []
        throughputs = []

        for algorithm, df in performance_data.items():
            algorithms.append(algorithm)

            # 提取关键指标
            if 'blocking_rate' in df.columns:
                blocking_rates.append(df['blocking_rate'].iloc[0] if len(df) > 0 else 0)
            else:
                blocking_rates.append(0)

            if 'spectrum_utilization' in df.columns:
                spectrum_utilizations.append(df['spectrum_utilization'].iloc[0] if len(df) > 0 else 0)
            else:
                spectrum_utilizations.append(0)

            if 'average_hop_count' in df.columns:
                hop_counts.append(df['average_hop_count'].iloc[0] if len(df) > 0 else 0)
            else:
                hop_counts.append(0)

            if 'throughput' in df.columns:
                throughputs.append(df['throughput'].iloc[0] if len(df) > 0 else 0)
            else:
                throughputs.append(0)

        # 创建2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Performance Comparison (Optimized Parameters)', fontsize=16, fontweight='bold')

        # 1. 阻塞率对比（线性刻度，科学计数法）- 无填充柱状图
        colors = [algorithm_colors.get(alg, '#000000') for alg in algorithms]
        bars1 = ax1.bar(algorithms, blocking_rates, fill=False,
                        edgecolor=colors, linewidth=2.5, alpha=0.9)
        ax1.set_title('Blocking Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Blocking Rate', fontsize=12)
        # 使用线性刻度并确保显示0
        ax1.set_ylim(bottom=0)
        # 设置科学计数法
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, -3))  # 强制使用科学计数法
        ax1.yaxis.set_major_formatter(formatter)
        ax1.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.8)
        ax1.tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值标签
        for bar, rate in zip(bars1, blocking_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.6f}', ha='center', va='bottom', fontsize=10)

        # 2. 频谱利用率对比 - 无填充柱状图
        bars2 = ax2.bar(algorithms, spectrum_utilizations, fill=False,
                        edgecolor=colors, linewidth=2.5, alpha=0.9)
        ax2.set_title('Spectrum Utilization Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Spectrum Utilization', fontsize=12)
        ax2.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.8)
        ax2.tick_params(axis='x', rotation=45)

        for bar, util in zip(bars2, spectrum_utilizations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{util:.4f}', ha='center', va='bottom', fontsize=10)

        # 3. 平均跳数对比 - 无填充柱状图
        bars3 = ax3.bar(algorithms, hop_counts, fill=False,
                        edgecolor=colors, linewidth=2.5, alpha=0.9)
        ax3.set_title('Average Hop Count Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Average Hop Count', fontsize=12)
        ax3.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.8)
        ax3.tick_params(axis='x', rotation=45)

        for bar, hops in zip(bars3, hop_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{hops:.2f}', ha='center', va='bottom', fontsize=10)

        # 4. 吞吐量对比 - 无填充柱状图
        bars4 = ax4.bar(algorithms, throughputs, fill=False,
                        edgecolor=colors, linewidth=2.5, alpha=0.9)
        ax4.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Throughput (Requests)', fontsize=12)
        ax4.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.8)
        ax4.tick_params(axis='x', rotation=45)

        for bar, throughput in zip(bars4, throughputs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{throughput:.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # 保存图表
        plot_file = os.path.join(self.log_dir, "algorithm_performance_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ 算法对比图已保存: {plot_file}")
        plt.show()

        # 创建算法排名表
        self._create_ranking_table(algorithms, blocking_rates, spectrum_utilizations, hop_counts, throughputs)

        print("✅ 汇总对比图绘制完成")

    def _create_ranking_table(self, algorithms, blocking_rates, spectrum_utilizations, hop_counts, throughputs):
        """创建算法排名表"""
        print("\n🏆 算法性能排名:")
        print("=" * 80)

        # 按阻塞率排序（越低越好）
        ranking_data = list(zip(algorithms, blocking_rates, spectrum_utilizations, hop_counts, throughputs))
        ranking_data.sort(key=lambda x: x[1])  # 按阻塞率排序

        print(f"{'排名':<4} {'算法':<25} {'阻塞率':<12} {'频谱利用率':<12} {'平均跳数':<10} {'吞吐量':<10}")
        print("-" * 80)

        for i, (alg, blocking, spectrum, hops, throughput) in enumerate(ranking_data, 1):
            print(f"{i:<4} {alg:<25} {blocking:<12.6f} {spectrum:<12.4f} {hops:<10.2f} {throughput:<10.0f}")

        print("=" * 80)
        print(f"最优算法: {ranking_data[0][0]} (阻塞率: {ranking_data[0][1]:.6f})")
        print(f"最差算法: {ranking_data[-1][0]} (阻塞率: {ranking_data[-1][1]:.6f})")
        performance_gap = (ranking_data[-1][1] - ranking_data[0][1]) / ranking_data[-1][1] * 100
        print(f"性能差距: {performance_gap:.1f}%")

    def _generate_chart_summary(self, performance_data: Dict[str, pd.DataFrame]):
        """生成图表创建总结"""
        metrics = ['blocking_rate', 'spectrum_utilization', 'fragmentation_rate',
                  'average_hop_count', 'computational_overhead', 'throughput']

        print(f"\n📊 图表生成总结:")
        print(f"   算法数量: {len(performance_data)} 种")
        print(f"   性能指标: {len(metrics)} 个")
        print(f"   生成图表:")

        for i, metric in enumerate(metrics, 1):
            filename = f"performance_{metric}.png"
            print(f"     {i}. {filename}")

        print(f"   图表保存位置: {self.log_dir}/")

        # 验证计算开销图表
        if 'computational_overhead' in metrics:
            print(f"   ✅ 计算开销图表已包含 (performance_computational_overhead.png)")
        else:
            print(f"   ❌ 计算开销图表缺失")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("🔍 开始完整数据分析...")
        
        if not self.algorithms:
            print("❌ 没有发现可分析的算法数据")
            return
        
        # 1. 分析性能指标
        performance_data = self.analyze_performance_metrics()
        
        # 2. 分析频谱利用率
        spectrum_analysis = self.analyze_spectrum_utilization()
        
        # 3. 分析请求决策
        decision_analysis = self.analyze_request_decisions()
        
        # 4. 生成对比报告
        if performance_data:
            self.generate_comparison_report(performance_data)
            self.create_visualization_plots(performance_data)
        
        # 5. 保存完整分析结果
        self.analysis_results = {
            'performance_data': {alg: df.to_dict() for alg, df in performance_data.items()},
            'spectrum_analysis': spectrum_analysis,
            'decision_analysis': decision_analysis
        }
        
        analysis_file = os.path.join(self.log_dir, "full_analysis_results.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 完整分析结果已保存: {analysis_file}")
        print(f"\n🎉 数据分析完成！")


if __name__ == "__main__":
    # 运行数据分析
    analyzer = DataAnalyzer()
    analyzer.run_full_analysis()
