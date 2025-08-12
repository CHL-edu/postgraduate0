#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数变化历史分析工具
专门用于分析NSFNet_with_logging.py运行产生的参数变化历史
生成CSV表格和可视化图表
"""

import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from config_reader import ConfigReader
from parameter_change_manager import ParameterChangeManager


def load_parameter_history(history_file="log/parameter_change_history.json"):
    """加载参数变化历史文件"""
    if not os.path.exists(history_file):
        print(f"❌ 参数变化历史文件不存在: {history_file}")
        print("请先运行 NSFNet_with_logging.py 生成参数变化历史")
        return None
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            print("⚠️  参数变化历史为空")
            return None
        
        print(f"✅ 成功加载参数变化历史，共 {len(history)} 条记录")
        print(f"   时隙范围: {history[0]['timeslot']} - {history[-1]['timeslot']}")
        return history
    
    except Exception as e:
        print(f"❌ 加载参数变化历史失败: {e}")
        return None


def export_to_csv(history, output_file="log/parameter_change_trends.csv"):
    """导出参数变化历史到CSV文件"""
    if not history:
        print("⚠️  没有数据可导出")
        return False
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 转换为DataFrame以便更好地处理
        df = pd.DataFrame(history)

        # 添加一些计算列
        df['change_factor_abs'] = df['change_factor'].abs()
        df['spectrum_utilization_ratio'] = df['test_num_spectrum_slots'] / df['test_num_spectrum_slots'].max()

        # Lambda相关计算列
        df['lambda_value'] = df['requests_per_timeslot']  # 明确标识Lambda
        df['lambda_change_rate'] = df['lambda_value'].diff().fillna(0)  # Lambda变化率
        df['lambda_normalized'] = (df['lambda_value'] - df['lambda_value'].min()) / (df['lambda_value'].max() - df['lambda_value'].min()) if df['lambda_value'].max() != df['lambda_value'].min() else 0

        # 负载强度分类
        lambda_mean = df['lambda_value'].mean()
        df['load_intensity'] = df['lambda_value'].apply(lambda x: 'High' if x > lambda_mean * 1.2 else ('Low' if x < lambda_mean * 0.8 else 'Medium'))
        
        # 保存到CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ 参数变化历史已导出到CSV: {output_file}")
        
        # 显示基本统计信息
        print(f"\n📊 数据统计:")
        print(f"   总记录数: {len(df)}")
        print(f"   时隙范围: {df['timeslot'].min()} - {df['timeslot'].max()}")
        print(f"   变化因子范围: {df['change_factor'].min():.3f} - {df['change_factor'].max():.3f}")
        
        return True
    
    except Exception as e:
        print(f"❌ 导出CSV失败: {e}")
        return False


def analyze_trends(history):
    """分析参数变化趋势"""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    print(f"\n=== 参数变化趋势分析 ===")
    
    # 基本信息
    print(f"📊 基本信息:")
    print(f"   时隙数量: {len(df)}")
    print(f"   时隙范围: {df['timeslot'].min()} - {df['timeslot'].max()}")
    
    # 变化因子分析
    print(f"\n🔄 变化因子分析:")
    print(f"   范围: {df['change_factor'].min():.3f} - {df['change_factor'].max():.3f}")
    print(f"   均值: {df['change_factor'].mean():.3f}")
    print(f"   标准差: {df['change_factor'].std():.3f}")
    
    # Lambda参数专门分析
    print(f"\n🎯 Lambda参数 (requests_per_timeslot) 专门分析:")
    if 'requests_per_timeslot' in df.columns:
        lambda_data = df['requests_per_timeslot']
        lambda_min = lambda_data.min()
        lambda_max = lambda_data.max()
        lambda_mean = lambda_data.mean()
        lambda_std = lambda_data.std()
        lambda_variation = (lambda_max - lambda_min) / lambda_mean * 100 if lambda_mean != 0 else 0

        print(f"   Lambda范围: [{lambda_min:.3f}, {lambda_max:.3f}]")
        print(f"   Lambda均值: {lambda_mean:.3f} (泊松分布参数)")
        print(f"   Lambda标准差: {lambda_std:.3f}")
        print(f"   Lambda变化幅度: {lambda_variation:.1f}%")

        # 计算Lambda变化率
        lambda_change_rate = lambda_data.diff().fillna(0)
        max_increase = lambda_change_rate.max()
        max_decrease = lambda_change_rate.min()
        print(f"   最大增长率: {max_increase:.3f} 请求/时隙")
        print(f"   最大下降率: {max_decrease:.3f} 请求/时隙")

        # Lambda稳定性分析
        cv = lambda_std / lambda_mean if lambda_mean != 0 else 0  # 变异系数
        print(f"   变异系数: {cv:.3f} (值越小越稳定)")

    # 各参数的变化范围
    numeric_columns = ['small_bandwidth_probability', 'duration_mean',
                      'requests_per_timeslot', 'test_num_spectrum_slots']

    print(f"\n📈 所有参数变化范围:")
    for col in numeric_columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            std_val = df[col].std()
            variation = (max_val - min_val) / mean_val * 100 if mean_val != 0 else 0

            # 为Lambda参数添加特殊标识
            param_name = col
            if col == 'requests_per_timeslot':
                param_name = f"{col} (Lambda参数)"

            print(f"   {param_name}:")
            print(f"     范围: [{min_val:.3f}, {max_val:.3f}]")
            print(f"     均值: {mean_val:.3f}, 标准差: {std_val:.3f}")
            print(f"     变化幅度: {variation:.1f}%")
    
    return df


def create_comprehensive_visualization(history, output_file=None):
    """创建综合的参数变化可视化"""
    if not history:
        print("⚠️  没有数据可可视化")
        return False
    
    df = pd.DataFrame(history)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建大型图表
    fig = plt.figure(figsize=(24, 20))

    # 主标题
    fig.suptitle('NSFNet Parameter Change Analysis - Comprehensive View', fontsize=20, fontweight='bold')

    # 创建子图布局 - 增加一行来容纳更多图表
    gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)
    
    timeslots = df['timeslot']
    
    # 1. 变化因子时间序列
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timeslots, df['change_factor'], 'b-', linewidth=2, alpha=0.8)
    ax1.fill_between(timeslots, df['change_factor'], alpha=0.3, color='blue')
    ax1.set_title('Change Factor Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Timeslot')
    ax1.set_ylabel('Change Factor')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 2. 小带宽请求概率
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(timeslots, df['small_bandwidth_probability'], 'r-', linewidth=2, marker='o', markersize=1)
    ax2.set_title('Small Bandwidth Probability')
    ax2.set_xlabel('Timeslot')
    ax2.set_ylabel('Probability')
    ax2.grid(True, alpha=0.3)
    
    # 3. 请求持续时间均值
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(timeslots, df['duration_mean'], 'g-', linewidth=2, marker='s', markersize=1)
    ax3.set_title('Duration Mean')
    ax3.set_xlabel('Timeslot')
    ax3.set_ylabel('Duration (timeslots)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 每时隙请求数量 (Lambda参数)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(timeslots, df['requests_per_timeslot'], 'm-', linewidth=2, marker='^', markersize=1)
    ax4.set_title('Lambda (Requests per Timeslot)')
    ax4.set_xlabel('Timeslot')
    ax4.set_ylabel('Lambda (Poisson Parameter)')
    ax4.grid(True, alpha=0.3)
    # 添加平均值线
    mean_lambda = df['requests_per_timeslot'].mean()
    ax4.axhline(y=mean_lambda, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_lambda:.2f}')
    ax4.legend()
    
    # 5. 频谱槽数量
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(timeslots, df['test_num_spectrum_slots'], 'c-', linewidth=2, marker='d', markersize=1)
    ax5.set_title('Spectrum Slots')
    ax5.set_xlabel('Timeslot')
    ax5.set_ylabel('Number of Slots')
    ax5.grid(True, alpha=0.3)
    
    # 6. 变化因子分布直方图
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(df['change_factor'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax6.set_title('Change Factor Distribution')
    ax6.set_xlabel('Change Factor')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    # 7. 参数相关性热图
    ax7 = fig.add_subplot(gs[2, 2])
    numeric_cols = ['change_factor', 'small_bandwidth_probability', 'duration_mean', 
                   'requests_per_timeslot', 'test_num_spectrum_slots']
    corr_matrix = df[numeric_cols].corr()
    im = ax7.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax7.set_xticks(range(len(numeric_cols)))
    ax7.set_yticks(range(len(numeric_cols)))
    ax7.set_xticklabels([col.replace('_', '\n') for col in numeric_cols], rotation=45, ha='right')
    ax7.set_yticklabels([col.replace('_', '\n') for col in numeric_cols])
    ax7.set_title('Parameter Correlation Matrix')
    plt.colorbar(im, ax=ax7, shrink=0.8)
    
    # 8. Lambda变化详细分析
    ax8 = fig.add_subplot(gs[3, 0])
    ax8.plot(timeslots, df['requests_per_timeslot'], 'm-', linewidth=3, alpha=0.8)
    ax8.fill_between(timeslots, df['requests_per_timeslot'], alpha=0.3, color='magenta')
    ax8.set_title('Lambda Parameter Detailed View', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Timeslot')
    ax8.set_ylabel('Lambda (Requests/Timeslot)')
    ax8.grid(True, alpha=0.3)
    # 添加统计信息
    lambda_min = df['requests_per_timeslot'].min()
    lambda_max = df['requests_per_timeslot'].max()
    lambda_mean = df['requests_per_timeslot'].mean()
    ax8.text(0.02, 0.98, f'Min: {lambda_min:.2f}\nMax: {lambda_max:.2f}\nMean: {lambda_mean:.2f}',
             transform=ax8.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 9. Lambda变化率分析
    ax9 = fig.add_subplot(gs[3, 1])
    lambda_change_rate = df['requests_per_timeslot'].diff().fillna(0)
    ax9.plot(timeslots[1:], lambda_change_rate[1:], 'orange', linewidth=2, marker='o', markersize=1)
    ax9.set_title('Lambda Change Rate')
    ax9.set_xlabel('Timeslot')
    ax9.set_ylabel('Change Rate (Δλ)')
    ax9.grid(True, alpha=0.3)
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # 10. Lambda分布直方图
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.hist(df['requests_per_timeslot'], bins=20, alpha=0.7, color='magenta', edgecolor='black')
    ax10.set_title('Lambda Distribution')
    ax10.set_xlabel('Lambda Value')
    ax10.set_ylabel('Frequency')
    ax10.grid(True, alpha=0.3)
    ax10.axvline(x=lambda_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {lambda_mean:.2f}')
    ax10.legend()

    # 11. 多参数对比（归一化）
    ax11 = fig.add_subplot(gs[4, :])

    # 归一化所有参数到0-1范围
    normalized_data = {}
    for col in numeric_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                normalized_data[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                normalized_data[col] = df[col] * 0  # 如果没有变化，设为0

    colors = ['blue', 'red', 'green', 'magenta', 'cyan']
    for i, (col, data) in enumerate(normalized_data.items()):
        display_name = col.replace('_', ' ').title()
        if col == 'requests_per_timeslot':
            display_name = 'Lambda (Requests/Timeslot)'
        ax11.plot(timeslots, data, color=colors[i % len(colors)],
                linewidth=2, label=display_name, alpha=0.8)

    ax11.set_title('Normalized Parameter Comparison (All Parameters)', fontsize=14, fontweight='bold')
    ax11.set_xlabel('Timeslot')
    ax11.set_ylabel('Normalized Value (0-1)')
    ax11.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax11.grid(True, alpha=0.3)
    
    # 保存图表
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"log/parameter_analysis_comprehensive_{timestamp}.png"
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 综合参数分析图表已保存到: {output_file}")
    
    plt.show()
    return True


def generate_summary_report(history, output_file="log/parameter_change_summary.txt"):
    """生成参数变化总结报告"""
    if not history:
        return False
    
    df = pd.DataFrame(history)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("NSFNet 参数变化分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本信息
            f.write("1. 基本信息\n")
            f.write("-" * 20 + "\n")
            f.write(f"总时隙数: {len(df)}\n")
            f.write(f"时隙范围: {df['timeslot'].min()} - {df['timeslot'].max()}\n")
            f.write(f"变化因子范围: {df['change_factor'].min():.3f} - {df['change_factor'].max():.3f}\n\n")
            
            # Lambda参数专门分析
            f.write("2. Lambda参数专门分析\n")
            f.write("-" * 20 + "\n")
            if 'requests_per_timeslot' in df.columns:
                lambda_data = df['requests_per_timeslot']
                lambda_change_rate = lambda_data.diff().fillna(0)

                f.write("Lambda (requests_per_timeslot) - 泊松分布参数:\n")
                f.write(f"  最小值: {lambda_data.min():.3f}\n")
                f.write(f"  最大值: {lambda_data.max():.3f}\n")
                f.write(f"  均值: {lambda_data.mean():.3f}\n")
                f.write(f"  标准差: {lambda_data.std():.3f}\n")
                f.write(f"  变异系数: {lambda_data.std()/lambda_data.mean():.3f}\n")
                f.write(f"  最大增长率: {lambda_change_rate.max():.3f} 请求/时隙\n")
                f.write(f"  最大下降率: {lambda_change_rate.min():.3f} 请求/时隙\n")
                variation = (lambda_data.max() - lambda_data.min()) / lambda_data.mean() * 100 if lambda_data.mean() != 0 else 0
                f.write(f"  变化幅度: {variation:.1f}%\n")

                # 负载强度分布
                lambda_mean = lambda_data.mean()
                high_load = (lambda_data > lambda_mean * 1.2).sum()
                low_load = (lambda_data < lambda_mean * 0.8).sum()
                medium_load = len(lambda_data) - high_load - low_load
                f.write(f"  高负载时隙数 (>1.2×均值): {high_load}\n")
                f.write(f"  中等负载时隙数: {medium_load}\n")
                f.write(f"  低负载时隙数 (<0.8×均值): {low_load}\n")

            # 其他参数统计
            f.write("\n3. 其他参数统计\n")
            f.write("-" * 20 + "\n")
            numeric_columns = ['small_bandwidth_probability', 'duration_mean',
                              'test_num_spectrum_slots']

            for col in numeric_columns:
                if col in df.columns:
                    f.write(f"\n{col}:\n")
                    f.write(f"  最小值: {df[col].min():.3f}\n")
                    f.write(f"  最大值: {df[col].max():.3f}\n")
                    f.write(f"  均值: {df[col].mean():.3f}\n")
                    f.write(f"  标准差: {df[col].std():.3f}\n")
                    variation = (df[col].max() - df[col].min()) / df[col].mean() * 100 if df[col].mean() != 0 else 0
                    f.write(f"  变化幅度: {variation:.1f}%\n")
            
            # 配置信息
            f.write("\n4. 配置信息\n")
            f.write("-" * 20 + "\n")
            try:
                config_reader = ConfigReader()
                change_config = config_reader.get_parameter_change_config()
                for key, value in change_config.items():
                    f.write(f"{key}: {value}\n")
            except:
                f.write("无法读取配置信息\n")
        
        print(f"📋 参数变化总结报告已保存到: {output_file}")
        return True
    
    except Exception as e:
        print(f"❌ 生成报告失败: {e}")
        return False


def main():
    """主函数"""
    print("🔍 NSFNet 参数变化历史分析工具")
    print("=" * 50)
    
    # 1. 加载数据
    history = load_parameter_history()
    if not history:
        print("❌ 无法加载参数变化历史，程序退出")
        return
    
    # 2. 导出CSV
    print("\n📊 导出CSV数据...")
    export_to_csv(history)
    
    # 3. 分析趋势
    print("\n📈 分析参数变化趋势...")
    analyze_trends(history)
    
    # 4. 创建可视化
    print("\n🎨 创建可视化图表...")
    create_comprehensive_visualization(history)
    
    # 5. 生成报告
    print("\n📋 生成总结报告...")
    generate_summary_report(history)
    
    print("\n✅ 分析完成！请查看log目录下的输出文件。")

    print("\n📊 生成的图表说明:")
    print("=" * 60)

    print("\n🔍 综合分析图表包含以下11个子图:")
    print("\n1️⃣  变化因子时间序列 (Change Factor Over Time)")
    print("   - 显示参数变化的驱动因子随时间的变化")
    print("   - 范围: -1到1，决定所有参数的变化方向和幅度")
    print("   - 蓝色填充区域表示变化因子的波动范围")

    print("\n2️⃣  小带宽请求概率 (Small Bandwidth Probability)")
    print("   - 显示小带宽请求在总请求中的比例变化")
    print("   - 影响网络负载分布和频谱利用效率")
    print("   - 红色圆点标记显示具体数值")

    print("\n3️⃣  请求持续时间均值 (Duration Mean)")
    print("   - 显示请求持续时间的平均值变化")
    print("   - 影响频谱资源的占用时长和释放频率")
    print("   - 绿色方块标记显示变化趋势")

    print("\n4️⃣  Lambda参数 (Lambda - Requests per Timeslot)")
    print("   - 显示泊松分布的lambda参数变化")
    print("   - 决定每个时隙到达的请求数量")
    print("   - 紫色三角标记，红色虚线显示平均值")
    print("   - 这是网络负载强度的核心指标")

    print("\n5️⃣  频谱槽数量 (Spectrum Slots)")
    print("   - 显示可用频谱槽总数的变化")
    print("   - 影响网络的总容量和承载能力")
    print("   - 青色菱形标记显示容量变化")

    print("\n6️⃣  变化因子分布 (Change Factor Distribution)")
    print("   - 变化因子的频率分布直方图")
    print("   - 显示参数变化的统计特性")
    print("   - 橙色柱状图，可观察变化的对称性")

    print("\n7️⃣  参数相关性矩阵 (Parameter Correlation Matrix)")
    print("   - 显示各参数之间的相关性")
    print("   - 颜色从蓝色(负相关)到红色(正相关)")
    print("   - 帮助理解参数间的相互影响关系")

    print("\n8️⃣  Lambda详细视图 (Lambda Parameter Detailed View)")
    print("   - Lambda参数的详细时间序列分析")
    print("   - 紫色填充显示变化范围")
    print("   - 左上角显示最小值、最大值和平均值")

    print("\n9️⃣  Lambda变化率 (Lambda Change Rate)")
    print("   - 显示Lambda参数的变化速率(导数)")
    print("   - 橙色圆点显示每个时隙的变化量")
    print("   - 黑色虚线表示零变化率基准线")

    print("\n🔟 Lambda分布直方图 (Lambda Distribution)")
    print("   - Lambda值的频率分布")
    print("   - 紫色柱状图显示数值分布")
    print("   - 红色虚线标记平均值位置")

    print("\n1️⃣1️⃣ 归一化参数对比 (Normalized Parameter Comparison)")
    print("   - 所有参数归一化到0-1范围的对比")
    print("   - 不同颜色线条代表不同参数")
    print("   - 便于观察各参数的相对变化模式")

    print("\n📁 输出文件说明:")
    print("=" * 60)
    print("📄 parameter_change_trends.csv")
    print("   - 包含所有参数的时间序列数据")
    print("   - 可用Excel等工具进一步分析")
    print("   - 包含计算列如变化幅度、利用率等")

    print("\n📄 parameter_change_summary.txt")
    print("   - 文本格式的分析报告")
    print("   - 包含统计摘要和配置信息")
    print("   - 便于存档和分享")

    print("\n🖼️  parameter_analysis_comprehensive_*.png")
    print("   - 包含所有11个子图的综合分析图表")
    print("   - 高分辨率(300 DPI)，适合论文和报告")
    print("   - 文件名包含时间戳，避免覆盖")

    print("\n💡 分析建议:")
    print("=" * 60)
    print("🔍 观察要点:")
    print("   1. Lambda参数的变化直接影响网络负载")
    print("   2. 小带宽概率影响频谱碎片化程度")
    print("   3. 持续时间影响资源占用模式")
    print("   4. 频谱槽数量决定网络总容量")
    print("   5. 参数间的相关性反映系统耦合度")

    print("\n📈 性能影响:")
    print("   - Lambda增大 → 阻塞率可能上升")
    print("   - 小带宽比例高 → 频谱利用率可能提升")
    print("   - 持续时间长 → 资源周转率下降")
    print("   - 频谱槽多 → 系统容量增加")

    print("\n🎯 应用场景:")
    print("   - 算法性能评估：观察不同负载下的表现")
    print("   - 系统优化：识别性能瓶颈和改进点")
    print("   - 容量规划：预测不同参数下的需求")
    print("   - 研究分析：理解参数变化对系统的影响")


if __name__ == "__main__":
    main()
