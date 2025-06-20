import networkx as nx
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
import json
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator


# 配置参数类
class Config:
    """
    存储模拟的所有配置参数。
    可以通过配置文件加载，或者使用默认值。
    """

    def __init__(self, config_file: Optional[str] = None):
        # 结果文件和图片备份路径
        self.business_analysis_path = "/data/chl/pyproject/backup/RSA/rsa_results/business_analysis"  # 业务分析结果保存路径
        self.imgbackup_path = "/data/chl/pyproject/backup/RSA/rsa_results/RSAimg"  # 模拟结果图片保存路径
        self.allocation_results_path = "/data/chl/pyproject/backup/RSA/rsa_results/allocation_results"  # 分配结果数据保存路径

        # 网络拓扑参数
        self.num_nodes = 14  # 网络中的节点数量
        self.num_links = 21  # 网络中的链路数量
        self.num_slots_per_link = 10  # 每条链路的频谱槽数量
        self.min_link_length = 100  # 链路的最小长度（用于K短路径计算）
        self.max_link_length = 1000  # 链路的最大长度

        # RSA算法参数
        self.default_k_paths = 3  # 计算K短路径时的K值

        # 业务请求生成参数（泊松过程和指数分布）
        # 场景A：轻负载 + 短业务持续时间 (默认配置)
        self.lambda_rate = 20  # 业务到达率 (λ) - 每秒平均到达的业务请求数
        self.mu_rate = 0.5  # 业务离开率 (μ) - 业务持续时间的倒数，平均业务持续时间为 1/μ 秒
        self.simulation_time = 100  # 模拟总时长 (秒)

        # 示例：其他负载场景的配置，当前被注释掉
        """
        # 场景B：中等负载 + 长业务持续时间
        self.lambda_rate = 40
        self.mu_rate = 0.1  # 平均业务持续10秒
        self.simulation_time = 300  # 延长模拟时间以达到稳态

        # 场景C：重负载 + 混合业务持续时间
        self.lambda_rate = 60
        self.mu_rate = 0.2  # 平均业务持续5秒
        self.simulation_time = 200
        """
        # ρ = λ/（μ*C） 用于计算网络负载，C为网络总容量（链路数*每链路槽数）

        # 如果指定了配置文件，则从文件中加载配置
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    for key, value in config_data.items():
                        # 仅更新Config类中已存在的属性
                        if hasattr(self, key):
                            setattr(self, key, value)
                print(f"配置已从 {config_file} 加载。")
            except json.JSONDecodeError:
                print(f"错误: 无法解析配置文件 {config_file}，请检查JSON格式。")
            except Exception as e:
                print(f"加载配置文件时发生未知错误: {e}")

        # 确保图片备份目录存在
        os.makedirs(self.imgbackup_path, exist_ok=True)
        # 确保分配结果目录存在
        os.makedirs(self.allocation_results_path, exist_ok=True)
        # 确保业务分析目录存在
        os.makedirs(self.business_analysis_path, exist_ok=True)


# 生成光网络拓扑
def generate_topology(config: Config, topology_type: str = "random") -> nx.Graph:
    """
    根据配置生成光网络拓扑。
    支持NSFNET固定拓扑和随机拓扑。

    Args:
        config (Config): 配置对象，包含节点数、链路数、槽数、链路长度范围等。
        topology_type (str): 拓扑类型，可以是"nsfnet"或"random"。

    Returns:
        nx.Graph: 生成的网络拓扑图。
    """
    G = nx.Graph()

    if topology_type == "nsfnet":
        # NSFNET 拓扑：14个节点，21条链路
        nodes = list(range(14))
        edges = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
            (4, 5), (4, 6), (5, 7), (5, 8), (6, 9), (7, 10), (7, 11), (8, 9),
            (8, 11), (9, 10), (9, 12), (11, 12), (12, 13)
        ]
        if config.num_nodes != 14 or config.num_links != 21:
            print("警告: NSFNET拓扑固定为14节点和21链路，将忽略配置中的节点数和链路数。")
        config.num_nodes = 14
        config.num_links = 21

        for i in nodes:
            G.add_node(i, label=f'N{i}')
        for u, v in edges:
            G.add_edge(u, v,
                       length=random.randint(config.min_link_length, config.max_link_length),
                       slots=[0] * config.num_slots_per_link,  # 0表示空闲，1表示占用
                       active_connections=[])  # 存储当前链路上活跃的连接信息
    else:
        # 随机拓扑生成
        if config.num_links < config.num_nodes - 1:
            raise ValueError("随机拓扑中链路数不能少于节点数减一（确保连通性）。")
        if config.num_links > config.num_nodes * (config.num_nodes - 1) / 2:
            raise ValueError("链路数不能超过最大可能链路数。")

        for i in range(config.num_nodes):
            G.add_node(i, label=f'N{i}')

        edges = set()
        # 确保拓扑是连通的（生成一个随机树）
        nodes_list = list(range(config.num_nodes))
        random.shuffle(nodes_list)
        for i in range(1, config.num_nodes):
            parent = random.randint(0, i - 1)
            u, v = sorted((nodes_list[parent], nodes_list[i]))  # 保证边的表示一致性
            edges.add((u, v))

        # 添加剩余的随机链路直到达到num_links
        remaining_edges = config.num_links - len(edges)
        if remaining_edges > 0:
            potential_edges = []
            for i in range(config.num_nodes):
                for j in range(i + 1, config.num_nodes):
                    if (i, j) not in edges:
                        potential_edges.append((i, j))

            # 随机选择附加的链路，避免重复
            try:
                additional_edges = random.sample(potential_edges, min(remaining_edges, len(potential_edges)))
                edges.update(additional_edges)
            except ValueError:
                print(f"警告: 无法生成 {config.num_links} 条链路，因为可能的链路数量不足。实际生成 {len(edges)} 条链路。")

        for u, v in edges:
            G.add_edge(u, v,
                       length=random.randint(config.min_link_length, config.max_link_length),
                       slots=[0] * config.num_slots_per_link,
                       active_connections=[])
    return G


# 计算K最短路径
def k_shortest_paths(G: nx.Graph, source: int, target: int, k: int = 3) -> List[List[int]]:
    """
    计算源节点到目标节点的K条最短路径。

    Args:
        G (nx.Graph): 网络拓扑图。
        source (int): 源节点。
        target (int): 目标节点。
        k (int): 要计算的最短路径数量。

    Returns:
        List[List[int]]: 包含K条最短路径的列表，每条路径表示为节点列表。
                         如果没有路径，则返回空列表。
    """
    try:
        # 使用NetworkX的shortest_simple_paths生成器，按长度排序
        # weight='length' 确保是根据链路长度计算最短路径
        paths = list(nx.shortest_simple_paths(G, source, target, weight='length'))
        return paths[:k]
    except nx.NetworkXNoPath:
        # 如果没有路径，捕获异常并返回空列表
        return []
    except Exception as e:
        print(f"计算K最短路径时发生错误: {e}")
        return []


# 查找连续空闲频谱槽
def find_continuous_slots(slots_status: List[int], num_required: int) -> Optional[Tuple[int, int]]:
    """
    在给定链路的频谱槽状态中查找连续的空闲槽。

    Args:
        slots_status (List[int]): 链路的频谱槽状态列表 (0空闲, 1占用)。
        num_required (int): 需要的连续槽数量。

    Returns:
        Optional[Tuple[int, int]]: 如果找到，返回 (起始槽索引, 结束槽索引) 的元组；否则返回 None。
    """
    count = 0
    start_index = 0
    for i, slot in enumerate(slots_status):
        if slot == 0:  # 如果当前槽空闲
            if count == 0:
                start_index = i  # 记录连续空闲槽的起始位置
            count += 1
            if count >= num_required:
                return start_index, start_index + count - 1  # 找到足够多的连续槽
        else:  # 如果当前槽被占用，中断连续计数
            count = 0
    return None  # 没有找到足够的连续空闲槽


# 频谱分配算法
def spectrum_allocation(G: nx.Graph, path: List[int], num_slots: int,
                        start_time: float, duration: float,
                        algorithm: str = "first_fit") -> Optional[Tuple[int, int]]:
    """
    在给定路径上尝试分配连续的频谱槽。

    Args:
        G (nx.Graph): 网络拓扑图。
        path (List[int]): 待分配频谱的路径 (节点列表)。
        num_slots (int): 需要分配的频谱槽数量。
        start_time (float): 连接的开始时间。
        duration (float): 连接的持续时间。
        algorithm (str): 频谱分配算法 ("first_fit", "best_fit", "random_fit")。

    Returns:
        Optional[Tuple[int, int]]: 如果分配成功，返回 (起始槽索引, 结束槽索引) 的元组；否则返回 None。
    """
    if not path or len(path) < 2:
        return None

    # 获取路径上所有链路的公共可用频谱槽状态
    # 初始化为第一条链路的槽状态
    common_available_slots = G[path[0]][path[1]]['slots'].copy()

    # 遍历路径上的其他链路，进行位与操作，找出所有链路都空闲的槽
    for i in range(1, len(path) - 1):
        u, v = path[i], path[i + 1]
        link_slots = G[u][v]['slots']
        # 将common_available_slots和link_slots进行按位逻辑与操作
        # 如果某个槽在两条链路中都为0 (空闲)，则结果为0，否则为1 (占用)
        common_available_slots = [1 if common_available_slots[j] == 1 or link_slots[j] == 1 else 0
                                  for j in range(len(common_available_slots))]

    allocation = None
    if algorithm == "first_fit":
        allocation = find_continuous_slots(common_available_slots, num_slots)
    elif algorithm == "best_fit":
        best_allocation = None
        min_size = float('inf')  # 记录找到的最小连续空闲块的大小

        # 遍历所有可能的连续空闲块
        current_count = 0
        current_start = 0
        for i, slot_status in enumerate(common_available_slots):
            if slot_status == 0:
                if current_count == 0:
                    current_start = i
                current_count += 1
            else:
                if current_count >= num_slots:  # 如果当前块满足需求
                    if current_count < min_size:  # 并且比之前找到的块更小
                        best_allocation = (current_start, current_start + current_count - 1)
                        min_size = current_count
                current_count = 0  # 重置计数器

        # 检查最后一个块（循环结束后）
        if current_count >= num_slots:
            if current_count < min_size:
                best_allocation = (current_start, current_start + current_count - 1)

        allocation = best_allocation

    elif algorithm == "random_fit":
        possible_allocations = []
        current_count = 0
        current_start = 0
        for i, slot_status in enumerate(common_available_slots):
            if slot_status == 0:
                if current_count == 0:
                    current_start = i
                current_count += 1
                if current_count >= num_slots:
                    # 找到一个满足条件的连续块
                    possible_allocations.append((current_start, current_start + num_slots - 1))
            else:
                current_count = 0

        if possible_allocations:
            # 从所有可能的连续块中随机选择一个
            allocation = random.choice(possible_allocations)
        else:
            allocation = None  # 没有找到任何可行的分配

    else:
        raise ValueError(f"不支持的频谱分配算法: {algorithm}")

    # 如果找到合适的频谱槽，则在所有相关链路上进行分配
    if allocation:
        start_slot_idx, end_slot_idx = allocation
        connection_info = {
            'start_time': start_time,
            'end_time': start_time + duration,
            'slots': (start_slot_idx, end_slot_idx)
        }

        # 遍历路径上的每条链路，更新其槽状态和活跃连接列表
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # 添加连接信息到链路的活跃连接列表
            G[u][v]['active_connections'].append(connection_info)
            # 将对应频谱槽标记为占用
            for j in range(start_slot_idx, end_slot_idx + 1):
                G[u][v]['slots'][j] = 1
        return start_slot_idx, end_slot_idx
    return None  # 未能成功分配


# 生成动态业务请求（泊松过程模拟业务到达，指数分布模拟业务持续时间）
def generate_dynamic_demands(config: Config) -> List[Tuple[int, int, int, float, float]]:
    """
    根据泊松过程生成动态业务请求。

    Args:
        config (Config): 配置对象，包含到达率、离开率和模拟时间。

    Returns:
        List[Tuple[int, int, int, float, float]]: 业务请求列表，每个请求包含
                                                   (源节点, 目标节点, 所需频谱槽数, 到达时间, 持续时间)。
    """
    demands = []
    current_time = 0.0  # 模拟当前时间

    while current_time < config.simulation_time:
        # 泊松过程：业务请求之间的间隔时间服从指数分布
        # lambda_rate 是平均到达率，所以期望间隔是 1/lambda_rate
        inter_arrival_time = np.random.exponential(1 / config.lambda_rate)
        current_time += inter_arrival_time

        if current_time >= config.simulation_time:
            break

        # 随机选择源和目标节点
        src = random.randint(0, config.num_nodes - 1)
        dst = random.randint(0, config.num_nodes - 1)
        while dst == src:  # 确保源和目标不同
            dst = random.randint(0, config.num_nodes - 1)

        # 随机选择所需的频谱槽数量（例如1到5个）
        num_required_slots = random.randint(1, 5)

        # 业务持续时间服从指数分布
        # mu_rate 是离开率，所以期望持续时间是 1/mu_rate
        duration = np.random.exponential(1 / config.mu_rate)

        demands.append((src, dst, num_required_slots, current_time, duration))

    # 按照到达时间对业务请求进行排序，以便按时间顺序处理
    return sorted(demands, key=lambda x: x[3])


# 动态RSA算法核心逻辑
def dynamic_rsa_algorithm(G: nx.Graph, demands: List[Tuple[int, int, int, float, float]],
                          config: Config, spectrum_algorithm: str = "first_fit") -> List[Dict]:
    """
    实现动态R(outing)S(pectrum)A(llocation)算法。
    处理一系列按时间排序的业务请求，包括连接的建立和释放。

    Args:
        G (nx.Graph): 网络拓扑图。
        demands (List[Tuple]): 业务请求列表。
        config (Config): 配置对象。
        spectrum_algorithm (str): 频谱分配算法名称。

    Returns:
        List[Dict]: 包含每个业务请求分配结果的列表。
    """
    results = []

    # 使用一个列表来跟踪所有活跃的连接，以便按结束时间排序并释放
    # connection_end_events: List[Tuple[float, Dict]] = [] # (end_time, connection_info)

    # 遍历按到达时间排序的业务请求
    for src, dst, slots_needed, arrival_time, duration in demands:
        # 在处理新业务请求之前，首先释放所有已到期的连接
        # 遍历所有边，检查其活跃连接
        for u, v in G.edges():
            expired_connections_indices = []
            for i, conn in enumerate(G[u][v]['active_connections']):
                # 如果连接的结束时间小于或等于当前业务请求的到达时间，则认为其已到期
                if conn['end_time'] <= arrival_time:
                    expired_connections_indices.append(i)
                    # 释放该连接占用的频谱槽
                    start_slot, end_slot = conn['slots']
                    for j in range(start_slot, end_slot + 1):
                        G[u][v]['slots'][j] = 0  # 标记为0（空闲）

            # 移除已到期的连接
            G[u][v]['active_connections'] = [conn for i, conn in enumerate(G[u][v]['active_connections'])
                                             if i not in expired_connections_indices]

        # 路由阶段：计算K最短路径
        paths = k_shortest_paths(G, src, dst, k=config.default_k_paths)

        allocated = False
        selected_path = None
        allocated_spectrum_range = None

        # 频谱分配阶段：尝试在K条路径上进行频谱分配
        for path in paths:
            spectrum_range = spectrum_allocation(G, path, slots_needed, arrival_time, duration,
                                                 algorithm=spectrum_algorithm)
            if spectrum_range:
                # 成功分配，记录结果并退出循环
                selected_path = path
                allocated_spectrum_range = spectrum_range
                allocated = True
                break  # 找到第一个可行的分配就停止

        # 记录业务请求的处理结果
        results.append({
            'source': src,
            'destination': dst,
            'path': selected_path,  # 如果分配成功，记录路径；否则为None
            'spectrum': allocated_spectrum_range,  # 如果分配成功，记录频谱范围；否则为None
            'slots_required': slots_needed,
            'arrival_time': arrival_time,
            'duration': duration,
            'end_time': arrival_time + duration,
            'status': 'Allocated' if allocated else 'Blocked'  # 标记为“已分配”或“被阻塞”
        })

    return results


# 可视化模拟结果
def visualize_results(G: nx.Graph, rsa_results: List[Dict], config: Config,
                      show_spectrum: bool = True, show_paths: bool = True,
                      show_timeline: bool = True) -> None:
    """
    可视化模拟结果，包括网络拓扑、平均频谱占用率和连接时间轴。

    Args:
        G (nx.Graph): 最终的网络拓扑图。
        rsa_results (List[Dict]): RSA算法的运行结果。
        config (Config): 配置对象。
        show_spectrum (bool): 是否显示频谱占用情况。
        show_paths (bool): 是否在拓扑图上显示分配的路径。
        show_timeline (bool): 是否显示连接时间轴。
    """
    num_plots = sum([show_spectrum, show_timeline]) + 1  # 至少显示拓扑图

    fig = plt.figure(figsize=(18, 6 * num_plots))  # 调整图表大小

    # --- 子图1: 网络拓扑和RSA路径分配 ---
    # 根据需要显示的子图数量动态调整布局
    if show_spectrum and show_timeline:
        ax1 = plt.subplot(num_plots, 1, 1)  # 2行2列或3行1列的第一个
    elif show_spectrum or show_timeline:
        ax1 = plt.subplot(num_plots, 1, 1)  # 2行1列的第一个
    else:
        ax1 = plt.subplot(1, 1, 1)  # 只有1个图

    pos = nx.spring_layout(G, seed=42)  # 使用固定种子，使布局可复现
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue',
            node_size=700, font_size=10, edge_color='gray', alpha=0.8)

    # 绘制链路长度标签
    edge_labels = nx.get_edge_attributes(G, 'length')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1, font_color='red')

    if show_paths:
        allocated_paths = [r for r in rsa_results if r['status'] == 'Allocated' and r['path'] is not None]
        # 使用不同的颜色绘制每条路径，使其更易区分
        colors = plt.cm.get_cmap('Paired', len(allocated_paths))  # 更好的颜色映射
        for i, result in enumerate(allocated_paths):
            path_edges = list(zip(result['path'], [result['path'][k + 1] for k in range(len(result['path']) - 1)]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax1,
                                   edge_color=[colors(i)], width=2, alpha=0.7,
                                   label=f"{result['source']}->{result['destination']}")
        # ax1.legend(title="Allocated Paths", loc='upper left', bbox_to_anchor=(1, 1)) # 如果路径过多，legend会很乱

    ax1.set_title("Optical Network Topology and RSA Path Allocation")
    ax1.set_axis_on()  # 确保坐标轴可见，对于拓扑图可能不需要，但如果需要微调布局可能有用
    ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # --- 子图2: 平均频谱槽占用率热力图 ---
    if show_spectrum:
        # 动态调整子图位置
        if show_timeline:
            ax2 = plt.subplot(num_plots, 1, 2)
        else:
            ax2 = plt.subplot(num_plots, 1, 2)  # 如果没有时间轴，它就是第二个图

        edge_list = list(G.edges())
        avg_slot_usage_matrix = np.zeros((len(edge_list), config.num_slots_per_link))

        # 计算每个链路每个槽的平均占用率
        # 注意: 这里的G['active_connections'] 在dynamic_rsa_algorithm运行结束后，可能不包含所有历史连接。
        # 如果需要精确的平均占用率，需要在 RSA 过程中累积每个槽的占用时间，而不仅仅是当前活跃的连接。
        # 为了模拟的准确性，这里我们假设 G['active_connections'] 包含了在模拟周期内发生过的所有连接信息。
        # 更好的方法是在 RSA 算法中维护一个更全面的历史记录。
        # 当前实现是计算“在模拟结束时，每个链路每个槽被历史连接占用的总时长 / 模拟总时长”

        for edge_idx, (u, v) in enumerate(edge_list):
            total_occupied_time_per_slot = [0.0] * config.num_slots_per_link
            # 遍历该链路上所有在模拟过程中激活过的连接
            # 注意：如果active_connections只保留当前活跃的，这个统计会不准确。
            # 我们需要的是所有历史连接的累积占用时间。
            # 假设 active_connections 在这里包含了所有历史连接
            # 实际上，dynamic_rsa_algorithm中的active_connections只维护当前活跃的连接。
            # 为了准确统计，RSA算法需要修改为累积每个槽的占用时间。
            # 暂时按照原代码的逻辑，它会基于“模拟结束时” G.edges() 中的 active_connections 来计算。
            # 而这些 active_connections 只包含那些在模拟结束时仍然“活跃”的连接，或者因为清除机制而留下来的。
            # 真正的平均占用率应该是在整个模拟时间内，每个槽被占用的总时间除以模拟总时间。

            # 修正：为了获得准确的平均占用率，需要 RSA 算法在每次分配和释放时，
            # 更新一个累积的“占用时间”统计。这里暂时先沿用原逻辑，但需要注意其局限性。
            # 更精确的实现会涉及每个槽的 (start_time, end_time) 列表，然后计算重叠区间。

            # 临时修正：假定rsa_results中status='Allocated'的连接代表了某个时间段的占用。
            # 但这仍然不足以精确统计“每个槽在整个模拟时间段内的平均占用率”。
            # 这里改为计算：对于每个槽，有多少个分配给它的连接。这也不是严格的占用率。
            # 更精确的实现需要累加每个槽在整个模拟过程中被占用的总时长。

            # 为简化，我们暂时假定这里是计算“在整个模拟过程中，每个槽被分配的总时长”
            # 这需要RSA算法将已释放连接的槽占用信息也传递出来，或者在链路属性中累加。
            # 鉴于当前代码结构，最好的近似是：
            # 统计所有成功分配的业务，计算它们在每个链路上的每个槽的“活跃”时间。

            # 重构 avg_slot_usage_matrix 的计算：
            # 遍历所有成功分配的业务
            for res in rsa_results:
                if res['status'] == 'Allocated' and res['path'] is not None and res['spectrum'] is not None:
                    src_node, dst_node = res['source'], res['destination']
                    start_slot, end_slot = res['spectrum']
                    duration_of_connection = res['duration']

                    # 检查该连接是否经过当前链路
                    for i_path in range(len(res['path']) - 1):
                        u_path, v_path = res['path'][i_path], res['path'][i_path + 1]
                        # 确保边是无向的，所以比较时需要考虑 (u,v) 和 (v,u)
                        if (u, v) == (u_path, v_path) or (u, v) == (v_path, u_path):
                            # 如果该连接使用了这条链路，则累加其占用的时间到对应槽位
                            for slot_idx in range(start_slot, end_slot + 1):
                                if 0 <= slot_idx < config.num_slots_per_link:  # 确保索引不越界
                                    total_occupied_time_per_slot[slot_idx] += duration_of_connection
                            break  # 此连接已处理完该链路，移到下一个连接

            # 计算平均占用率：总占用时间 / 模拟总时间
            for slot_idx in range(config.num_slots_per_link):
                avg_slot_usage_matrix[edge_idx, slot_idx] = total_occupied_time_per_slot[
                                                                slot_idx] / config.simulation_time

        im = ax2.imshow(avg_slot_usage_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax2.set_title("Average Spectrum Slot Occupancy Rate (Cumulative over Simulation Time)")
        ax2.set_xlabel("Spectrum Slot Index")
        ax2.set_ylabel("Link Index")
        ax2.set_yticks(range(len(edge_list)))
        ax2.set_yticklabels([f"{u}-{v}" for u, v in edge_list])
        plt.colorbar(im, ax=ax2, label='Average Occupancy Rate (0 to 1)')
        ax2.set_aspect('auto')  # 避免图像被拉伸或压缩

    # --- 子图3: 连接时间轴 ---
    if show_timeline:
        ax3 = plt.subplot(num_plots, 1, num_plots)  # 放在最后一个子图位置

        allocated = [r for r in rsa_results if r['status'] == 'Allocated']

        # 将模拟时间戳转换为datetime对象，以便matplotlib正确绘制时间轴
        # 找到所有连接的最小到达时间和最大结束时间，用于设置时间轴范围
        min_time = min([r['arrival_time'] for r in allocated]) if allocated else 0
        max_time = max([r['end_time'] for r in allocated]) if allocated else config.simulation_time

        # 将相对时间转换为 datetime 对象，以模拟实际时间（从当前时间开始）
        base_datetime = datetime.now()

        # 按照连接的到达时间排序，以便在时间轴上更清晰地显示
        allocated_sorted = sorted(allocated, key=lambda x: x['arrival_time'])

        # 为每个连接在时间轴上绘制水平条
        for i, result in enumerate(allocated_sorted):
            # 将相对时间转换为绝对时间（datetime对象）
            start_dt = base_datetime.replace(microsecond=0) + \
                       (datetime.fromtimestamp(result['arrival_time']) - datetime.fromtimestamp(0))
            end_dt = base_datetime.replace(microsecond=0) + \
                     (datetime.fromtimestamp(result['end_time']) - datetime.fromtimestamp(0))

            ax3.barh(i, (end_dt - start_dt),
                     left=start_dt,
                     color=plt.cm.jet(i / len(allocated_sorted)) if allocated_sorted else 'blue',  # 使用jet颜色映射
                     alpha=0.6,
                     label=f"Conn {result['source']}->{result['destination']} (Slots:{result['slots_required']})"
                     )

        ax3.set_title("Connection Timeline")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Connection Index")  # 使用索引作为Y轴，更易于理解

        # 设置X轴为日期时间格式
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        # 动态调整X轴刻度间隔，根据模拟时间长度
        if max_time - min_time < 60:  # 模拟时间小于1分钟，显示每5秒
            ax3.xaxis.set_major_locator(mdates.SecondLocator(interval=5))
        elif max_time - min_time < 3600:  # 模拟时间小于1小时，显示每分钟
            ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        else:  # 模拟时间更长，显示每小时
            ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))

        # 旋转X轴标签，避免重叠
        fig.autofmt_xdate()
        ax3.grid(True, axis='x', linestyle='--', alpha=0.7)

        # 调整Y轴范围，使其更紧凑
        if allocated_sorted:
            ax3.set_ylim(-0.5, len(allocated_sorted) - 0.5)

        # 添加图例
        # 如果连接数量过多，图例会变得非常大且难以阅读，可以考虑不显示或简化
        # if len(allocated_sorted) < 20: # 限制图例数量
        #     ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # 调整布局，为可能的图例留出空间

    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_filename = os.path.join(config.imgbackup_path, f"RSA_result_{timestamp}.png")
    plt.savefig(img_filename, dpi=300, bbox_inches='tight')  # 提高DPI，保存高质量图片
    print(f"可视化结果已保存到: {img_filename}")

    plt.show()  # 显示图表


# 备份分配结果到JSON文件
def backup_allocation_results(rsa_results: List[Dict], config: Config) -> None:
    """
    将RSA分配结果备份到JSON文件。

    Args:
        rsa_results (List[Dict]): RSA算法的运行结果。
        config (Config): 配置对象，包含备份路径。
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 使用os.path.join确保跨平台的路径兼容性
    filename = os.path.join(config.allocation_results_path, f"rsa_results_{timestamp}.json")

    # 确保目录存在，Config类中已经处理，但这里再加一层防护
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        with open(filename, 'w', encoding='utf-8') as f:  # 指定utf-8编码
            json.dump(rsa_results, f, indent=2, ensure_ascii=False)  # ensure_ascii=False支持中文
        print(f"分配结果已备份到: {filename}")
    except IOError as e:
        print(f"错误: 无法写入分配结果文件 {filename}: {e}")


# 备份业务分析结果到TXT文件
def backup_business_analysis(demands: List[Tuple[int, int, int, float, float]],
                             analysis: Dict, config: Config) -> None:
    """
    将业务需求和性能分析结果备份到TXT文件。

    Args:
        demands (List[Tuple]): 生成的业务请求列表。
        analysis (Dict): 性能分析结果字典。
        config (Config): 配置对象，包含备份路径。
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(config.business_analysis_path, f"business_analysis_{timestamp}.txt")

    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        with open(filename, 'w', encoding='utf-8') as f:  # 指定utf-8编码
            f.write(f"=== 模拟时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write("=== 生成的动态业务需求 ===\n")
            # 格式化输出业务需求
            for i, (src, dst, slots, arrival, duration) in enumerate(demands, 1):
                f.write(f"业务{i:04d}: {src:2d} -> {dst:2d}, 需要{slots}个频谱槽, "
                        f"到达时间: {arrival:.2f}s, 持续时间: {duration:.2f}s\n")

            f.write("\n=== 性能分析 ===\n")
            # 格式化输出性能分析结果
            for key, value in analysis.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("=" * 50 + "\n")

        print(f"业务需求和性能分析已备份到: {filename}")
    except IOError as e:
        print(f"错误: 无法写入业务分析文件 {filename}: {e}")


# 统计分析模拟结果
def analyze_results(rsa_results: List[Dict]) -> Dict:
    """
    对RSA模拟结果进行统计分析，计算阻塞概率、平均路径长度等指标。

    Args:
        rsa_results (List[Dict]): RSA算法的运行结果。

    Returns:
        Dict: 包含各项性能指标的字典。
    """
    total_demands = len(rsa_results)
    allocated_demands = sum(1 for r in rsa_results if r['status'] == 'Allocated')
    blocked_demands = total_demands - allocated_demands

    # 阻塞概率
    blocking_probability = blocked_demands / total_demands if total_demands > 0 else 0

    avg_path_length = 0.0
    avg_slots_used = 0.0
    avg_duration = 0.0

    if allocated_demands > 0:
        # 计算平均路径长度（跳数）
        total_path_hops = sum(
            (len(r['path']) - 1) for r in rsa_results if r['status'] == 'Allocated' and r['path'] is not None)
        avg_path_length = total_path_hops / allocated_demands

        # 计算平均使用频谱槽数
        total_slots_used_count = sum(
            (r['spectrum'][1] - r['spectrum'][0] + 1) for r in rsa_results if
            r['status'] == 'Allocated' and r['spectrum'] is not None)
        avg_slots_used = total_slots_used_count / allocated_demands

        # 计算平均连接持续时间
        total_duration_sum = sum(r['duration'] for r in rsa_results if r['status'] == 'Allocated')
        avg_duration = total_duration_sum / allocated_demands

    return {
        "总业务数": total_demands,
        "已分配业务数": allocated_demands,
        "被阻塞业务数": blocked_demands,
        "阻塞概率": blocking_probability,
        "平均路径长度 (跳数)": avg_path_length,
        "平均使用频谱槽数": avg_slots_used,
        "平均连接持续时间 (秒)": avg_duration
    }


# 用户交互界面
def user_interface() -> None:
    """
    提供一个交互式的用户界面，引导用户配置和运行模拟。
    """
    print("=" * 50)
    print("=== 动态RSA光网络频谱分配模拟器 ===")
    print("=" * 50)

    # 初始化配置，尝试从默认配置文件加载
    config = Config(config_file="config.json")  # 可以指定一个配置文件名，例如 config.json

    try:
        print("\n请选择网络拓扑类型:")
        print("1. NSFNET 拓扑 (固定14节点, 21链路)")
        print("2. 随机拓扑")
        topology_choice = input("请输入选择 (1-2, 默认 1): ").strip() or "1"
        topology_type = "nsfnet" if topology_choice == "1" else "random"

        # 如果选择随机拓扑，则允许用户输入节点和链路数
        if topology_type == "random":
            while True:
                try:
                    num_nodes_input = input(f"请输入网络节点数 (默认 {config.num_nodes}): ").strip()
                    config.num_nodes = int(num_nodes_input) if num_nodes_input else config.num_nodes
                    if config.num_nodes < 2:
                        print("节点数至少为2。")
                        continue
                    break
                except ValueError:
                    print("请输入有效的整数节点数。")

            while True:
                try:
                    num_links_input = input(f"请输入网络链路数 (默认 {config.num_links}): ").strip()
                    config.num_links = int(num_links_input) if num_links_input else config.num_links
                    if config.num_links < config.num_nodes - 1 and config.num_nodes > 1:
                        print(f"链路数至少为 {config.num_nodes - 1} 以保证拓扑连通。")
                        continue
                    if config.num_links > config.num_nodes * (config.num_nodes - 1) / 2:
                        print("链路数不能超过最大可能链路数。")
                        continue
                    break
                except ValueError:
                    print("请输入有效的整数链路数。")
        else:  # NSFNET拓扑，固定节点和链路数
            print(f"NSFNET拓扑将使用固定的 {config.num_nodes} 节点和 {config.num_links} 链路。")
            # Config中已经初始化为14, 21，这里无需额外设置

        while True:
            try:
                k_paths_input = input(f"请输入要计算的最短路径数量K (默认 {config.default_k_paths}): ").strip()
                config.default_k_paths = int(k_paths_input) if k_paths_input else config.default_k_paths
                if config.default_k_paths <= 0:
                    print("K值必须大于0。")
                    continue
                break
            except ValueError:
                print("请输入有效的整数K值。")

        while True:
            try:
                lambda_input = input(f"请输入业务到达率λ (默认 {config.lambda_rate}): ").strip()
                config.lambda_rate = float(lambda_input) if lambda_input else config.lambda_rate
                if config.lambda_rate <= 0:
                    print("到达率λ必须大于0。")
                    continue
                break
            except ValueError:
                print("请输入有效的浮点数到达率。")

        while True:
            try:
                mu_input = input(f"请输入业务离开率μ (默认 {config.mu_rate}): ").strip()
                config.mu_rate = float(mu_input) if mu_input else config.mu_rate
                if config.mu_rate <= 0:
                    print("离开率μ必须大于0。")
                    continue
                break
            except ValueError:
                print("请输入有效的浮点数离开率。")

        while True:
            try:
                sim_time_input = input(f"请输入模拟总时间 (秒) (默认 {config.simulation_time}): ").strip()
                config.simulation_time = float(sim_time_input) if sim_time_input else config.simulation_time
                if config.simulation_time <= 0:
                    print("模拟时间必须大于0。")
                    continue
                break
            except ValueError:
                print("请输入有效的浮点数模拟时间。")

        print("\n请选择频谱分配算法:")
        print("1. First-Fit (首次适配): 寻找第一个满足需求的连续空闲槽。")
        print("2. Best-Fit (最佳适配): 寻找最小的、满足需求的连续空闲槽块。")
        print("3. Random-Fit (随机适配): 从所有满足需求的连续空闲槽块中随机选择一个。")
        algo_choice = input("请输入选择 (1-3, 默认 1): ").strip() or "1"

        spectrum_algorithm_map = {
            "1": "first_fit",
            "2": "best_fit",
            "3": "random_fit"
        }
        spectrum_algorithm = spectrum_algorithm_map.get(algo_choice, "first_fit")
        if algo_choice not in spectrum_algorithm_map:
            print("无效选择，将使用默认的 First-Fit 算法。")

        print("\n可视化选项:")
        show_spectrum = input("是否显示平均频谱占用情况? (y/n, 默认 y): ").lower().strip() != "n"
        show_paths = input("是否在拓扑图上显示分配的路径? (y/n, 默认 y): ").lower().strip() != "n"
        show_timeline = input("是否显示连接时间轴? (y/n, 默认 y): ").lower().strip() != "n"

        print("\n=== 模拟配置确认 ===")
        print(f"拓扑类型: {'NSFNET' if topology_type == 'nsfnet' else '随机拓扑'}")
        print(f"网络节点数: {config.num_nodes}")
        print(f"网络链路数: {config.num_links}")
        print(f"每链路频谱槽数: {config.num_slots_per_link}")
        print(f"链路长度范围: {config.min_link_length}-{config.max_link_length}")
        print(f"到达率λ: {config.lambda_rate} (业务/秒)")
        print(f"离开率μ: {config.mu_rate} (1/秒, 对应平均持续时间 {1 / config.mu_rate:.2f}秒)")
        print(f"模拟总时间: {config.simulation_time} 秒")
        print(f"频谱分配算法: {spectrum_algorithm.replace('_', ' ').title()}")
        print(f"计算的最短路径数K: {config.default_k_paths}")
        print(f"显示平均频谱占用: {'是' if show_spectrum else '否'}")
        print(f"显示分配路径: {'是' if show_paths else '否'}")
        print(f"显示时间轴: {'是' if show_timeline else '否'}")
        print(f"结果备份路径: {os.path.abspath(config.allocation_results_path)}")
        print(f"分析备份路径: {os.path.abspath(config.business_analysis_path)}")
        print(f"图片备份路径: {os.path.abspath(config.imgbackup_path)}")

        confirm = input("\n是否根据以上配置开始模拟? (y/n, 默认 y): ").lower().strip() != "n"
        if not confirm:
            print("模拟已取消。")
            return

        print(f"\n正在生成 {config.num_nodes} 个节点和 {config.num_links} 条链路的网络拓扑 ({topology_type})...")
        G = generate_topology(config, topology_type=topology_type)

        print(f"正在生成动态业务需求 (λ={config.lambda_rate}, μ={config.mu_rate})...")
        demands = generate_dynamic_demands(config)
        print(f"共生成 {len(demands)} 个业务请求。")

        # # 打印生成的前几个业务需求（可选）
        # print("\n部分生成的业务需求 (前5个):")
        # for i, (src, dst, slots, arrival, duration) in enumerate(demands[:5], 1):
        #     print(f"业务{i}: {src} -> {dst}, 需要{slots}个频谱槽, "
        #           f"到达时间: {arrival:.2f}s, 持续时间: {duration:.2f}s")
        # if len(demands) > 5:
        #     print("...")

        print(
            f"\n正在运行动态RSA算法 (频谱分配: {spectrum_algorithm.replace('_', ' ').title()}, K={config.default_k_paths})...")
        rsa_results = dynamic_rsa_algorithm(G, demands, config, spectrum_algorithm=spectrum_algorithm)

        print("\n正在可视化结果...")
        visualize_results(G, rsa_results, config,
                          show_spectrum=show_spectrum,
                          show_paths=show_paths,
                          show_timeline=show_timeline)

        analysis = analyze_results(rsa_results)
        print("\n=== 性能分析结果 ===")
        for key, value in analysis.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print("\n正在备份结果...")
        backup_allocation_results(rsa_results, config)
        backup_business_analysis(demands, analysis, config)

        print("\n=== 模拟完成！ ===")

    except ValueError as e:
        print(f"\n输入错误: {e}")
    except nx.NetworkXError as e:
        print(f"\n网络拓扑错误: {e}")
    except KeyboardInterrupt:
        print("\n操作已取消。")
    except Exception as e:
        print(f"\n发生未知错误: {e}")


if __name__ == "__main__":
    user_interface()