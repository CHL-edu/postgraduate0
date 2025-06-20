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


# 配置参数
class Config:
    def __init__(self, config_file=None):
        self.business_analysis_path = "/data/chl/pyproject/backup/RSA/rsa_results/testRSAnode"
        self.imgbackup_path = "/data/chl/pyproject/backup/RSA/RSAimg"
        self.allocation_results_path = "/data/chl/pyproject/backup/RSA/rsa_results/rsa_results"
        self.num_nodes = 14
        self.num_links = 21
        self.num_slots_per_link = 10
        self.min_link_length = 100
        self.max_link_length = 1000
        self.default_k_paths = 3

        # 场景A：轻负载 + 短业务持续时间
        self.lambda_rate = 20
        self.mu_rate = 0.5  # 平均业务持续2秒
        self.simulation_time = 100  # 模拟100秒
        """
        # 场景B：中等负载 + 长业务持续时间
        self.lambda_rate = 40
        self.mu_rate = 0.1  # 平均业务持续10秒
        self.simulation_time = 300  # 延长模拟时间以达到稳态
        
        # 场景C：重负载 + 混合业务持续时间
        self.lambda_rate = 60
        self.mu_rate = 0.2  # 平均业务持续5秒
        self.simulation_time = 200"""
        #ρ = λ/（μ*C） C= num_links * num_slots_per_link
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

        os.makedirs(self.imgbackup_path, exist_ok=True)


# 生成光网络拓扑（支持固定和随机拓扑）
def generate_topology(config: Config, topology_type: str = "random") -> nx.Graph:
    G = nx.Graph()

    if topology_type == "nsfnet":
        # NSFNET 拓扑：14个节点，21条链路
        nodes = list(range(14))
        edges = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 4), (3, 10),
            (4, 5), (4, 6), (5, 7), (5, 8), (6, 9), (7, 10), (7, 11), (8, 9),
            (8, 11), (9, 10), (9, 12), (11, 12), (12, 13)
        ]
        for i in nodes:
            G.add_node(i, label=f'N{i}')
        for u, v in edges:
            G.add_edge(u, v,
                       length=random.randint(config.min_link_length, config.max_link_length),
                       slots=[0] * config.num_slots_per_link,
                       active_connections=[])
    else:
        # 随机拓扑
        for i in range(config.num_nodes):
            G.add_node(i, label=f'N{i}')

        edges = []
        nodes = list(range(config.num_nodes))
        random.shuffle(nodes)
        for i in range(1, config.num_nodes):
            parent = random.randint(0, i - 1)
            edges.append((nodes[parent], nodes[i]))

        remaining_edges = config.num_links - (config.num_nodes - 1)
        if remaining_edges > 0:
            potential_edges = [(i, j) for i in range(config.num_nodes)
                               for j in range(i + 1, config.num_nodes)
                               if (i, j) not in edges]
            additional_edges = random.sample(potential_edges, min(remaining_edges, len(potential_edges)))
            edges.extend(additional_edges)

        for u, v in edges:
            G.add_edge(u, v,
                       length=random.randint(config.min_link_length, config.max_link_length),
                       slots=[0] * config.num_slots_per_link,
                       active_connections=[])
    return G


# 计算K最短路径
def k_shortest_paths(G: nx.Graph, source: int, target: int, k: int = 3) -> List[List[int]]:
    try:
        return list(nx.shortest_simple_paths(G, source, target, weight='length'))[:k]
    except nx.NetworkXNoPath:
        return []


# 频谱分配算法
def find_continuous_slots(slots: List[int], num_required: int) -> Optional[Tuple[int, int]]:
    count = 0
    start = 0
    for i, slot in enumerate(slots):
        if slot == 0:
            if count == 0:
                start = i
            count += 1
            if count >= num_required:
                return start, start + count - 1
        else:
            count = 0
    return None


def spectrum_allocation(G: nx.Graph, path: List[int], num_slots: int,
                        start_time: float, duration: float,
                        algorithm: str = "first_fit") -> Optional[Tuple[int, int]]:
    available_slots = None
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        slots = G[u][v]['slots']
        if available_slots is None:
            available_slots = slots.copy()
        else:
            available_slots = [1 if a == 1 or b == 1 else 0 for a, b in zip(available_slots, slots)]

    if algorithm == "first_fit":
        allocation = find_continuous_slots(available_slots, num_slots)
    elif algorithm == "best_fit":
        best_allocation = None
        min_size = float('inf')
        count = 0
        start = 0
        for i, slot in enumerate(available_slots):
            if slot == 0:
                if count == 0:
                    start = i
                count += 1
                if count >= num_slots and count < min_size:
                    best_allocation = (start, start + count - 1)
                    min_size = count
            else:
                count = 0
        allocation = best_allocation
    elif algorithm == "random_fit":
        possible_allocations = []
        count = 0
        start = 0
        for i, slot in enumerate(available_slots):
            if slot == 0:
                if count == 0:
                    start = i
                count += 1
                if count >= num_slots:
                    possible_allocations.append((start, start + count - 1))
            else:
                count = 0
        allocation = random.choice(possible_allocations) if possible_allocations else None
    else:
        raise ValueError(f"不支持的频谱分配算法: {algorithm}")

    if allocation:
        start, end = allocation
        connection = {'start_time': start_time, 'end_time': start_time + duration,
                      'slots': (start, end)}
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            G[u][v]['active_connections'].append(connection)
            for j in range(start, end + 1):
                G[u][v]['slots'][j] = 1
        return start, end
    return None


# 生成动态业务请求（泊松过程）
def generate_dynamic_demands(config: Config) -> List[Tuple[int, int, int, float, float]]:
    demands = []
    current_time = 0
    while current_time < config.simulation_time:
        inter_arrival = np.random.exponential(1 / config.lambda_rate)
        current_time += inter_arrival

        if current_time >= config.simulation_time:
            break

        src = random.randint(0, config.num_nodes - 1)
        dst = random.randint(0, config.num_nodes - 1)
        while dst == src:
            dst = random.randint(0, config.num_nodes - 1)

        slots = random.randint(1, 5)
        duration = np.random.exponential(1 / config.mu_rate)
        demands.append((src, dst, slots, current_time, duration))

    return sorted(demands, key=lambda x: x[3])


# RSA算法（动态版本）
def dynamic_rsa_algorithm(G: nx.Graph, demands: List[Tuple[int, int, int, float, float]],
                          config: Config, spectrum_algorithm: str = "first_fit") -> List[Dict]:
    results = []
    current_time = 0

    for src, dst, slots, arrival_time, duration in demands:
        current_time = max(current_time, arrival_time)
        for u, v in G.edges():
            expired = []
            for i, conn in enumerate(G[u][v]['active_connections']):
                if conn['end_time'] <= current_time:
                    expired.append(i)
                    for j in range(conn['slots'][0], conn['slots'][1] + 1):
                        G[u][v]['slots'][j] = 0
            G[u][v]['active_connections'] = [conn for i, conn in enumerate(G[u][v]['active_connections'])
                                             if i not in expired]

        paths = k_shortest_paths(G, src, dst, k=config.default_k_paths)
        allocated = False

        for path in paths:
            spectrum = spectrum_allocation(G, path, slots, arrival_time, duration,
                                           algorithm=spectrum_algorithm)
            if spectrum:
                results.append({
                    'source': src,
                    'destination': dst,
                    'path': path,
                    'spectrum': spectrum,
                    'slots_required': slots,
                    'arrival_time': arrival_time,
                    'duration': duration,
                    'end_time': arrival_time + duration,
                    'status': 'Allocated'
                })
                allocated = True
                break

        if not allocated:
            results.append({
                'source': src,
                'destination': dst,
                'path': None,
                'spectrum': None,
                'slots_required': slots,
                'arrival_time': arrival_time,
                'duration': duration,
                'end_time': arrival_time + duration,
                'status': 'Blocked'
            })

    return results


# 可视化（修改为显示平均占用情况）
def visualize_results(G: nx.Graph, rsa_results: List[Dict], config: Config,
                      show_spectrum: bool = True, show_paths: bool = True,
                      show_timeline: bool = True) -> None:
    plt.figure(figsize=(20, 12))

    # 网络拓扑
    ax1 = plt.subplot(221 if show_spectrum and show_timeline else 111)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, edge_color='gray')

    if show_paths:
        allocated_paths = [r for r in rsa_results if r['status'] == 'Allocated']
        colors = plt.cm.jet(np.linspace(0, 1, len(allocated_paths)))
        for i, result in enumerate(allocated_paths):
            path_edges = list(zip(result['path'], result['path'][1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax1,
                                   edge_color=[colors[i]], width=2, alpha=0.7)

    ax1.set_title("Optical Network Topology and RSA Path Allocation")

    # 频谱占用（改为平均占用率）
    if show_spectrum:
        ax2 = plt.subplot(222 if show_timeline else 112)
        edge_list = list(G.edges())
        avg_slot_usage = []

        for u, v in edge_list:
            total_occupied_time = [0] * config.num_slots_per_link
            for conn in G[u][v]['active_connections']:
                start_slot, end_slot = conn['slots']
                duration = conn['end_time'] - conn['start_time']
                for j in range(start_slot, end_slot + 1):
                    total_occupied_time[j] += duration

            # 计算平均占用率
            avg_occupation = [time / config.simulation_time for time in total_occupied_time]
            avg_slot_usage.append(avg_occupation)

        im = ax2.imshow(avg_slot_usage, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax2.set_title("Average Spectrum Slot Occupancy Rate")
        ax2.set_xlabel("Spectrum slot index")
        ax2.set_ylabel("Link index")
        ax2.set_yticks(range(len(edge_list)))
        ax2.set_yticklabels([f"{u}-{v}" for u, v in edge_list])
        plt.colorbar(im, ax=ax2, label='Average occupancy rate (0 to 1)')

    # 时间轴可视化
    if show_timeline:
        ax3 = plt.subplot(212)
        allocated = [r for r in rsa_results if r['status'] == 'Allocated']
        for i, result in enumerate(allocated):
            start = datetime.fromtimestamp(result['arrival_time'])
            end = datetime.fromtimestamp(result['end_time'])
            ax3.barh(i, (end - start).total_seconds() / 3600,
                     left=start,
                     color=plt.cm.jet(i / len(allocated)),
                     alpha=0.6,
                     label=f"Connection {result['source']}→{result['destination']}")

        ax3.set_title("Connection Timeline")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Connections")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax3.grid(True, axis='x', linestyle='--', alpha=0.7)
        if allocated:
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_filename = f"{config.imgbackup_path}/RSA_result_{timestamp}.png"
    plt.savefig(img_filename, dpi=500, bbox_inches='tight')
    print(f"可视化结果已保存到: {img_filename}")

    plt.tight_layout()
    plt.show()


# 备份分配结果
def backup_allocation_results(rsa_results: List[Dict], config: Config) -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{config.allocation_results_path}_{timestamp}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(rsa_results, f, indent=2)
    print(f"分配结果已备份到: {filename}")


# 备份业务分析
def backup_business_analysis(demands: List[Tuple[int, int, int, float, float]],
                             analysis: Dict, config: Config) -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{config.business_analysis_path}_{timestamp}.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write(f"=== 模拟时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write("=== 生成的动态业务需求 ===\n")
        for i, (src, dst, slots, arrival, duration) in enumerate(demands, 1):
            f.write(f"业务{i}: {src} -> {dst}, 需要{slots}个频谱槽, "
                    f"到达时间: {arrival:.2f}, 持续时间: {duration:.2f}\n")

        f.write("\n=== 性能分析 ===\n")
        for key, value in analysis.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n")

    print(f"业务需求和性能分析已备份到: {filename}")


# 统计分析
def analyze_results(rsa_results: List[Dict]) -> Dict:
    total_demands = len(rsa_results)
    allocated_demands = sum(1 for r in rsa_results if r['status'] == 'Allocated')
    blocked_demands = total_demands - allocated_demands

    blocking_probability = blocked_demands / total_demands if total_demands > 0 else 0

    avg_path_length = 0
    avg_slots_used = 0
    avg_duration = 0
    if allocated_demands > 0:
        total_path_length = sum(len(r['path']) - 1 for r in rsa_results if r['status'] == 'Allocated')
        avg_path_length = total_path_length / allocated_demands

        total_slots_used = sum(
            r['spectrum'][1] - r['spectrum'][0] + 1 for r in rsa_results if r['status'] == 'Allocated')
        avg_slots_used = total_slots_used / allocated_demands

        total_duration = sum(r['duration'] for r in rsa_results if r['status'] == 'Allocated')
        avg_duration = total_duration / allocated_demands

    return {
        "总业务数": total_demands,
        "已分配业务数": allocated_demands,
        "被阻塞业务数": blocked_demands,
        "阻塞概率": blocking_probability,
        "平均路径长度": avg_path_length,
        "平均使用频谱槽数": avg_slots_used,
        "平均连接持续时间": avg_duration
    }


# 用户交互（添加拓扑选择）
def user_interface() -> None:
    print("=" * 50)
    print("=== 动态RSA光网络频谱分配模拟器 ===")
    print("=" * 50)

    config = Config()

    try:
        print("\n请选择网络拓扑类型:")
        print("1. NSFNET 拓扑")
        print("2. 随机拓扑")
        topology_choice = input("请输入选择 (1-2, 默认 1): ") or "1"
        topology_type = "nsfnet" if topology_choice == "1" else "random"

        config.num_nodes = int(input(f"请输入网络节点数 (默认 {config.num_nodes}): ") or config.num_nodes)
        config.num_links = int(input(f"请输入网络链路数 (默认 {config.num_links}): ") or config.num_links)
        config.default_k_paths = int(
            input(f"请输入要计算的最短路径数量 (默认 {config.default_k_paths}): ") or config.default_k_paths)
        config.lambda_rate = float(input(f"请输入到达率λ (默认 {config.lambda_rate}): ") or config.lambda_rate)
        config.mu_rate = float(input(f"请输入离开率μ (默认 {config.mu_rate}): ") or config.mu_rate)
        config.simulation_time = float(
            input(f"请输入模拟总时间 (默认 {config.simulation_time}): ") or config.simulation_time)

        print("\n请选择频谱分配算法:")
        print("1. First-Fit (首次适配)")
        print("2. Best-Fit (最佳适配)")
        print("3. Random-Fit (随机适配)")
        algo_choice = input("请输入选择 (1-3, 默认 1): ") or "1"

        if algo_choice == "1":
            spectrum_algorithm = "first_fit"
        elif algo_choice == "2":
            spectrum_algorithm = "best_fit"
        elif algo_choice == "3":
            spectrum_algorithm = "random_fit"
        else:
            print("无效选择，使用默认的 First-Fit 算法")
            spectrum_algorithm = "first_fit"

        print("\n可视化选项:")
        show_spectrum = input("是否显示频谱占用情况? (y/n, 默认 y): ").lower() != "n"
        show_paths = input("是否在拓扑图上显示分配的路径? (y/n, 默认 y): ").lower() != "n"
        show_timeline = input("是否显示连接时间轴? (y/n, 默认 y): ").lower() != "n"

        print("\n=== 配置确认 ===")
        print(f"拓扑类型: {'NSFNET' if topology_type == 'nsfnet' else '随机拓扑'}")
        print(f"网络节点数: {config.num_nodes}")
        print(f"网络链路数: {config.num_links}")
        print(f"到达率λ: {config.lambda_rate}")
        print(f"离开率μ: {config.mu_rate}")
        print(f"模拟总时间: {config.simulation_time}")
        print(f"频谱分配算法: {spectrum_algorithm}")
        print(f"计算的最短路径数: {config.default_k_paths}")
        print(f"显示频谱占用: {'是' if show_spectrum else '否'}")
        print(f"显示分配路径: {'是' if show_paths else '否'}")
        print(f"显示时间轴: {'是' if show_timeline else '否'}")

        confirm = input("\n是否继续? (y/n, 默认 y): ").lower() != "n"
        if not confirm:
            print("模拟已取消")
            return

        print(f"\n生成 {config.num_nodes} 个节点和 {config.num_links} 条链路的网络拓扑...")
        G = generate_topology(config, topology_type=topology_type)

        print(f"生成动态业务需求 (λ={config.lambda_rate}, μ={config.mu_rate})...")
        demands = generate_dynamic_demands(config)

        print("\n生成的业务需求:")
        for i, (src, dst, slots, arrival, duration) in enumerate(demands, 1):
            print(f"业务{i}: {src} -> {dst}, 需要{slots}个频谱槽, "
                  f"到达时间: {arrival:.2f}, 持续时间: {duration:.2f}")

        print(f"\n运行动态RSA算法 (频谱分配: {spectrum_algorithm}, K={config.default_k_paths})...")
        rsa_results = dynamic_rsa_algorithm(G, demands, config, spectrum_algorithm=spectrum_algorithm)

        print("\n可视化结果...")
        visualize_results(G, rsa_results, config,
                          show_spectrum=show_spectrum,
                          show_paths=show_paths,
                          show_timeline=show_timeline)

        analysis = analyze_results(rsa_results)
        print("\n=== 性能分析 ===")
        for key, value in analysis.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print("\n备份结果...")
        backup_allocation_results(rsa_results, config)
        backup_business_analysis(demands, analysis, config)

        print("\n=== 模拟完成 ===")

    except ValueError as e:
        print(f"输入错误: {e}")
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    user_interface()