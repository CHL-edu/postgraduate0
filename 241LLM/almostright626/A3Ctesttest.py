from __future__ import division
import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import os
from time import sleep
from time import time
import struct
from collections import defaultdict, deque
from ACNettesttest import AC_Net
from almostright626.Agenttesttest import DeepRMSA_Agent
import logging
import signal
import json
import configparser
from typing import Dict, List, Tuple, Optional
import psutil  # 用于监控系统资源

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        self.config = configparser.ConfigParser()
        if config_file and os.path.exists(config_file):
            self.config.read(config_file)
            logger.info(f"配置文件加载成功: {config_file}")
        else:
            self._set_default_config()

    def _set_default_config(self):
        """设置默认配置"""
        # 网络参数
        self.config['network'] = {
            'link_num': '44',
            'node_num': '14',
            'slot_total': '100'
        }

        # 路径参数
        self.config['path'] = {
            'n_paths': '10',
            'm_positions': '1',
            'k_path': '5'
        }

        # 模型参数
        self.config['model'] = {
            'num_layers': '5',
            'layer_size': '128',
            'regu_scalar': '1e-4'
        }

        # 训练参数
        self.config['training'] = {
            'max_cpu': '4',
            'lambda_req': '12',
            'gamma': '0.95',
            'episode_size': '1000',
            'batch_size': '200',
            'max_episodes': '50000',
            'save_interval': '100',
            'early_stop_bp_threshold': '0.05',
            'early_stop_reward_threshold': '900',
            'memory_limit_gb': '90'  # 内存限制
        }

        # 路径配置
        self.config['paths'] = {
            'output_dir': '/data/chl/pyproject/DeepRSA/output2/',
            'candidate_paths_file': '/data/chl/pyproject/DeepRSA/Src_Dst_Paths.dat'
        }

    def get_config(self) -> Dict:
        """获取完整配置"""
        config = {}
        for section in self.config.sections():
            config[section] = dict(self.config[section])
        return config

    def save_config(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w') as configfile:
            self.config.write(configfile)


class Config:
    """配置类 - 从ConfigManager加载"""

    def __init__(self, config_manager: ConfigManager):
        cm = config_manager

        # 网络参数
        self.LINK_NUM = cm.config.getint('network', 'link_num')
        self.NODE_NUM = cm.config.getint('network', 'node_num')
        self.SLOT_TOTAL = cm.config.getint('network', 'slot_total')

        # 路径参数
        self.N = cm.config.getint('path', 'n_paths')
        self.M = cm.config.getint('path', 'm_positions')
        self.k_path = cm.config.getint('path', 'k_path')
        self.n_actions = self.k_path * self.M

        # 模型参数
        self.model2_flag = 0
        self.num_layers = cm.config.getint('model', 'num_layers')
        self.layer_size = cm.config.getint('model', 'layer_size')
        self.regu_scalar = cm.config.getfloat('model', 'regu_scalar')

        # 训练参数
        self.max_cpu = cm.config.getint('training', 'max_cpu')
        self.lambda_req = cm.config.getint('training', 'lambda_req')
        self.lambda_time = [14]
        self.gamma = cm.config.getfloat('training', 'gamma')
        self.episode_size = cm.config.getint('training', 'episode_size')
        self.batch_size = cm.config.getint('training', 'batch_size')
        self.max_episodes = cm.config.getint('training', 'max_episodes')
        self.save_interval = cm.config.getint('training', 'save_interval')
        self.early_stop_bp_threshold = cm.config.getfloat('training', 'early_stop_bp_threshold')
        self.early_stop_reward_threshold = cm.config.getfloat('training', 'early_stop_reward_threshold')
        self.memory_limit_gb = cm.config.getfloat('training', 'memory_limit_gb')

        # 流量参数
        self.nonuniform = False

        # 输入维度
        self.x_dim_p = self.NODE_NUM * 2 + self.k_path * (
                1 + self.M * 2 + 2 + self.model2_flag * 3)
        self.x_dim_v = self.x_dim_p

        # 输出路径
        self.output_dir = cm.config.get('paths', 'output_dir')
        self.model_path = os.path.join(self.output_dir, 'model')
        self.candidate_paths_file = cm.config.get('paths', 'candidate_paths_file')
        self.load_model = False


def create_linkmap() -> Dict[int, Dict[int, Tuple[int, int]]]:
    """创建NSFNet拓扑 - 使用双向字典优化"""
    linkmap = defaultdict(lambda: defaultdict(lambda: None))

    # NSFNet topology
    links = [
        (1, 2, 0, 1050), (2, 1, 3, 1050), (1, 3, 1, 1500), (3, 1, 6, 1500),
        (1, 8, 2, 2400), (8, 1, 22, 2400), (2, 3, 4, 600), (3, 2, 7, 600),
        (2, 4, 5, 750), (4, 2, 9, 750), (3, 6, 8, 1800), (6, 3, 15, 1800),
        (4, 5, 10, 600), (5, 4, 12, 600), (4, 11, 11, 1950), (11, 4, 32, 1950),
        (5, 6, 13, 1200), (6, 5, 16, 1200), (5, 7, 14, 600), (7, 5, 19, 600),
        (6, 10, 17, 1050), (10, 6, 29, 1050), (6, 14, 18, 1800), (14, 6, 41, 1800),
        (7, 8, 20, 750), (8, 7, 23, 750), (7, 10, 21, 1350), (10, 7, 30, 1350),
        (8, 9, 24, 750), (9, 8, 25, 750), (9, 10, 26, 750), (10, 9, 31, 750),
        (9, 12, 27, 300), (12, 9, 35, 300), (9, 13, 28, 300), (13, 9, 38, 300),
        (11, 12, 33, 600), (12, 11, 36, 600), (11, 13, 34, 750), (13, 11, 39, 750),
        (12, 14, 37, 300), (14, 12, 42, 300), (13, 14, 40, 150), (14, 13, 43, 150)
    ]

    for src, dst, link_id, distance in links:
        linkmap[src][dst] = (link_id, distance)

    return dict(linkmap)  # 转换为普通字典以提高访问速度


def load_candidate_paths(path_file: str, NODE_NUM: int, N: int) -> Dict:
    """加载候选路径 - 改进的错误处理"""
    Candidate_Paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

    if not os.path.exists(path_file):
        logger.warning(f"路径文件 {path_file} 不存在，将使用默认路径生成策略")
        return dict(Candidate_Paths)

    try:
        with open(path_file, 'rb') as fp:
            for ii in range(1, NODE_NUM * NODE_NUM + 1):
                if ii % NODE_NUM == 0:
                    i = ii // NODE_NUM
                    j = (ii % NODE_NUM) + NODE_NUM
                else:
                    i = (ii // NODE_NUM) + 1
                    j = ii % NODE_NUM

                temp_num = []
                for tt in range(N):
                    data = fp.read(4)
                    if len(data) < 4:
                        logger.error(f"文件读取不完整：期望4字节，实际{len(data)}字节")
                        return dict(Candidate_Paths)
                    temp_num.append(struct.unpack("i", data)[0])

                if i != j:
                    for k in range(N):
                        if temp_num[k] > 0:
                            path_data = fp.read(4 * temp_num[k])
                            if len(path_data) < 4 * temp_num[k]:
                                logger.error(f"路径数据读取不完整")
                                return dict(Candidate_Paths)
                            temp_path = list(struct.unpack("i" * temp_num[k], path_data))
                            Candidate_Paths[i][j][k] = temp_path

        logger.info(f"成功加载候选路径文件: {path_file}")

        # 验证路径数据
        path_count = sum(1 for i in Candidate_Paths
                         for j in Candidate_Paths[i]
                         for k in Candidate_Paths[i][j]
                         if Candidate_Paths[i][j][k])
        logger.info(f"共加载 {path_count} 条有效路径")

    except Exception as e:
        logger.error(f"加载路径文件时出错: {e}")
        import traceback
        traceback.print_exc()

    return dict(Candidate_Paths)


def generate_src_dest_pairs(NODE_NUM: int, prob: np.ndarray) -> Tuple[List, np.ndarray, int]:
    """生成源-目的节点对 - 优化版本"""
    Src_Dest_Pair = []
    prob_list = []

    for ii in range(NODE_NUM):
        for jj in range(NODE_NUM):
            if ii != jj:
                prob_list.append(prob[ii, jj])
                Src_Dest_Pair.append([ii + 1, jj + 1])

    num_src_dest_pair = len(Src_Dest_Pair)
    prob_arr = np.array(prob_list)

    # 确保概率和为1
    prob_sum = prob_arr.sum()
    if prob_sum > 0:
        prob_arr = prob_arr / prob_sum
    else:
        # 均匀分布作为后备
        prob_arr = np.ones(num_src_dest_pair) / num_src_dest_pair

    return Src_Dest_Pair, prob_arr, num_src_dest_pair


def setup_environment(config: Config) -> None:
    """设置环境和创建输出目录"""
    directories = [
        config.model_path,
        os.path.join(config.output_dir, 'logs'),
        os.path.join(config.output_dir, 'checkpoints'),
        os.path.join(config.output_dir, 'figures')  # 用于保存图表
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(directory):
            logger.info(f"📁 使用目录: {directory}")
        else:
            logger.info(f"💡 创建目录: {directory}")


def signal_handler(signum, frame):
    """处理中断信号"""
    logger.info("\n⚠️  接收到中断信号，正在优雅地停止训练...")
    if 'coord' in globals() and coord is not None:
        coord.request_stop()


class ResourceMonitor:
    """系统资源监控器"""

    def __init__(self, memory_limit_gb: float):
        self.memory_limit_gb = memory_limit_gb
        self.process = psutil.Process()
        self.start_time = time()

    def check_memory(self) -> Tuple[bool, float]:
        """检查内存使用情况"""
        memory_gb = self.process.memory_info().rss / 1024 ** 3
        return memory_gb < self.memory_limit_gb, memory_gb

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_gb': self.process.memory_info().rss / 1024 ** 3,
            'memory_percent': self.process.memory_percent(),
            'num_threads': self.process.num_threads(),
            'runtime_second': (time() - self.start_time)
        }


class TrainingMonitor:
    """训练监控器 - 增强版本"""

    def __init__(self, config: Config):
        self.config = config
        self.start_time = time()
        self.episode_times = deque(maxlen=100)
        self.best_bp = 1.0
        self.best_reward = -float('inf')
        self.bp_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.convergence_patience = 500
        self.no_improvement_count = 0

    def update(self, episode: int, bp: float, reward: float, episode_time: float):
        """更新监控数据"""
        self.episode_times.append(episode_time)
        self.bp_history.append(bp)
        self.reward_history.append(reward)

        improved = False
        if bp < self.best_bp:
            self.best_bp = bp
            improved = True
            logger.info(f"🎯 新的最佳阻塞率: {bp:.4f} (Episode {episode})")

        if reward > self.best_reward:
            self.best_reward = reward
            improved = True
            logger.info(f"🎯 新的最佳奖励: {reward:.2f} (Episode {episode})")

        if improved:
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

    def should_early_stop(self, recent_bp: float, recent_reward: float) -> bool:
        """检查是否应该早停"""
        # 目标达成
        if (recent_bp < self.config.early_stop_bp_threshold and
                recent_reward > self.config.early_stop_reward_threshold):
            logger.info("✅ 性能目标已达成")
            return True

        # 长时间无改善
        if self.no_improvement_count > self.convergence_patience:
            logger.info("⚠️ 长时间无改善，触发早停")
            return True

        return False

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_time = time() - self.start_time
        avg_episode_time = np.mean(self.episode_times) if self.episode_times else 0

        return {
            'total_time_hours': total_time / 3600,
            'avg_episode_time': avg_episode_time,
            'best_bp': self.best_bp,
            'best_reward': self.best_reward,
            'recent_bp': np.mean(list(self.bp_history)[-100:]) if len(self.bp_history) > 0 else 1.0,
            'recent_reward': np.mean(list(self.reward_history)[-100:]) if len(self.reward_history) > 0 else 0.0,
            'episodes_since_improvement': self.no_improvement_count
        }


def validate_config(config: Config) -> bool:
    """验证配置的合理性"""
    if config.SLOT_TOTAL <= 0:
        logger.error("SLOT_TOTAL 必须大于0")
        return False

    if config.k_path <= 0 or config.M <= 0:
        logger.error("k_path 和 M 必须大于0")
        return False

    if config.batch_size > config.episode_size:
        logger.warning(
            f"batch_size ({config.batch_size}) 大于 episode_size ({config.episode_size})，这可能导致训练效率低下")

    return True


# 主程序
def main():
    """主函数 - 增强版本"""
    global coord
    coord = None
    checkpoint_manager = None
    agents = []
    agent_threads = []
    resource_monitor = None

    try:
        # 加载配置
        config_manager = ConfigManager('config.ini')
        config = Config(config_manager)

        # 验证配置
        if not validate_config(config):
            logger.error("配置验证失败")
            return

        # 保存配置副本
        config_manager.save_config(os.path.join(config.output_dir, 'training_config.ini'))

        # 设置环境
        setup_environment(config)

        # 初始化资源监控器
        resource_monitor = ResourceMonitor(config.memory_limit_gb)
        logger.info(f"🔧 内存限制设置为 {config.memory_limit_gb} GB")

        # 创建链路映射
        linkmap = create_linkmap()

        # 流量分布矩阵（用于非均匀流量）
        trafic_dis = np.array([
            [0, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 1],
            [2, 0, 2, 1, 8, 2, 1, 5, 3, 5, 1, 5, 1, 4],
            [1, 2, 0, 2, 3, 2, 11, 20, 5, 2, 1, 1, 1, 2],
            [1, 1, 2, 0, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2],
            [1, 8, 3, 1, 0, 3, 3, 7, 3, 3, 1, 5, 2, 5],
            [4, 2, 2, 1, 3, 0, 2, 1, 2, 2, 1, 1, 1, 2],
            [1, 1, 11, 2, 3, 2, 0, 9, 4, 20, 1, 8, 1, 4],
            [1, 5, 20, 1, 7, 1, 9, 0, 27, 7, 2, 3, 2, 4],
            [2, 3, 5, 2, 3, 2, 4, 27, 0, 75, 2, 9, 3, 1],
            [1, 5, 2, 2, 3, 2, 20, 7, 75, 0, 1, 1, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 2, 1, 61],
            [1, 5, 1, 2, 5, 1, 8, 3, 9, 1, 2, 0, 1, 81],
            [1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 0, 2],
            [1, 4, 2, 2, 5, 2, 4, 4, 0, 1, 61, 81, 2, 0]
        ])
        prob = trafic_dis / np.sum(trafic_dis)

        # 生成源-目的对
        Src_Dest_Pair, prob_arr, num_src_dest_pair = generate_src_dest_pairs(config.NODE_NUM, prob)
        logger.info(f"📊 生成了 {num_src_dest_pair} 个源-目的节点对")

        # 加载候选路径
        Candidate_Paths = load_candidate_paths(config.candidate_paths_file, config.NODE_NUM, config.N)

        # 创建全局计数器
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

        # 创建学习率调度器
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True
        )
        trainer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # 创建主网络
        logger.info("🚀 正在创建主网络...")
        master_network = AC_Net(
            scope='global',
            trainer=None,
            x_dim_p=config.x_dim_p,
            x_dim_v=config.x_dim_v,
            n_actions=config.n_actions,
            num_layers=config.num_layers,
            layer_size=config.layer_size,
            regu_scalar=config.regu_scalar
        )

        logger.info(
            f"✅ 主网络已初始化. 参数总数: {sum([tf.size(v).numpy() for v in master_network.trainable_variables]):,}")

        # 打印模型摘要
        master_network.summary()

        # 创建检查点
        checkpoint = tf.train.Checkpoint(
            optimizer=trainer,
            model=master_network,
            global_episodes=global_episodes
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            os.path.join(config.output_dir, 'checkpoints'),
            max_to_keep=5,
            checkpoint_name='ckpt'
        )

        # 加载模型（如果需要）
        if config.load_model:
            latest_checkpoint = checkpoint_manager.latest_checkpoint
            if latest_checkpoint:
                checkpoint.restore(latest_checkpoint)
                logger.info(f"✅ 模型已从检查点恢复: {latest_checkpoint}")
                logger.info(f"   继续从 Episode {global_episodes.numpy()} 开始训练")
            else:
                logger.warning("⚠️  未找到检查点，从头开始训练")

        # 确定代理数量
        num_agents = min(multiprocessing.cpu_count(), config.max_cpu)
        logger.info(f"🔧 使用 {num_agents} 个并行训练代理")

        # 创建训练监控器
        monitor = TrainingMonitor(config)

        # 创建代理
        for i in range(num_agents):
            agent = DeepRMSA_Agent(
                i, trainer, linkmap, config.LINK_NUM, config.NODE_NUM,
                config.SLOT_TOTAL, config.k_path, config.M, config.lambda_req,
                config.lambda_time, len(config.lambda_time), config.gamma,
                config.episode_size, config.batch_size, Src_Dest_Pair,
                Candidate_Paths, num_src_dest_pair, config.model_path,
                global_episodes, config.regu_scalar, config.x_dim_p,
                config.x_dim_v, config.n_actions, config.num_layers,
                config.layer_size, config.model2_flag, config.nonuniform,
                prob_arr, monitor, config
            )
            agents.append(agent)

        # 创建协调器
        coord = tf.train.Coordinator()

        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 启动代理线程
        logger.info("🏃 启动训练代理...")

        for i, agent in enumerate(agents):
            agent_rmsa = lambda ag=agent: ag.rmsa(coord, checkpoint_manager, master_network)
            t = threading.Thread(target=agent_rmsa, name=f"Agent-{i}")
            t.daemon = True
            t.start()
            sleep(0.5)  # 错开启动时间
            agent_threads.append(t)

        # 监控循环
        monitor_interval = 60  # 每60秒检查一次
        last_monitor_time = time()

        while not coord.should_stop():
            try:
                # 定期检查资源使用
                if time() - last_monitor_time > monitor_interval:
                    memory_ok, memory_gb = resource_monitor.check_memory()
                    if not memory_ok:
                        logger.error(f"❌ 内存使用超限: {memory_gb:.2f} GB > {config.memory_limit_gb} GB")
                        coord.request_stop()
                        break

                    # 记录资源使用情况
                    stats = resource_monitor.get_stats()
                    logger.info(f"📊 系统资源: CPU {stats['cpu_percent']:.1f}%, "
                                f"内存 {stats['memory_gb']:.2f} GB ({stats['memory_percent']:.1f}%), "
                                f"运行时间 {stats['runtime_second']:.2f} s")

                    last_monitor_time = time()

                # 检查线程是否存活
                alive_threads = [t for t in agent_threads if t.is_alive()]
                if len(alive_threads) < len(agent_threads):
                    logger.warning(f"⚠️  部分训练线程已停止: {len(alive_threads)}/{len(agent_threads)} 存活")
                    if len(alive_threads) == 0:
                        logger.error("❌ 所有训练线程已停止")
                        break

                sleep(1)  # 短暂休眠避免忙等待

            except KeyboardInterrupt:
                logger.info("\n⚠️  检测到键盘中断")
                coord.request_stop()
                break

        # 等待训练完成
        coord.join(agent_threads, stop_grace_period_secs=30)

    except KeyboardInterrupt:
        logger.info("\n⚠️  训练被用户中断")
        if coord:
            coord.request_stop()
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        if coord:
            coord.request_stop()
    finally:
        # 等待所有线程结束
        if coord and agent_threads:
            logger.info("⏳ 等待所有线程结束...")
            alive_count = sum(1 for t in agent_threads if t.is_alive())
            if alive_count > 0:
                logger.info(f"   {alive_count} 个线程仍在运行...")
                coord.join(agent_threads, stop_grace_period_secs=10)

        # 保存最终模型
        if checkpoint_manager:
            try:
                final_save_path = checkpoint_manager.save()
                logger.info(f"✅ 最终模型已保存至: {final_save_path}")
            except Exception as e:
                logger.error(f"❌ 保存最终模型时出错: {e}")

        # 保存训练统计
        if 'monitor' in locals() and monitor:
            stats = monitor.get_stats()
            stats['final_episode'] = global_episodes.numpy() if 'global_episodes' in locals() else 0

            # 添加资源使用统计
            if resource_monitor:
                stats.update(resource_monitor.get_stats())

            stats_file = os.path.join(config.output_dir, 'training_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"📊 训练统计已保存至: {stats_file}")

        logger.info("🎉 训练完成!")
        logger.info(f"   总训练回合数: {global_episodes.numpy() if 'global_episodes' in locals() else 'N/A'}")
        if 'monitor' in locals() and monitor:
            logger.info(f"   最佳阻塞率: {monitor.best_bp:.4f}")
            logger.info(f"   最佳奖励: {monitor.best_reward:.2f}")
            logger.info(f"   总训练时间: {monitor.get_stats()['total_time_hours']:.2f} 小时")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    tf.random.set_seed(42)

    main()