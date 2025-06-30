from __future__ import division
import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import os
import sys
from random import choice
from time import sleep
from time import time
import scipy.signal
import struct
from collections import defaultdict, deque
from ACNettest import AC_Net
from Agenttest import DeepRMSA_Agent
import logging
import signal

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# author Xiaoliang Chen, xlichen@ucdavis.edu
# copyright NGNS lab @ucdavis

# key features: uniform/nonuniform traffic distribution; window-based training; policy embedded with epsilon-greedy approach

# -----------------------------------------------------------

# 定义网络拓扑
def create_linkmap():
    """创建NSFNet拓扑"""
    linkmap = defaultdict(lambda: defaultdict(lambda: None))

    # NSFNet topology
    linkmap[1][2] = (0, 1050)
    linkmap[2][1] = (3, 1050)
    linkmap[1][3] = (1, 1500)
    linkmap[3][1] = (6, 1500)
    linkmap[1][8] = (2, 2400)
    linkmap[8][1] = (22, 2400)

    linkmap[2][3] = (4, 600)
    linkmap[3][2] = (7, 600)
    linkmap[2][4] = (5, 750)
    linkmap[4][2] = (9, 750)
    linkmap[3][6] = (8, 1800)
    linkmap[6][3] = (15, 1800)

    linkmap[4][5] = (10, 600)
    linkmap[5][4] = (12, 600)
    linkmap[4][11] = (11, 1950)
    linkmap[11][4] = (32, 1950)
    linkmap[5][6] = (13, 1200)
    linkmap[6][5] = (16, 1200)
    linkmap[5][7] = (14, 600)
    linkmap[7][5] = (19, 600)

    linkmap[6][10] = (17, 1050)
    linkmap[10][6] = (29, 1050)
    linkmap[6][14] = (18, 1800)
    linkmap[14][6] = (41, 1800)
    linkmap[7][8] = (20, 750)
    linkmap[8][7] = (23, 750)
    linkmap[7][10] = (21, 1350)
    linkmap[10][7] = (30, 1350)

    linkmap[8][9] = (24, 750)
    linkmap[9][8] = (25, 750)
    linkmap[9][10] = (26, 750)
    linkmap[10][9] = (31, 750)
    linkmap[9][12] = (27, 300)
    linkmap[12][9] = (35, 300)
    linkmap[9][13] = (28, 300)
    linkmap[13][9] = (38, 300)

    linkmap[11][12] = (33, 600)
    linkmap[12][11] = (36, 600)
    linkmap[11][13] = (34, 750)
    linkmap[13][11] = (39, 750)
    linkmap[12][14] = (37, 300)
    linkmap[14][12] = (42, 300)
    linkmap[13][14] = (40, 150)
    linkmap[14][13] = (43, 150)

    return linkmap


# 创建链路映射
linkmap = create_linkmap()


# 配置参数
class Config:
    """配置类"""

    def __init__(self):
        # 网络参数
        self.LINK_NUM = 44
        self.NODE_NUM = 14
        self.SLOT_TOTAL = 100

        # 路径参数
        self.N = 10  # number of paths each src-dest pair
        self.M = 1  # first M starting FS allocation positions are considered
        self.k_path = 5
        self.n_actions = self.k_path * self.M

        # 模型参数
        self.model2_flag = 0
        self.num_layers = 5
        self.layer_size = 128
        self.regu_scalar = 1e-4

        # 训练参数
        self.max_cpu = 4
        self.lambda_req = 12
        self.lambda_time = [14]
        self.gamma = 0.95
        self.episode_size = 1000
        self.batch_size = 200

        # 流量参数
        self.nonuniform = False

        # 输入维度
        self.x_dim_p = self.NODE_NUM * 2 + self.k_path * (
                1 + self.M * 2 + 2 + self.model2_flag * 3)
        self.x_dim_v = self.x_dim_p

        # 输出路径
        self.output_dir = "/data/chl/pyproject/DeepRSA/output2/"
        self.model_path = os.path.join(self.output_dir, 'model')
        self.load_model = False


# 创建配置实例
config = Config()

# 流量分布矩阵（用于非均匀流量）
trafic_dis = [[0, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 1],
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
              [1, 4, 2, 2, 5, 2, 4, 4, 0, 1, 61, 81, 2, 0]]

prob = np.array(trafic_dis) / np.sum(trafic_dis)


def load_candidate_paths(path_file, NODE_NUM, N):
    """加载候选路径"""
    Candidate_Paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

    if not os.path.exists(path_file):
        logger.warning(f"路径文件 {path_file} 不存在，将使用默认路径生成策略")
        # 生成一些默认路径或使用其他策略
        # 这里可以实现一个简单的K最短路径算法
        return Candidate_Paths

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
                    temp_num += list(struct.unpack("i" * 1, fp.read(4 * 1)))

                if i != j:
                    for k in range(N):
                        if temp_num[k] > 0:
                            temp_path = list(struct.unpack("i" * temp_num[k], fp.read(4 * temp_num[k])))
                            Candidate_Paths[i][j][k] = temp_path

        logger.info(f"成功加载候选路径文件: {path_file}")
    except Exception as e:
        logger.error(f"加载路径文件时出错: {e}")

    return Candidate_Paths


def generate_src_dest_pairs(NODE_NUM, prob):
    """生成源-目的节点对"""
    Src_Dest_Pair = []
    prob_arr = []

    for ii in range(NODE_NUM):
        for jj in range(NODE_NUM):
            if ii != jj:
                prob_arr.append(prob[ii][jj])
                temp = [ii + 1, jj + 1]
                Src_Dest_Pair.append(temp)

    num_src_dest_pair = len(Src_Dest_Pair)
    prob_arr = np.array(prob_arr)
    prob_arr = prob_arr / prob_arr.sum()  # 确保概率和为1

    return Src_Dest_Pair, prob_arr, num_src_dest_pair


def setup_environment():
    """设置环境和创建输出目录"""
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path, exist_ok=True)
        logger.info(f"💡 输出目录已创建: {config.model_path}")
    else:
        logger.info(f"📁 使用现有输出目录: {config.model_path}")


def signal_handler(signum, frame):
    """处理中断信号"""
    logger.info("\n⚠️  接收到中断信号，正在优雅地停止训练...")
    coord.request_stop()


# 主程序
def main():
    """主函数"""
    # 设置环境
    setup_environment()

    # 生成源-目的对
    Src_Dest_Pair, prob_arr, num_src_dest_pair = generate_src_dest_pairs(config.NODE_NUM, prob)

    # 加载候选路径
    path_file = '/data/chl/pyproject/DeepRSA/Src_Dst_Paths.dat'
    Candidate_Paths = load_candidate_paths(path_file, config.NODE_NUM, config.N)

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

    logger.info(f"✅ 主网络已初始化. 策略变量数: {len(master_network.get_policy_variables())}, "
                f"价值变量数: {len(master_network.get_value_variables())}")

    # 创建检查点
    checkpoint = tf.train.Checkpoint(
        optimizer=trainer,
        model=master_network,
        global_episodes=global_episodes
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        config.model_path,
        max_to_keep=5
    )

    # 加载模型（如果需要）
    if config.load_model:
        latest_checkpoint = checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            logger.info(f"✅ 模型已从检查点恢复: {latest_checkpoint}")
        else:
            logger.warning("⚠️  未找到检查点，从头开始训练")

    # 确定代理数量
    num_agents = min(multiprocessing.cpu_count(), config.max_cpu)
    logger.info(f"🔧 使用 {num_agents} 个并行训练代理")

    # 创建代理
    agents = []
    for i in range(num_agents):
        agent = DeepRMSA_Agent(
            i, trainer, linkmap, config.LINK_NUM, config.NODE_NUM,
            config.SLOT_TOTAL, config.k_path, config.M, config.lambda_req,
            config.lambda_time, len(config.lambda_time), config.gamma,
            config.episode_size, config.batch_size, Src_Dest_Pair,
            Candidate_Paths, num_src_dest_pair, config.model_path,
            global_episodes, config.regu_scalar, config.x_dim_p,
            config.x_dim_v, config.n_actions, config.num_layers,
            config.layer_size, config.model2_flag, config.nonuniform, prob_arr
        )
        agents.append(agent)

    # 创建协调器
    global coord
    coord = tf.train.Coordinator()

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动代理线程
    agent_threads = []
    logger.info("🏃 启动训练代理...")

    for agent in agents:
        agent_rmsa = lambda ag=agent: ag.rmsa(coord, checkpoint_manager, master_network)
        t = threading.Thread(target=agent_rmsa, name=f"Agent-{agent.name}")
        t.daemon = True  # 设置为守护线程
        t.start()
        sleep(0.5)  # 错开启动时间
        agent_threads.append(t)

    try:
        # 等待所有线程完成
        coord.join(agent_threads)
    except KeyboardInterrupt:
        logger.info("\n⚠️  训练被用户中断")
        coord.request_stop()
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {e}")
        coord.request_stop()
    finally:
        # 等待所有线程结束
        logger.info("⏳ 等待所有线程结束...")
        coord.join(agent_threads, stop_grace_period_secs=10)

        # 保存最终模型
        if not config.load_model:
            try:
                final_save_path = checkpoint_manager.save()
                logger.info(f"✅ 最终模型已保存至: {final_save_path}")
            except Exception as e:
                logger.error(f"❌ 保存最终模型时出错: {e}")

        logger.info("🎉 训练完成!")


if __name__ == "__main__":
    main()