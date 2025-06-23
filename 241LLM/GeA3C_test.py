# GeA3C.py (优化日志输出版本)

from __future__ import division
import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import os
import logging
from random import choice
from time import sleep
from time import time
import scipy.signal
import struct
from collections import defaultdict
from GeACNet import AC_Net
from GeAgent import DeepRMSA_Agent

# author Xiaoliang Chen, xlichen@ucdavis.edu
# copyright NGNS lab @ucdavis

# key features: uniform/nonuniform traffic distribution; window-based training; policy embedded with epsilon-greedy approach

# 设置日志目录
LOG_DIR = "/data/chl/pyproject/DeepRSA/output/"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 配置日志记录器
def setup_logger(name, log_file, level=logging.INFO, console=False):
    """设置日志记录器，分别输出到文件和控制台（可选）"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # 清除现有处理器，防止重复添加

    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # 控制台处理器（可选）
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    return logger

# 初始化日志记录器
model_logger = setup_logger('model', 'model.log')
runtime_logger = setup_logger('runtime', 'runtime.log')
console_logger = setup_logger('console', 'console.log', console=True)

# -----------------------------------------------------------
# ... (linkmap, nonuniform, trafic_dis 等所有常量定义保持不变) ...
linkmap = defaultdict(lambda: defaultdict(lambda: None))
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
nonuniform = False
trafic_dis = [[0, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 1], [2, 0, 2, 1, 8, 2, 1, 5, 3, 5, 1, 5, 1, 4],
              [1, 2, 0, 2, 3, 2, 11, 20, 5, 2, 1, 1, 1, 2], [1, 1, 2, 0, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2],
              [1, 8, 3, 1, 0, 3, 3, 7, 3, 3, 1, 5, 2, 5], [4, 2, 2, 1, 3, 0, 2, 1, 2, 2, 1, 1, 1, 2],
              [1, 1, 11, 2, 3, 2, 0, 9, 4, 20, 1, 8, 1, 4], [1, 5, 20, 1, 7, 1, 9, 0, 27, 7, 2, 3, 2, 4],
              [2, 3, 5, 2, 3, 2, 4, 27, 0, 75, 2, 9, 3, 1], [1, 5, 2, 2, 3, 2, 20, 7, 75, 0, 1, 1, 2, 1],
              [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 2, 1, 61], [1, 5, 1, 2, 5, 1, 8, 3, 9, 1, 2, 0, 1, 81],
              [1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 0, 2], [1, 4, 2, 2, 5, 2, 4, 4, 0, 1, 61, 81, 2, 0]]
prob = np.array(trafic_dis) / np.sum(trafic_dis)
LINK_NUM = 44
NODE_NUM = 14
model2_flag = 0
N = 10
M = 1
k_path = 5
n_actions = k_path * M
x_dim_p = NODE_NUM * 2 + k_path * (1 + M * 2 + 2 + model2_flag * 3)
x_dim_v = NODE_NUM * 2 + k_path * (1 + M * 2 + 2 + model2_flag * 3)
num_layers = 5
layer_size = 128
regu_scalar = 1e-4
max_cpu = 4
lambda_req = 12
lambda_time = [14]
SLOT_TOTAL = 100
len_lambda_time = len(lambda_time)
gamma = 0.95
episode_size = 1000
batch_size = 200
Src_Dest_Pair = []
prob_arr = []
for ii in range(NODE_NUM):
    for jj in range(NODE_NUM):
        if ii != jj:
            prob_arr.append(prob[ii][jj])
            temp = []
            temp.append(ii + 1)
            temp.append(jj + 1)
            Src_Dest_Pair.append(temp)
num_src_dest_pair = len(Src_Dest_Pair)
prob_arr[-1] += 1 - sum(prob_arr)
Candidate_Paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
fp = open('Src_Dst_Paths.dat', 'rb')
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
            temp_path = list(struct.unpack("i" * temp_num[k], fp.read(4 * temp_num[k])))
            Candidate_Paths[i][j][k] = temp_path
fp.close()
# -----------------------------------------------------------

load_model = False
model_path = 'model'

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model_logger.info("Initializing global master network...")
    master_network = AC_Net(scope='global',
                            x_dim_p=x_dim_p,
                            x_dim_v=x_dim_v,
                            n_actions=n_actions,
                            num_layers=num_layers,
                            layer_size=layer_size,
                            regu_scalar=regu_scalar)

    dummy_input_p = tf.random.normal((1, x_dim_p))
    dummy_input_v = tf.random.normal((1, x_dim_v))
    master_network((dummy_input_p, dummy_input_v))
    console_logger.info("Global master network built successfully.")
    model_logger.info("Global master network built with %d layers, layer size %d, and %d actions.", num_layers, layer_size, n_actions)

    # 修复: 在此处添加优化器初始化代码
    model_logger.info("Initializing optimizer state...")
    zero_grads = [tf.zeros_like(v) for v in master_network.trainable_variables]
    trainer.apply_gradients(zip(zero_grads, master_network.trainable_variables))
    console_logger.info("Optimizer state initialized.")
    model_logger.info("Optimizer state initialized with Adam optimizer, learning rate: %f.", 1e-5)

    num_agents = multiprocessing.cpu_count()
    if num_agents > max_cpu:
        num_agents = max_cpu
    console_logger.info("Starting %d agent threads (limited by max_cpu=%d).", num_agents, max_cpu)
    model_logger.info("Configured %d agents for parallel training.", num_agents)

    agents = []
    for i in range(num_agents):
        agents.append(DeepRMSA_Agent(i,
                                     trainer,
                                     linkmap, LINK_NUM, NODE_NUM, SLOT_TOTAL, k_path, M,
                                     lambda_req, lambda_time, len_lambda_time, gamma,
                                     episode_size, batch_size, Src_Dest_Pair, Candidate_Paths,
                                     num_src_dest_pair, model_path, global_episodes,
                                     regu_scalar, x_dim_p, x_dim_v, n_actions, num_layers,
                                     layer_size, model2_flag, nonuniform, prob_arr))
        runtime_logger.debug("Agent %d initialized with episode_size=%d, batch_size=%d.", i, episode_size, batch_size)

    checkpoint = tf.train.Checkpoint(step=global_episodes, optimizer=trainer, model=master_network)
    manager = tf.train.CheckpointManager(checkpoint, directory=model_path, max_to_keep=5)

if load_model:
    status = checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        console_logger.info("Restored model from checkpoint: %s", manager.latest_checkpoint)
        model_logger.info("Restored model from checkpoint: %s", manager.latest_checkpoint)
    else:
        console_logger.warning("No checkpoint found, initializing from scratch.")
        model_logger.warning("No checkpoint found, initializing from scratch.")
else:
    console_logger.info("Initializing training from scratch (no model loaded).")
    model_logger.info("Initializing training from scratch (no model loaded).")

agent_threads = []
for i, agent in enumerate(agents):
    agent_rmsa = lambda: agent.rmsa(master_network, coord, manager)
    t = threading.Thread(target=agent_rmsa, name=f"Agent-{i}")
    t.start()
    sleep(0.5)
    agent_threads.append(t)
    runtime_logger.info("Started thread for Agent-%d.", i)
    console_logger.debug("Thread for Agent-%d started.", i)

console_logger.info("All %d agent threads started, waiting for completion...", len(agent_threads))
coord = tf.train.Coordinator()
coord.join(agent_threads)
console_logger.info("All agent threads completed.")
runtime_logger.info("All agent threads completed.")