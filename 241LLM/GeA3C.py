# GeA3C.py (已修改并整合)

from __future__ import division
import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import os
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

# -----------------------------------------------------------
# --- 常量定义 (保持不变) ---
linkmap = defaultdict(lambda: defaultdict(lambda: None))
linkmap[1][2] = (0, 1050);
linkmap[2][1] = (3, 1050);
linkmap[1][3] = (1, 1500);
linkmap[3][1] = (6, 1500);
linkmap[1][8] = (2, 2400);
linkmap[8][1] = (22, 2400)
linkmap[2][3] = (4, 600);
linkmap[3][2] = (7, 600);
linkmap[2][4] = (5, 750);
linkmap[4][2] = (9, 750);
linkmap[3][6] = (8, 1800);
linkmap[6][3] = (15, 1800)
linkmap[4][5] = (10, 600);
linkmap[5][4] = (12, 600);
linkmap[4][11] = (11, 1950);
linkmap[11][4] = (32, 1950);
linkmap[5][6] = (13, 1200);
linkmap[6][5] = (16, 1200);
linkmap[5][7] = (14, 600);
linkmap[7][5] = (19, 600)
linkmap[6][10] = (17, 1050);
linkmap[10][6] = (29, 1050);
linkmap[6][14] = (18, 1800);
linkmap[14][6] = (41, 1800);
linkmap[7][8] = (20, 750);
linkmap[8][7] = (23, 750);
linkmap[7][10] = (21, 1350);
linkmap[10][7] = (30, 1350)
linkmap[8][9] = (24, 750);
linkmap[9][8] = (25, 750);
linkmap[9][10] = (26, 750);
linkmap[10][9] = (31, 750);
linkmap[9][12] = (27, 300);
linkmap[12][9] = (35, 300);
linkmap[9][13] = (28, 300);
linkmap[13][9] = (38, 300)
linkmap[11][12] = (33, 600);
linkmap[12][11] = (36, 600);
linkmap[11][13] = (34, 750);
linkmap[13][11] = (39, 750);
linkmap[12][14] = (37, 300);
linkmap[14][12] = (42, 300);
linkmap[13][14] = (40, 150);
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

# ==================== 修改开始 ====================
# 1. 添加统一的输出目录
# 详细注释: 定义一个基础输出目录，用于存放所有生成的文件，如模型、日志和性能指标。
# 这样便于管理和查找，而不是将文件散落在项目各处。
output_dir = "/data/chl/pyproject/DeepRSA/output/"

# 2. 修改 model_path 以使用新的输出目录
# 详细注释: 将模型保存路径指向统一定义的输出文件夹中的 'model' 子文件夹。
# 使用 os.path.join 确保路径拼接的跨平台兼容性。
load_model = False
model_path = os.path.join(output_dir, 'model') # 修改此行

# 详细注释: 确保基础输出目录和模型目录都存在，如果不存在则创建。
# exist_ok=True 参数可以避免在目录已存在时抛出错误。
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
    print(f"💡 输出目录已创建: {model_path}")
# ==================== 修改结束 ====================


with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.keras.optimizers.Adam(learning_rate=1e-5)

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
    # ==================== 修改开始 ====================
    # 3. 优化控制台输出
    # 详细注释: 使用更友好和信息丰富的格式输出状态信息，增加表情符号以提高可读性。
    print("✅ 全局主网络构建成功。")
    # ==================== 修改结束 ====================

    # --- 修复: 添加优化器初始化代码 ---
    # 手动初始化优化器状态
    # 这是一个关键步骤，可以防止在多线程中出现 'Adam' object has no attribute '_momentums' 的错误
    zero_grads = [tf.zeros_like(v) for v in master_network.trainable_variables]
    trainer.apply_gradients(zip(zero_grads, master_network.trainable_variables))
    print("✅ 优化器状态初始化成功。")

    num_agents = multiprocessing.cpu_count()
    if num_agents > max_cpu:
        num_agents = max_cpu

    agents = []
    for i in range(num_agents):
        # ==================== 修改开始 ====================
        # 4. 向 Agent 传递输出目录
        # 详细注释: 在创建每个 Agent 实例时，将统一的输出目录 `output_dir` 作为参数传入。
        # 这使得 Agent 内部也能知道文件应该保存在哪里，确保了所有输出路径的一致性。
        agents.append(DeepRMSA_Agent(i,
                                     trainer,
                                     linkmap, LINK_NUM, NODE_NUM, SLOT_TOTAL, k_path, M,
                                     lambda_req, lambda_time, len_lambda_time, gamma,
                                     episode_size, batch_size, Src_Dest_Pair, Candidate_Paths,
                                     num_src_dest_pair, model_path, global_episodes,
                                     regu_scalar, x_dim_p, x_dim_v, n_actions, num_layers,
                                     layer_size, model2_flag, nonuniform, prob_arr,
                                     output_dir)) # 添加 output_dir 参数
        # ==================== 修改结束 ====================


    checkpoint = tf.train.Checkpoint(step=global_episodes, optimizer=trainer, model=master_network)
    manager = tf.train.CheckpointManager(checkpoint, directory=model_path, max_to_keep=5)

coord = tf.train.Coordinator()

if load_model:
    status = checkpoint.restore(manager.latest_checkpoint)
    # ==================== 修改开始 ====================
    # 5. 优化模型加载时的控制台输出
    # 详细注释: 提供更清晰的日志，明确告知用户模型是从检查点恢复还是从头开始初始化。
    if manager.latest_checkpoint:
        print(f"✅ 模型成功从检查点恢复: {manager.latest_checkpoint}")
    else:
        print("⚠️ 未找到模型检查点，将从头开始初始化。")
else:
    print("ℹ️ 'load_model' 设置为 False，将从头开始初始化模型。")
    # ==================== 修改结束 ====================

agent_threads = []
for agent in agents:
    agent_rmsa = lambda: agent.rmsa(master_network, coord, manager)
    t = threading.Thread(target=(agent_rmsa))
    t.start()
    sleep(0.5)
    agent_threads.append(t)

coord.join(agent_threads)