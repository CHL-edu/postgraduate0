from __future__ import division
from collections import defaultdict, deque
import struct
import types
import string
import numpy as np
import math
import copy
import random
import datetime
from ACNettesttest import AC_Net
import threading
import multiprocessing
import tensorflow as tf
import scipy.signal
from random import choice
from time import sleep
from time import time
from collections import defaultdict
import os
import logging
from contextlib import contextmanager
import json

# 配置基本日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@contextmanager
def safe_file_write(filepath, mode='a'):
    """安全的文件写入上下文管理器，处理潜在的IO异常。"""
    f = None
    try:
        f = open(filepath, mode)
        yield f
    except IOError as e:
        logging.error(f"无法写入文件 {filepath}: {e}")
        yield None
    finally:
        if f is not None:
            try:
                f.close()
            except:
                pass


class RunningStats:
    """用于动态特征归一化的运行时统计"""

    def __init__(self, shape=()):
        self.n = 0
        self.mean = np.zeros(shape)
        self.M2 = np.zeros(shape)

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_stats(self):
        if self.n < 2:
            return self.mean, np.ones_like(self.mean)
        else:
            variance = self.M2 / (self.n - 1)
            return self.mean, np.sqrt(variance + 1e-8)


class ExperienceBuffer:
    """高效的经验回放缓冲区"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def get_batch(self, batch_size):
        """获取一批数据（不移除）"""
        if len(self.buffer) < batch_size:
            # 返回所有非None的元素
            return [exp for exp in self.buffer if exp is not None]

        # 随机采样，过滤None元素
        valid_buffer = [exp for exp in self.buffer if exp is not None]
        if len(valid_buffer) < batch_size:
            return valid_buffer

        return random.sample(valid_buffer, batch_size)

    def clear(self):
        self.buffer.clear()
        self.position = 0

    def __len__(self):
        return sum(1 for exp in self.buffer if exp is not None)


class DeepRMSA_Agent():
    _episode_lock = threading.Lock()
    _file_lock = threading.Lock()
    _slot_map_lock = threading.Lock()

    def __init__(self,
                 id,
                 trainer,
                 linkmap,
                 LINK_NUM,
                 NODE_NUM,
                 SLOT_TOTAL,
                 k_path,
                 M,
                 lambda_req,
                 lambda_time,
                 len_lambda_time,
                 gamma,
                 episode_size,
                 batch_size,
                 Src_Dest_Pair,
                 Candidate_Paths,
                 num_src_dest_pair,
                 model_path,
                 global_episodes,
                 regu_scalar,
                 x_dim_p,
                 x_dim_v,
                 n_actions,
                 num_layers,
                 layer_size,
                 model2_flag,
                 nonuniform,
                 prob_arr,
                 monitor=None,
                 config=None):
        self.name = 'agent_' + str(id)
        self.trainer = trainer
        self.linkmap = linkmap
        self.LINK_NUM = LINK_NUM
        self.NODE_NUM = NODE_NUM
        self.SLOT_TOTAL = SLOT_TOTAL
        self.k_path = k_path
        self.M = M
        self.lambda_req = lambda_req
        self.lambda_time = lambda_time
        self.len_lambda_time = len_lambda_time
        self.gamma = gamma
        self.episode_size = episode_size
        self.batch_size = batch_size
        self.Src_Dest_Pair = Src_Dest_Pair
        self.Candidate_Paths = Candidate_Paths
        self.num_src_dest_pair = num_src_dest_pair
        self.model_path = model_path
        self.model2_flag = model2_flag
        self.nonuniform = nonuniform
        self.prob_arr = prob_arr
        self.global_episodes = global_episodes
        self.monitor = monitor
        self.config = config

        self.episode_rewards = []
        self.episode_blocking = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.create_file_writer(os.path.join(self.model_path, "train_" + self.name))

        self.x_dim_p = x_dim_p
        self.x_dim_v = x_dim_v
        self.n_actions = n_actions

        # 改进的epsilon衰减策略
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.05
        self.epsilon_decay_steps = 200000
        self.epsilon_decay_rate = 0.995

        self.local_network = AC_Net(scope=self.name,
                                    trainer=self.trainer,
                                    x_dim_p=self.x_dim_p,
                                    x_dim_v=self.x_dim_v,
                                    n_actions=self.n_actions,
                                    num_layers=num_layers,
                                    layer_size=layer_size,
                                    regu_scalar=regu_scalar)

        # 使用numpy数组以提高性能
        self.slot_map = np.ones((self.LINK_NUM, self.SLOT_TOTAL), dtype=np.int8)
        self.slot_map_t = np.zeros((self.LINK_NUM, self.SLOT_TOTAL), dtype=np.float32)

        self.service_time = self.lambda_time[np.random.randint(0, self.len_lambda_time)]
        self.lambda_intervals = 1 / self.lambda_req
        self.request_set = {}

        # 使用运行时统计进行动态归一化
        self.feature_stats = {
            'num_fs': RunningStats(),
            'slot_position': RunningStats(),
            'slot_continue': RunningStats(),
            'total_available': RunningStats(),
            'mean_size': RunningStats()
        }

        # 使用改进的经验缓冲区
        self.MAX_BUFFER_SIZE = 2000
        self.episode_buffer = ExperienceBuffer(self.MAX_BUFFER_SIZE)

        self.request_counter = 0

    def _get_path(self, src, dst, Candidate_Paths, k):
        """获取从源到目的地的第k条路径 - 安全版本"""
        if src == dst:
            return []

        try:
            # 使用嵌套的get方法安全访问
            src_paths = Candidate_Paths.get(src, {})
            dst_paths = src_paths.get(dst, {})
            path = dst_paths.get(k, None)

            if path is not None and isinstance(path, list):
                return path
        except Exception as e:
            logging.debug(f"获取路径时出错 src:{src}, dst:{dst}, k:{k}: {e}")

        return []

    def calclink(self, p):
        """计算路径上的链路"""
        if len(p) < 2:
            return []

        path_link = []
        for a, b in zip(p[:-1], p[1:]):
            link_info = self.linkmap.get(a, {}).get(b, None)
            if link_info:
                path_link.append(link_info[0])
            else:
                logging.debug(f"链路 {a}->{b} 不存在")
                return []
        return path_link

    def get_new_slot_temp_safe(self, slot_temp, path_link, slot_map):
        """安全地获取路径上的可用频谱槽"""
        with self._slot_map_lock:
            slot_temp_array = np.array(slot_temp, dtype=np.int8)

            for link_id in path_link:
                if 0 <= link_id < slot_map.shape[0]:
                    slot_temp_array &= slot_map[link_id][:len(slot_temp_array)]

            return slot_temp_array.tolist()

    def mark_vector(self, vector, default):
        """标记连续的频谱块 - 优化版本"""
        if not vector:
            return 0, [], []

        vector_array = np.array(vector)
        is_available = (vector_array == default)

        if not np.any(is_available):
            return 0, [], []

        # 找到连续块的起始和结束位置
        diff = np.diff(np.concatenate(([0], is_available.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        slotflag = starts.tolist()
        slotscontinue = (ends - starts).tolist()

        return len(starts), slotflag, slotscontinue

    def judge_availability(self, slot_temp, current_slots, FS_id):
        """判断频谱块是否可用 - 修正版本"""
        flag, slotflag, slotscontinue = self.mark_vector(slot_temp, 1)

        if flag <= 0 or not slotscontinue:
            return 0, -1, -1

        # 找到所有足够大的频谱块
        valid_blocks = []
        for i in range(len(slotscontinue)):
            if slotscontinue[i] >= current_slots:
                fs = slotflag[i]
                fe = fs + current_slots - 1

                # 验证范围有效性和可用性
                if fe < len(slot_temp):
                    # 使用numpy切片检查
                    if np.all(np.array(slot_temp[fs:fe + 1]) == 1):
                        valid_blocks.append((i, fs, slotscontinue[i]))

        if not valid_blocks:
            return 0, -1, -1

        # 使用首次适应(First Fit)或最佳适应(Best Fit)策略
        # 这里使用首次适应，但限制FS_id在有效范围内
        if FS_id >= len(valid_blocks):
            # 如果FS_id超出范围，使用模运算映射到有效范围
            # 这保证了动作空间的连续性
            actual_fs_id = FS_id % len(valid_blocks)
        else:
            actual_fs_id = FS_id

        _, fs, _ = valid_blocks[actual_fs_id]
        fe = fs + current_slots - 1

        return 1, fs, fe

    def update_slot_map_for_committing_wp(self, slot_map, current_wp_link, current_fs, current_fe, slot_map_t,
                                          current_TTL):
        """提交工作路径时更新频谱图 - numpy优化版本"""
        with self._slot_map_lock:
            for link_id in current_wp_link:
                if 0 <= link_id < len(slot_map) and current_fe < slot_map.shape[1]:
                    slot_map[link_id, current_fs:current_fe + 1] = 0
                    slot_map_t[link_id, current_fs:current_fe + 1] = current_TTL

        return slot_map, slot_map_t

    def update_slot_map_for_releasing_wp(self, slot_map, current_wp_link, current_fs, current_fe):
        """释放工作路径时更新频谱图 - numpy优化版本"""
        with self._slot_map_lock:
            for link_id in current_wp_link:
                if 0 <= link_id < len(slot_map) and current_fe < slot_map.shape[1]:
                    slot_map[link_id, current_fs:current_fe + 1] = 1

        return slot_map

    def release(self, slot_map, request_set, slot_map_t, time_to):
        """释放过期的请求 - 修正版本"""
        with self._slot_map_lock:
            del_ids = []

            # 修正：只更新被占用槽位的TTL
            # 创建掩码，标识被占用的槽位
            occupied_mask = (slot_map == 0)

            # 只对被占用的槽位更新TTL
            slot_map_t = np.where(occupied_mask,
                                  np.maximum(0, slot_map_t - time_to),
                                  slot_map_t)

            # 处理每个请求
            for req_id, req_info in request_set.items():
                links, fs, fe, ttl = req_info
                new_ttl = ttl - time_to

                if new_ttl <= 0:
                    del_ids.append(req_id)
                    # 释放频谱
                    for link in links:
                        if 0 <= link < slot_map.shape[0] and fe < slot_map.shape[1]:
                            slot_map[link, fs:fe + 1] = 1
                            slot_map_t[link, fs:fe + 1] = 0
                else:
                    request_set[req_id][3] = new_ttl

            # 删除过期请求
            for req_id in del_ids:
                del request_set[req_id]

            # 清理过期但未释放的槽（应该不会发生，但作为安全措施）
            expired_mask = (slot_map_t <= 0) & (slot_map == 0)
            slot_map[expired_mask] = 1
            slot_map_t[expired_mask] = 0

        return slot_map, request_set, slot_map_t

    def cal_len(self, path):
        """计算路径长度"""
        if len(path) < 2:
            return 0

        path_len = 0
        for a, b in zip(path[:-1], path[1:]):
            link_info = self.linkmap.get(a, {}).get(b, None)
            if link_info:
                path_len += link_info[1]
            else:
                logging.debug(f"链路 {a}->{b} 长度未定义")
                return 0
        return path_len

    def cal_FS(self, bandwidth, path_len):
        """根据带宽和路径长度计算所需频谱槽数 - 修正版本"""
        # 输入验证
        if bandwidth <= 0:
            logging.warning(f"无效的带宽值: {bandwidth}")
            return 0

        if path_len <= 0:
            logging.warning(f"无效的路径长度: {path_len}")
            return 0

        # 计算频谱槽数
        if path_len <= 625:
            num_FS = math.ceil(bandwidth / (4 * 12.5)) + 1
        elif path_len <= 1250:
            num_FS = math.ceil(bandwidth / (3 * 12.5)) + 1
        elif path_len <= 2500:
            num_FS = math.ceil(bandwidth / (2 * 12.5)) + 1
        else:
            num_FS = math.ceil(bandwidth / (1 * 12.5)) + 1

        return int(num_FS)

    def discount(self, x):
        """计算折扣奖励"""
        return scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]

    def get_epsilon(self):
        """获取当前的epsilon值（改进的指数衰减）"""
        steps = self.global_episodes.numpy() * self.episode_size
        if steps >= self.epsilon_decay_steps:
            return self.final_epsilon
        else:
            # 指数衰减
            decay_factor = self.epsilon_decay_rate ** (steps / 1000)
            return self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * decay_factor

    def normalize_feature(self, value, feature_name):
        """使用运行时统计进行特征归一化"""
        if feature_name in self.feature_stats:
            mean, std = self.feature_stats[feature_name].get_stats()
            return (value - mean) / (std + 1e-8)
        return value

    def update_feature_stats(self, feature_name, value):
        """更新特征统计"""
        if feature_name in self.feature_stats:
            self.feature_stats[feature_name].update(value)

    @tf.function(reduce_retracing=True)
    def train_step(self, local_network, input_p, input_v, actions, advantages, target_v):
        """单步训练"""
        with tf.GradientTape() as tape:
            policy_total_loss, policy_loss, entropy = local_network.get_policy_loss(input_p, actions, advantages)
            value_loss = local_network.get_value_loss(input_v, target_v)
            total_loss = policy_total_loss + value_loss * 0.5

        gradients = tape.gradient(total_loss, local_network.trainable_variables)
        return value_loss, policy_loss, entropy, gradients

    def train(self, episode_buffer_data, global_network, bootstrap_value):
        """训练网络 - 改进的批量处理"""
        try:
            if len(episode_buffer_data) == 0:
                return (0.0, 0.0, 0.0)

            # 提取数据
            input_p = np.vstack([item[0] for item in episode_buffer_data])
            input_v = np.vstack([item[1] for item in episode_buffer_data])
            actions = np.array([item[2] for item in episode_buffer_data])
            rewards = np.array([item[3] for item in episode_buffer_data])
            values = np.array([item[4] for item in episode_buffer_data])

            # 计算折扣奖励和优势
            rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
            discounted_rewards = self.discount(rewards_plus)[:-1]

            value_plus = np.asarray(values.tolist() + [bootstrap_value])
            advantages = discounted_rewards - value_plus[:-1]

            # 优势标准化
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 转换为张量
            input_p_tensor = tf.convert_to_tensor(input_p, dtype=tf.float32)
            input_v_tensor = tf.convert_to_tensor(input_v, dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
            advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
            target_v_tensor = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

            # 获取损失和梯度
            v_l, p_l, e_l, combined_grads = self.train_step(
                self.local_network, input_p_tensor, input_v_tensor,
                actions_tensor, advantages_tensor, target_v_tensor
            )

            # 梯度裁剪和应用
            all_global_vars = global_network.trainable_variables
            valid_grads_and_vars = [(grad, var) for grad, var in zip(combined_grads, all_global_vars) if
                                    grad is not None]

            if valid_grads_and_vars:
                valid_grads, valid_vars = zip(*valid_grads_and_vars)
                clipped_grads, _ = tf.clip_by_global_norm(valid_grads, 40.0)
                self.trainer.apply_gradients(zip(clipped_grads, valid_vars))

            batch_size = len(episode_buffer_data)
            return (v_l.numpy() / batch_size, p_l.numpy() / batch_size, e_l.numpy() / batch_size)

        except Exception as e:
            logging.error(f"训练过程中出错: {e}")
            return (0.0, 0.0, 0.0)

    def update_target_graph(self, global_network, local_network):
        """从全局网络更新本地网络参数"""
        try:
            for global_var, local_var in zip(global_network.trainable_variables, local_network.trainable_variables):
                local_var.assign(global_var)
        except Exception as e:
            logging.error(f"更新目标网络时出错: {e}")

    def rmsa(self, coord, checkpoint_manager, global_network):
        """主训练循环 - 修正版本"""
        total_steps = 0
        action_onehot = list(range(self.n_actions))
        sd_onehot = list(range(self.num_src_dest_pair))
        node_onehot = np.eye(self.NODE_NUM).tolist()
        all_nega_ones = [-1] * (3 + 2 * self.M)

        print(f'Starting {self.name}')
        episode_times = deque(maxlen=100)

        while not coord.should_stop():
            try:
                episode_start_time = time()
                self.update_target_graph(global_network, self.local_network)

                episode_count = self.global_episodes.numpy()
                mean_value_loss, mean_policy_loss, mean_entropy = 0.0, 0.0, 0.0

                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                num_blocks = 0

                # 用于批量更新特征统计
                feature_updates = defaultdict(list)

                self.episode_buffer.clear()

                while episode_step_count < self.episode_size:
                    # 释放过期请求
                    time_to = np.random.exponential(self.lambda_intervals)
                    self.slot_map, self.request_set, self.slot_map_t = self.release(
                        self.slot_map, self.request_set, self.slot_map_t, time_to)

                    # 生成新请求
                    if self.nonuniform:
                        sd_id = np.random.choice(sd_onehot, p=self.prob_arr)
                        temp = self.Src_Dest_Pair[sd_id]
                    else:
                        temp = self.Src_Dest_Pair[np.random.randint(0, self.num_src_dest_pair)]

                    current_src, current_dst = temp[0], temp[1]
                    current_bandwidth = np.random.randint(25, 101)
                    current_TTL = np.random.exponential(self.service_time)

                    # 构建输入特征
                    Input_feature = []
                    Input_feature += node_onehot[current_src - 1]
                    Input_feature += node_onehot[current_dst - 1]

                    for x in range(self.k_path):
                        path = self._get_path(current_src, current_dst, self.Candidate_Paths, x)
                        if not path:
                            Input_feature += all_nega_ones
                            continue

                        path_len = self.cal_len(path)
                        if path_len == 0:
                            Input_feature += all_nega_ones
                            continue

                        num_FS = self.cal_FS(current_bandwidth, path_len)
                        if num_FS == 0:
                            Input_feature += all_nega_ones
                            continue

                        slot_temp = [1] * self.SLOT_TOTAL
                        path_links = self.calclink(path)

                        if not path_links:
                            Input_feature += all_nega_ones
                            continue

                        # 使用线程安全的版本
                        slot_temp = self.get_new_slot_temp_safe(slot_temp, path_links, self.slot_map)

                        flag, slotflag, slotscontinue = self.mark_vector(slot_temp, 1)

                        if flag == 0 or not slotscontinue or max(slotscontinue) < num_FS:
                            Input_feature += all_nega_ones
                        else:
                            # 收集特征值用于批量更新
                            feature_updates['num_fs'].append(num_FS)

                            # 使用当前统计进行归一化
                            Input_feature.append(self.normalize_feature(num_FS, 'num_fs'))

                            slotscontinue_array = np.array(slotscontinue)
                            idx = np.where(slotscontinue_array >= num_FS)[0]

                            for jj in range(self.M):
                                if len(idx) > jj:
                                    pos = slotflag[idx[jj]]
                                    cont = slotscontinue[idx[jj]]

                                    feature_updates['slot_position'].append(pos)
                                    feature_updates['slot_continue'].append(cont)

                                    Input_feature.append(self.normalize_feature(pos, 'slot_position'))
                                    Input_feature.append(self.normalize_feature(cont, 'slot_continue'))
                                else:
                                    Input_feature += [-1, -1]

                            total_avail = sum(slotscontinue)
                            mean_avail = np.mean(slotscontinue) if slotscontinue else 0

                            feature_updates['total_available'].append(total_avail)
                            feature_updates['mean_size'].append(mean_avail)

                            Input_feature.append(self.normalize_feature(total_avail, 'total_available'))
                            Input_feature.append(self.normalize_feature(mean_avail, 'mean_size'))

                    Input_feature = np.reshape(np.array(Input_feature), (1, self.x_dim_p))
                    Input_feature_tensor = tf.convert_to_tensor(Input_feature, dtype=tf.float32)

                    # 获取策略和价值
                    prob_dist, value, entro = self.local_network.get_policy_value(
                        Input_feature_tensor, Input_feature_tensor, training=False)

                    # Epsilon-greedy动作选择
                    epsilon = self.get_epsilon()
                    if random.random() < epsilon:
                        action_id = np.random.choice(action_onehot)
                    else:
                        action_id = np.argmax(prob_dist[0].numpy())

                    # 执行动作
                    path_id = action_id // self.M
                    FS_id = action_id % self.M
                    path = self._get_path(current_src, current_dst, self.Candidate_Paths, path_id)

                    blocking = 0
                    if not path:
                        blocking = 1
                    else:
                        path_len = self.cal_len(path)
                        if path_len > 0:
                            num_FS = self.cal_FS(current_bandwidth, path_len)
                            if num_FS > 0:
                                slot_temp = [1] * self.SLOT_TOTAL
                                path_links = self.calclink(path)

                                if path_links:
                                    # 使用线程安全的版本
                                    slot_temp = self.get_new_slot_temp_safe(slot_temp, path_links, self.slot_map)

                                    flag, fs_start, fs_end = self.judge_availability(slot_temp, num_FS, FS_id)

                                    if flag == 1 and fs_start >= 0 and fs_end >= 0:
                                        self.slot_map, self.slot_map_t = self.update_slot_map_for_committing_wp(
                                            self.slot_map, path_links, fs_start, fs_end, self.slot_map_t, current_TTL)

                                        # 使用全局唯一的请求ID
                                        with self._episode_lock:
                                            request_id = f"{self.name}_{self.request_counter}"
                                            self.request_counter += 1

                                        self.request_set[request_id] = [list(path_links), fs_start, fs_end, current_TTL]
                                    else:
                                        blocking = 1
                                else:
                                    blocking = 1
                            else:
                                blocking = 1
                        else:
                            blocking = 1

                    # 计算奖励
                    r_t = 1 - 2 * blocking
                    num_blocks += blocking
                    episode_reward += r_t
                    total_steps += 1
                    episode_step_count += 1

                    episode_values.append(value[0, 0].numpy())

                    # 存储经验
                    self.episode_buffer.push([Input_feature[0], Input_feature[0], action_id, r_t, value[0, 0].numpy()])

                    # 批量训练
                    if len(self.episode_buffer) >= self.batch_size and episode_step_count % 10 == 0:
                        # 计算bootstrap值
                        bootstrap_value = 0.0

                        # 获取训练批次
                        train_data = self.episode_buffer.get_batch(self.batch_size)

                        if train_data:
                            # 如果还有后续经验，使用最后一个值作为bootstrap
                            if episode_step_count < self.episode_size - 1:
                                bootstrap_value = episode_values[-1] if episode_values else 0.0

                            v_l, p_l, e_l = self.train(train_data, global_network, bootstrap_value)
                            mean_value_loss += v_l
                            mean_policy_loss += p_l
                            mean_entropy += e_l

                        # 定期更新网络
                        if episode_step_count % 50 == 0:
                            self.update_target_graph(global_network, self.local_network)

                # Episode结束，批量更新特征统计
                for feature_name, values in feature_updates.items():
                    if feature_name in self.feature_stats and values:
                        for value in values:
                            self.feature_stats[feature_name].update(value)

                # 处理剩余数据
                if len(self.episode_buffer) > 0:
                    remaining_data = self.episode_buffer.get_batch(len(self.episode_buffer))
                    if remaining_data:
                        v_l, p_l, e_l = self.train(remaining_data, global_network, 0.0)
                        mean_value_loss += v_l
                        mean_policy_loss += p_l
                        mean_entropy += e_l
                    self.episode_buffer.clear()

                # 计算统计信息
                bp = num_blocks / self.episode_size
                self.episode_blocking.append(bp)
                self.episode_rewards.append(episode_reward)
                avg_value = np.mean(episode_values) if episode_values else 0.0
                self.episode_mean_values.append(avg_value)

                episode_time = time() - episode_start_time
                episode_times.append(episode_time)

                # 更新全局episode计数
                with self._episode_lock:
                    self.global_episodes.assign_add(1)
                    current_episode = self.global_episodes.numpy()

                # 获取当前学习率
                current_lr = self.trainer.learning_rate.numpy()

                # 衰减探索参数
                if current_episode % 100 == 0:
                    self.local_network.decay_exploration_params(decay_rate=0.995)

                # 更新监控器
                if self.monitor:
                    self.monitor.update(current_episode, bp, episode_reward, episode_time)

                # 早停检查
                if len(self.episode_blocking) > 100:
                    recent_bp = np.mean(self.episode_blocking[-100:])
                    recent_reward = np.mean(self.episode_rewards[-100:])

                    if self.monitor and self.monitor.should_early_stop(recent_bp, recent_reward):
                        print(f"🎯 性能目标达成！BP: {recent_bp:.4f}, Reward: {recent_reward:.2f}")
                        coord.request_stop()
                        break

                # Agent 0 负责日志记录
                if self.name == 'agent_0':
                    avg_episode_time = np.mean(episode_times)
                    print(
                        f"[Episode {current_episode}] BP: {bp:.4f} | "
                        f"Reward: {episode_reward:.2f} | Value: {avg_value:.4f} | "
                        f"V_Loss: {mean_value_loss:.4f} | P_Loss: {mean_policy_loss:.4f} | "
                        f"Entropy: {mean_entropy:.4f} | Epsilon: {epsilon:.4f} | "
                        f"LR: {current_lr:.6f} | Time: {episode_time:.2f}s"
                    )

                    # 日志记录
                    log_data = {
                        'episode': int(current_episode),
                        'blocking_probability': float(bp),
                        'reward': float(episode_reward),
                        'value': float(avg_value),
                        'value_loss': float(mean_value_loss),
                        'policy_loss': float(mean_policy_loss),
                        'entropy': float(mean_entropy),
                        'epsilon': float(epsilon),
                        'learning_rate': float(current_lr),
                        'episode_time': float(episode_time),
                        'temperature': float(self.local_network.temperature.numpy()),
                        'noise_scale': float(self.local_network.noise_scale.numpy())
                    }

                    with self._file_lock:
                        with safe_file_write(os.path.join(self.model_path, 'training_log.json'), 'a') as fp:
                            if fp:
                                json.dump(log_data, fp)
                                fp.write('\n')

                    # TensorBoard日志
                    with self.summary_writer.as_default():
                        tf.summary.scalar('blocking_probability', bp, step=current_episode)
                        tf.summary.scalar('episode_reward', episode_reward, step=current_episode)
                        tf.summary.scalar('average_value', avg_value, step=current_episode)
                        tf.summary.scalar('value_loss', mean_value_loss, step=current_episode)
                        tf.summary.scalar('policy_loss', mean_policy_loss, step=current_episode)
                        tf.summary.scalar('entropy', mean_entropy, step=current_episode)
                        tf.summary.scalar('epsilon', epsilon, step=current_episode)
                        tf.summary.scalar('learning_rate', current_lr, step=current_episode)
                        tf.summary.scalar('temperature', self.local_network.temperature.numpy(), step=current_episode)
                        tf.summary.scalar('noise_scale', self.local_network.noise_scale.numpy(), step=current_episode)
                    # 打印TensorBoard指标到控制台
                    if current_episode % 10 == 0:  # 每10个episode打印一次
                        print(f"\n{'='*60}")
                        print(f"📊 TensorBoard Metrics - Episode {current_episode}")
                        print(f"{'='*60}")
                        print(f"Blocking Probability: {bp:.4f}")
                        print(f"Episode Reward: {episode_reward:.2f}")
                        print(f"Average Value: {avg_value:.4f}")
                        print(f"Value Loss: {mean_value_loss:.4f}")
                        print(f"Policy Loss: {mean_policy_loss:.4f}")
                        print(f"Entropy: {mean_entropy:.4f}")
                        print(f"Epsilon: {epsilon:.4f}")
                        print(f"Learning Rate: {current_lr:.6f}")
                        print(f"Temperature: {self.local_network.temperature.numpy():.4f}")
                        print(f"Noise Scale: {self.local_network.noise_scale.numpy():.4f}")
                        print(f"{'='*60}\n")
                    # 定期保存模型
                    if current_episode % 100 == 0 and checkpoint_manager:
                        save_path = checkpoint_manager.save()
                        print(f"✅ 模型已保存 (Episode {current_episode}): {save_path}")

                        # 保存网络配置
                        config_path = os.path.join(self.model_path, f'network_config_ep{current_episode}.json')
                        with open(config_path, 'w') as f:
                            json.dump(self.local_network.get_config(), f, indent=2)

            except Exception as e:
                logging.error(f"{self.name} 训练循环中出错: {e}")
                import traceback
                traceback.print_exc()

                # 尝试恢复
                if coord and not coord.should_stop():
                    logging.info(f"{self.name} 尝试恢复训练...")
                    sleep(5)  # 短暂等待
                    continue
                else:
                    break

        # 清理资源
        try:
            self.summary_writer.close()
        except:
            pass