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
from ACNettest import AC_Net
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


class DeepRMSA_Agent():
    _episode_lock = threading.Lock()
    _file_lock = threading.Lock()  # 添加文件写入锁

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
                 prob_arr):
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
        self.epsilon_decay_rate = 0.995  # 指数衰减率

        self.local_network = AC_Net(scope=self.name,
                                    trainer=self.trainer,
                                    x_dim_p=self.x_dim_p,
                                    x_dim_v=self.x_dim_v,
                                    n_actions=self.n_actions,
                                    num_layers=num_layers,
                                    layer_size=layer_size,
                                    regu_scalar=regu_scalar)

        self.slot_map = [[1 for x in range(self.SLOT_TOTAL)] for y in
                         range(self.LINK_NUM)]
        self.slot_map_t = [[0 for x in range(self.SLOT_TOTAL)] for y in
                           range(self.LINK_NUM)]
        self.service_time = self.lambda_time[np.random.randint(0, self.len_lambda_time)]
        self.lambda_intervals = 1 / self.lambda_req
        self.request_set = {}
        self.his_slotmap = []

        # 动态计算归一化参数（添加epsilon防止除零）
        self.norm_params = self._calculate_norm_params()
        self.norm_epsilon = 1e-8

        # 使用deque自动管理缓冲区大小
        self.MAX_BUFFER_SIZE = 2000
        self.episode_buffer = deque(maxlen=self.MAX_BUFFER_SIZE)

    def _calculate_norm_params(self):
        """动态计算归一化参数"""
        return {
            'num_fs': {'mean': 5.5, 'std': max(3.5, 1e-8)},
            'slot_position': {'mean': self.SLOT_TOTAL / 2, 'std': max(self.SLOT_TOTAL / 4, 1e-8)},
            'slot_continue': {'mean': self.SLOT_TOTAL / 10, 'std': max(self.SLOT_TOTAL / 10, 1e-8)},
            'total_available': {'mean': self.SLOT_TOTAL / 2, 'std': max(self.SLOT_TOTAL / 4, 1e-8)},
            'mean_size': {'mean': self.SLOT_TOTAL / 20, 'std': max(self.SLOT_TOTAL / 20, 1e-8)}
        }

    def update_target_graph(self, global_network, local_network):
        """从全局网络更新本地网络参数"""
        try:
            for global_var, local_var in zip(global_network.trainable_variables, local_network.trainable_variables):
                local_var.assign(global_var)
        except Exception as e:
            logging.error(f"更新目标网络时出错: {e}")

    def _get_path(self, src, dst, Candidate_Paths, k):
        """获取从源到目的地的第k条路径"""
        if src == dst:
            logging.debug(f'源节点和目的节点相同: {src}')
            return []

        if src not in Candidate_Paths or dst not in Candidate_Paths[src]:
            logging.debug(f'未找到从 {src} 到 {dst} 的路径')
            return []

        paths = Candidate_Paths[src][dst]
        if k >= len(paths) or paths[k] is None:
            return []

        return paths[k]

    def calclink(self, p):
        """计算路径上的链路"""
        path_link = []
        if len(p) < 2:
            return []

        for a, b in zip(p[:-1], p[1:]):
            if a in self.linkmap and b in self.linkmap[a]:
                k = self.linkmap[a][b][0]
                path_link.append(k)
            else:
                logging.warning(f"链路 {a}->{b} 不存在")
                return []
        return path_link

    def get_new_slot_temp(self, slot_temp, path_link, slot_map):
        """获取路径上的可用频谱槽"""
        for i in path_link:
            if i < len(slot_map):
                for j in range(min(len(slot_temp), len(slot_map[i]))):
                    slot_temp[j] = slot_map[i][j] & slot_temp[j]
        return slot_temp

    def mark_vector(self, vector, default):
        """标记连续的频谱块 - 修复版本"""
        le = len(vector)
        flag = 0
        slotscontinue = []
        slotflag = []

        i = 0
        while i < le:
            # 找到下一个可用槽的起始位置
            try:
                # 从当前位置开始查找
                idx = vector[i:].index(default)
                i += idx
                flag += 1
                slotflag.append(i)

                # 找到连续可用块的长度
                j = i + 1
                while j < le and vector[j] == default:
                    j += 1

                slot_len = j - i
                slotscontinue.append(slot_len)
                i = j

            except ValueError:
                # 没有更多可用块
                break

        return flag, slotflag, slotscontinue

    def judge_availability(self, slot_temp, current_slots, FS_id):
        """判断频谱块是否可用"""
        (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)
        fs = -1
        fe = -1

        if flag > 0 and slotscontinue:
            # 找到所有足够大的频谱块
            valid_blocks = [(i, slotflag[i], slotscontinue[i])
                            for i in range(len(slotscontinue))
                            if slotscontinue[i] >= current_slots]

            if valid_blocks and 0 <= FS_id < len(valid_blocks):
                _, fs, size = valid_blocks[FS_id]
                fe = fs + current_slots - 1
                # 确保fe不超出范围
                if fe < len(slot_temp):
                    return 1, fs, fe

        return 0, fs, fe

    def update_slot_map_for_committing_wp(self, slot_map, current_wp_link, current_fs, current_fe, slot_map_t,
                                          current_TTL):
        """提交工作路径时更新频谱图"""
        for ll in current_wp_link:
            if ll < len(slot_map):
                for s in range(current_fs, min(current_fe + 1, len(slot_map[ll]))):
                    slot_map[ll][s] = 0
                    if ll < len(slot_map_t) and s < len(slot_map_t[ll]):
                        slot_map_t[ll][s] = current_TTL
        return slot_map, slot_map_t

    def update_slot_map_for_releasing_wp(self, slot_map, current_wp_link, current_fs, current_fe):
        """释放工作路径时更新频谱图"""
        for ll in current_wp_link:
            if ll < len(slot_map):
                for s in range(current_fs, min(current_fe + 1, len(slot_map[ll]))):
                    slot_map[ll][s] = 1
        return slot_map

    def release(self, slot_map, request_set, slot_map_t, time_to):
        """释放过期的请求"""
        del_ids = []

        # 使用numpy数组加速
        slot_map_array = np.array(slot_map)
        slot_map_t_array = np.array(slot_map_t)

        # 更新所有请求的剩余时间
        for req_id, req_info in request_set.items():
            req_info[3] -= time_to
            if req_info[3] <= 0:
                links, fs, fe, _ = req_info
                del_ids.append(req_id)
                # 释放频谱
                for link in links:
                    if link < len(slot_map_array) and fe < len(slot_map_array[link]):
                        slot_map_array[link, fs:fe + 1] = 1
                        slot_map_t_array[link, fs:fe + 1] = 0

        # 删除过期请求
        for req_id in del_ids:
            del request_set[req_id]

        # 更新TTL
        slot_map_t_array = np.maximum(0, slot_map_t_array - time_to)

        # 清理过期但未释放的槽
        expired_mask = (slot_map_t_array == 0) & (slot_map_array == 0)
        slot_map_array[expired_mask] = 1

        return slot_map_array.tolist(), request_set, slot_map_t_array.tolist()

    def cal_len(self, path):
        """计算路径长度"""
        path_len = 0
        if len(path) < 2:
            return 0

        for a, b in zip(path[:-1], path[1:]):
            if a in self.linkmap and b in self.linkmap[a]:
                path_len += self.linkmap[a][b][1]
            else:
                logging.warning(f"链路 {a}->{b} 长度未定义")
        return path_len

    def cal_FS(self, bandwidth, path_len):
        """根据带宽和路径长度计算所需频谱槽数"""
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

    @tf.function(reduce_retracing=True)
    def train_step(self, local_network, input_p, input_v, actions, advantages, target_v):
        """单步训练"""
        with tf.GradientTape() as tape:
            # 在一个 tape 中计算所有损失
            policy_total_loss, policy_loss, entropy = local_network.get_policy_loss(input_p, actions, advantages)
            value_loss = local_network.get_value_loss(input_v, target_v)

            # 合并损失
            total_loss = policy_total_loss + value_loss * 0.5

        # 计算梯度
        all_local_vars = local_network.trainable_variables
        gradients = tape.gradient(total_loss, all_local_vars)

        return value_loss, policy_loss, entropy, gradients

    def train(self, episode_buffer_data, global_network, bootstrap_value):
        """训练网络"""
        try:
            # 转换为列表并验证数据
            episode_buff = []
            for item in episode_buffer_data:
                if len(item) == 5:  # 确保每个项目有5个元素
                    episode_buff.append(item)

            if len(episode_buff) == 0:
                return (0.0, 0.0, 0.0)

            # 提取数据
            input_p = np.vstack([item[0] for item in episode_buff])
            input_v = np.vstack([item[1] for item in episode_buff])
            actions = np.array([item[2] for item in episode_buff])
            rewards = np.array([item[3] for item in episode_buff])
            values = np.array([item[4] for item in episode_buff])

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

            # 获取全局网络的可训练变量
            all_global_vars = global_network.trainable_variables

            # 过滤有效梯度
            valid_grads_and_vars = [(grad, var) for grad, var in zip(combined_grads, all_global_vars) if
                                    grad is not None]

            if valid_grads_and_vars:
                valid_grads, valid_vars_tuple = zip(*valid_grads_and_vars)
                # 梯度裁剪
                clipped_grads, grad_norm = tf.clip_by_global_norm(valid_grads, 40.0)
                # 应用梯度
                self.trainer.apply_gradients(zip(clipped_grads, valid_vars_tuple))

            actual_batch_size = len(episode_buff)
            return (v_l.numpy() / actual_batch_size,
                    p_l.numpy() / actual_batch_size,
                    e_l.numpy() / actual_batch_size)

        except Exception as e:
            logging.error(f"训练过程中出错: {e}")
            return (0.0, 0.0, 0.0)

    def rmsa(self, coord, checkpoint_manager, global_network):
        """主训练循环"""
        total_steps = 0
        action_onehot = [x for x in range(self.n_actions)]
        sd_onehot = [x for x in range(self.num_src_dest_pair)]
        node_onehot = np.diag([1 for x in range(self.NODE_NUM)]).tolist()

        all_nega_ones = [-1 for _ in range(3 + 2 * self.M)]

        print(f'Starting {self.name}')

        # 性能监控
        episode_times = deque(maxlen=100)

        while not coord.should_stop():
            try:
                episode_start_time = time()

                # 更新本地网络
                self.update_target_graph(global_network, self.local_network)

                episode_count = self.global_episodes.numpy()
                mean_value_loss, mean_policy_loss, mean_entropy = 0.0, 0.0, 0.0

                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                num_blocks = 0

                # 清空缓冲区
                self.episode_buffer.clear()

                while episode_step_count < self.episode_size:
                    # 释放过期请求
                    time_to = np.random.exponential(self.lambda_intervals)
                    (self.slot_map, self.request_set, self.slot_map_t) = self.release(
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
                        num_FS = self.cal_FS(current_bandwidth, path_len)
                        slot_temp = [1] * self.SLOT_TOTAL
                        path_links = self.calclink(path)

                        if not path_links:
                            Input_feature += all_nega_ones
                            continue

                        slot_temp = self.get_new_slot_temp(slot_temp, path_links, self.slot_map)
                        (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)

                        if flag == 0 or not slotscontinue or np.max(slotscontinue) < num_FS:
                            Input_feature += all_nega_ones
                        else:
                            # 归一化特征（添加epsilon防止除零）
                            Input_feature.append(
                                (num_FS - self.norm_params['num_fs']['mean']) /
                                (self.norm_params['num_fs']['std'] + self.norm_epsilon))
                            slotscontinue_array = np.array(slotscontinue)
                            idx = np.where(slotscontinue_array >= num_FS)[0]

                            for jj in range(self.M):
                                if len(idx) > jj:
                                    pos = slotflag[idx[jj]]
                                    cont = slotscontinue[idx[jj]]
                                    Input_feature.append(
                                        (pos - self.norm_params['slot_position']['mean']) /
                                        (self.norm_params['slot_position']['std'] + self.norm_epsilon))
                                    Input_feature.append(
                                        (cont - self.norm_params['slot_continue']['mean']) /
                                        (self.norm_params['slot_continue']['std'] + self.norm_epsilon))
                                else:
                                    Input_feature += [-1, -1]

                            total_avail = sum(slotscontinue)
                            mean_avail = np.mean(slotscontinue) if slotscontinue else 0
                            Input_feature.append(
                                (total_avail - self.norm_params['total_available']['mean']) /
                                (self.norm_params['total_available']['std'] + self.norm_epsilon))
                            Input_feature.append(
                                (mean_avail - self.norm_params['mean_size']['mean']) /
                                (self.norm_params['mean_size']['std'] + self.norm_epsilon))

                    Input_feature = np.reshape(np.array(Input_feature), (1, self.x_dim_p))
                    Input_feature_tensor = tf.convert_to_tensor(Input_feature, dtype=tf.float32)

                    # 获取策略和价值
                    prob_dist, value, entro = self.local_network.get_policy_value(
                        Input_feature_tensor, Input_feature_tensor)

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
                        num_FS = self.cal_FS(current_bandwidth, path_len)
                        slot_temp = [1] * self.SLOT_TOTAL
                        path_links = self.calclink(path)

                        if path_links:
                            slot_temp = self.get_new_slot_temp(slot_temp, path_links, self.slot_map)
                            (flag, fs_start, fs_end) = self.judge_availability(slot_temp, num_FS, FS_id)

                            if flag == 1 and fs_start >= 0 and fs_end >= 0:
                                self.slot_map, self.slot_map_t = self.update_slot_map_for_committing_wp(
                                    self.slot_map, path_links, fs_start, fs_end, self.slot_map_t, current_TTL)
                                self.request_set[episode_step_count] = [list(path_links), fs_start, fs_end, current_TTL]
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
                    self.episode_buffer.append(
                        [Input_feature[0], Input_feature[0], action_id, r_t, value[0, 0].numpy()])

                    # 批量训练
                    if len(self.episode_buffer) >= self.batch_size:
                        train_data = list(self.episode_buffer)[:self.batch_size]

                        # 计算bootstrap值
                        if len(self.episode_buffer) > self.batch_size:
                            next_state = self.episode_buffer[self.batch_size][0]
                            next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
                            _, next_value, _ = self.local_network.get_policy_value(next_state_tensor, next_state_tensor)
                            bootstrap_value = next_value[0, 0].numpy()
                        else:
                            bootstrap_value = 0.0

                        v_l, p_l, e_l = self.train(train_data, global_network, bootstrap_value)

                        mean_value_loss += v_l
                        mean_policy_loss += p_l
                        mean_entropy += e_l

                        # 移除已训练的数据
                        for _ in range(min(self.batch_size, len(self.episode_buffer))):
                            self.episode_buffer.popleft()

                        # 更新本地网络
                        self.update_target_graph(global_network, self.local_network)

                # 处理剩余数据
                if len(self.episode_buffer) > 0:
                    v_l, p_l, e_l = self.train(list(self.episode_buffer), global_network, 0.0)
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

                # 更新全局episode计数（所有agent都更新）
                with self._episode_lock:
                    self.global_episodes.assign_add(1)
                    current_episode = self.global_episodes.numpy()

                # Agent 0 负责日志记录和模型保存
                if self.name == 'agent_0':
                    avg_episode_time = np.mean(episode_times)
                    print(
                        f"[Episode {current_episode}] BP: {bp:.4f} | "
                        f"Reward: {episode_reward:.2f} | Value: {avg_value:.4f} | "
                        f"V_Loss: {mean_value_loss:.4f} | P_Loss: {mean_policy_loss:.4f} | "
                        f"Entropy: {mean_entropy:.4f} | Epsilon: {epsilon:.4f} | "
                        f"Time: {episode_time:.2f}s (avg: {avg_episode_time:.2f}s)"
                    )

                    # 写入日志文件（使用文件锁）
                    log_data = {
                        'episode': int(current_episode),
                        'blocking_probability': float(bp),
                        'reward': float(episode_reward),
                        'value': float(avg_value),
                        'value_loss': float(mean_value_loss),
                        'policy_loss': float(mean_policy_loss),
                        'entropy': float(mean_entropy),
                        'epsilon': float(epsilon),
                        'episode_time': float(episode_time)
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
                        tf.summary.scalar('episode_time', episode_time, step=current_episode)

                    # 定期保存模型
                    if current_episode % 100 == 0:
                        save_path = checkpoint_manager.save()
                        print(f"Model saved at episode {current_episode}: {save_path}")

            except Exception as e:
                logging.error(f"{self.name} 训练循环中出错: {e}")
                import traceback
                traceback.print_exc()
                continue