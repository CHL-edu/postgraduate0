from __future__ import division
from collections import defaultdict
import struct
import types
import string
import numpy as np
import math
import copy
import random
import datetime
from AC_Net import AC_Net
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

# 配置基本日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@contextmanager
def safe_file_write(filepath, mode='a'):
    """安全的文件写入上下文管理器，处理潜在的IO异常。"""
    try:
        f = open(filepath, mode)
        yield f
    except IOError as e:
        logging.error(f"无法写入文件 {filepath}: {e}")
        yield None
    finally:
        if 'f' in locals() and f is not None:
            f.close()


class DeepRMSA_Agent():
    _episode_lock = threading.Lock()

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

        self.initial_epsilon = 1.0
        self.final_epsilon = 0.05
        self.epsilon_decay_steps = 200000

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

    def update_target_graph(self, global_network, local_network):
        for global_var, local_var in zip(global_network.variables, local_network.variables):
            local_var.assign(global_var)

    def _get_path(self, src, dst, Candidate_Paths, k):
        if src == dst:
            logging.warning(f'源节点和目的节点相同: {src}')
            return []

        if src not in Candidate_Paths or dst not in Candidate_Paths[src]:
            logging.warning(f'未找到从 {src} 到 {dst} 的路径')
            return []

        if not (0 <= k < len(Candidate_Paths[src][dst])):
            logging.warning(f'路径索引 {k} 超出范围 (0-{len(Candidate_Paths[src][dst]) - 1}) for {src}->{dst}')
            return []

        path = Candidate_Paths[src][dst][k]
        return path if path is not None else []

    def calclink(self, p):
        path_link = []
        for a, b in zip(p[:-1], p[1:]):
            k = self.linkmap[a][b][0]
            path_link.append(k)
        return path_link

    def get_new_slot_temp(self, slot_temp, path_link, slot_map):
        for i in path_link:
            for j in range(self.SLOT_TOTAL):
                slot_temp[j] = slot_map[i][j] & slot_temp[j]
        return slot_temp

    def mark_vector(self, vector, default):
        le = len(vector)
        flag = 0
        slotscontinue = []
        slotflag = []

        ii = 0
        while ii <= le - 1:
            tempvector = vector[ii:le]
            try:
                a = tempvector.index(default)
                ii += a
                flag += 1
                slotflag.append(ii)
                m = vector[ii + 1:le]
                try:
                    n = m.index(1 - default)
                    slotcontinue = n + 1
                    slotscontinue.append(slotcontinue)
                    ii += slotcontinue
                except ValueError:
                    slotscontinue.append(le - ii)
                    break
            except ValueError:
                break
        return flag, slotflag, slotscontinue

    def judge_availability(self, slot_temp, current_slots, FS_id):
        (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)
        fs = -1
        fe = -1
        if flag > 0:
            n = len(slotscontinue)
            flag_availability = 0
            t = 0
            for i in range(n):
                if slotscontinue[i] >= current_slots:
                    if t == FS_id:
                        fs = slotflag[i]
                        fe = slotflag[i] + current_slots - 1
                        flag_availability = 1
                        return flag_availability, fs, fe
                    t += 1
            return flag_availability, fs, fe
        else:
            flag_availability = 0
        return flag_availability, fs, fe

    def update_slot_map_for_committing_wp(self, slot_map, current_wp_link, current_fs, current_fe, slot_map_t,
                                          current_TTL):
        for ll in current_wp_link:
            for s in range(current_fs, current_fe + 1):
                slot_map[ll][s] = 0
                slot_map_t[ll][s] = current_TTL
        return slot_map, slot_map_t

    def update_slot_map_for_releasing_wp(self, slot_map, current_wp_link, current_fs, current_fe):
        for ll in current_wp_link:
            for s in range(current_fs, current_fe + 1):
                slot_map[ll][s] = 1
        return slot_map

    def release(self, slot_map, request_set, slot_map_t, time_to):
        del_ids = []
        for req_id, req_info in request_set.items():
            req_info[3] -= time_to
            if req_info[3] <= 0:
                links, fs, fe, _ = req_info
                del_ids.append(req_id)
                slot_map_array = np.array(slot_map)
                slot_map_t_array = np.array(slot_map_t)
                for link in links:
                    slot_map_array[link, fs:fe + 1] = 1
                    slot_map_t_array[link, fs:fe + 1] = 0
                slot_map = slot_map_array.tolist()
                slot_map_t = slot_map_t_array.tolist()

        for req_id in del_ids:
            del request_set[req_id]

        slot_map_t_array = np.array(slot_map_t)
        slot_map_t_array = np.maximum(0, slot_map_t_array - time_to)

        slot_map_array = np.array(slot_map)
        expired_mask = (slot_map_t_array == 0) & (slot_map_array == 0)
        slot_map_array[expired_mask] = 1

        return slot_map_array.tolist(), request_set, slot_map_t_array.tolist()

    def cal_len(self, path):
        path_len = 0
        for a, b in zip(path[:-1], path[1:]):
            path_len += self.linkmap[a][b][1]
        return path_len

    def cal_FS(self, bandwidth, path_len):
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
        return scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]

    # ========================================================================================
    # 修改点: 统一梯度计算
    # 原因: 为了解决 "Unknown variable" 错误，必须在单个 GradientTape 中计算所有损失，
    # 并为所有可训练变量一次性生成梯度列表。
    # ========================================================================================
    @tf.function(reduce_retracing=True)
    def train_step(self, local_network, input_p, input_v, actions, advantages, target_v):
        with tf.GradientTape() as tape:
            # 在一个 tape 中计算所有损失
            policy_total_loss, policy_loss, entropy = local_network.get_policy_loss(input_p, actions, advantages)
            value_loss = local_network.get_value_loss(input_v, target_v)

            # 将损失相加，以便计算合并的梯度
            # 注意：如果需要，可以为 value_loss 添加一个权重
            total_loss = policy_total_loss + value_loss * 0.5

            # 一次性获取所有局部变量和对应的梯度
        all_local_vars = local_network.trainable_variables
        gradients = tape.gradient(total_loss, all_local_vars)

        return value_loss, policy_loss, entropy, gradients

    def train(self, episode_buffer, global_network, bootstrap_value):
        episode_buff = np.array(episode_buffer, dtype=object)

        actual_batch_size = len(episode_buff)
        if actual_batch_size == 0:
            return (0.0, 0.0, 0.0)

        input_p = np.vstack(episode_buff[:, 0])
        input_v = np.vstack(episode_buff[:, 1])
        actions = episode_buff[:, 2]
        rewards = episode_buff[:, 3]
        values = episode_buff[:, 4]

        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = self.discount(rewards_plus)[:-1]

        value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = discounted_rewards - value_plus[:-1]

        input_p_tensor = tf.convert_to_tensor(input_p, dtype=tf.float32)
        input_v_tensor = tf.convert_to_tensor(input_v, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        target_v_tensor = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

        # 1. 从 train_step 获取损失和合并后的梯度
        v_l, p_l, e_l, combined_grads = self.train_step(
            self.local_network, input_p_tensor, input_v_tensor,
            actions_tensor, advantages_tensor, target_v_tensor
        )

        # ========================================================================================
        # 修改点: 单次应用所有梯度
        # 原因: 优化器期望在一次调用中接收所有它管理的变量的梯度。
        # 我们将合并后的梯度列表与全局网络的所有变量配对，然后进行一次应用。
        # ========================================================================================

        # 2. 获取全局网络的所有可训练变量
        all_global_vars = global_network.trainable_variables

        # 3. 过滤掉 None 梯度（如果有），然后裁剪并应用
        valid_grads_and_vars = [(grad, var) for grad, var in zip(combined_grads, all_global_vars) if grad is not None]

        if valid_grads_and_vars:
            # 只对有效的梯度进行操作
            valid_grads, valid_vars_tuple = zip(*valid_grads_and_vars)
            clipped_grads, _ = tf.clip_by_global_norm(valid_grads, 40.0)

            # 4. 单次调用 apply_gradients
            self.trainer.apply_gradients(zip(clipped_grads, valid_vars_tuple))

        return (v_l.numpy() / actual_batch_size,
                p_l.numpy() / actual_batch_size,
                e_l.numpy() / actual_batch_size)

    def rmsa(self, coord, checkpoint, global_network):
        MAX_BUFFER_SIZE = 5000

        total_steps = 0
        episode_buffer = []
        action_onehot = [x for x in range(self.n_actions)]
        sd_onehot = [x for x in range(self.num_src_dest_pair)]
        node_onehot = np.diag([1 for x in range(self.NODE_NUM)]).tolist()

        norm_params = {
            'num_fs': {'mean': 5.5, 'std': 3.5},
            'slot_position': {'mean': self.SLOT_TOTAL / 2, 'std': self.SLOT_TOTAL / 2},
            'slot_continue': {'mean': 8.0, 'std': 8.0},
            'total_available': {'mean': self.SLOT_TOTAL / 2, 'std': self.SLOT_TOTAL / 2},
            'mean_size': {'mean': 4.0, 'std': 4.0}
        }
        all_nega_ones = [-1 for _ in range(3 + 2 * self.M)]

        print('Starting ' + self.name)
        while not coord.should_stop():
            self.update_target_graph(global_network, self.local_network)

            episode_count = self.global_episodes.numpy()
            mean_value_loss, mean_policy_loss, mean_entropy = 0.0, 0.0, 0.0

            episode_values = []
            episode_reward = 0
            episode_step_count = 0

            num_blocks = 0

            episode_buffer = []

            while episode_step_count < self.episode_size:
                time_to = np.random.exponential(self.lambda_intervals)
                (self.slot_map, self.request_set, self.slot_map_t) = self.release(self.slot_map, self.request_set,
                                                                                  self.slot_map_t, time_to)

                if self.nonuniform:
                    sd_id = np.random.choice(sd_onehot, p=self.prob_arr)
                    temp = self.Src_Dest_Pair[sd_id]
                else:
                    temp = self.Src_Dest_Pair[np.random.randint(0, self.num_src_dest_pair)]

                current_src, current_dst = temp[0], temp[1]
                current_bandwidth = np.random.randint(25, 101)
                current_TTL = np.random.exponential(self.service_time)

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
                    slot_temp = self.get_new_slot_temp(slot_temp, path_links, self.slot_map)
                    (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)

                    if flag == 0 or not slotscontinue or np.max(slotscontinue) < num_FS:
                        Input_feature += all_nega_ones
                    else:
                        Input_feature.append((num_FS - norm_params['num_fs']['mean']) / norm_params['num_fs']['std'])
                        slotscontinue_array = np.array(slotscontinue)
                        idx = np.where(slotscontinue_array >= num_FS)[0]
                        for jj in range(self.M):
                            if len(idx) > jj:
                                pos = slotflag[idx[jj]]
                                cont = slotscontinue[idx[jj]]
                                Input_feature.append(
                                    (pos - norm_params['slot_position']['mean']) / norm_params['slot_position']['std'])
                                Input_feature.append(
                                    (cont - norm_params['slot_continue']['mean']) / norm_params['slot_continue']['std'])
                            else:
                                Input_feature += [-1, -1]

                        total_avail = sum(slotscontinue)
                        mean_avail = np.mean(slotscontinue) if slotscontinue else 0
                        Input_feature.append(
                            (total_avail - norm_params['total_available']['mean']) / norm_params['total_available'][
                                'std'])
                        Input_feature.append(
                            (mean_avail - norm_params['mean_size']['mean']) / norm_params['mean_size']['std'])

                Input_feature = np.reshape(np.array(Input_feature), (1, self.x_dim_p))
                Input_feature_tensor = tf.convert_to_tensor(Input_feature, dtype=tf.float32)

                prob_dist, value, entro = self.local_network.get_policy_value(Input_feature_tensor,
                                                                              Input_feature_tensor)

                epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * min(
                    self.global_episodes.numpy() * self.episode_size / self.epsilon_decay_steps, 1.0)

                if random.random() < epsilon:
                    action_id = np.random.choice(action_onehot)
                else:
                    action_id = np.argmax(prob_dist[0].numpy())

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
                    slot_temp = self.get_new_slot_temp(slot_temp, path_links, self.slot_map)
                    (flag, fs_start, fs_end) = self.judge_availability(slot_temp, num_FS, FS_id)

                    if flag == 1:
                        self.slot_map, self.slot_map_t = self.update_slot_map_for_committing_wp(self.slot_map,
                                                                                                path_links, fs_start,
                                                                                                fs_end, self.slot_map_t,
                                                                                                current_TTL)
                        self.request_set[episode_step_count] = [list(path_links), fs_start, fs_end, current_TTL]
                    else:
                        blocking = 1

                r_t = 1 - 2 * blocking
                num_blocks += blocking
                episode_reward += r_t
                total_steps += 1
                episode_step_count += 1

                episode_values.append(value[0, 0].numpy())

                episode_buffer.append([Input_feature[0], Input_feature[0], action_id, r_t, value[0, 0].numpy()])

                if len(episode_buffer) > MAX_BUFFER_SIZE:
                    episode_buffer = episode_buffer[-MAX_BUFFER_SIZE:]
                    logging.warning(f"缓冲区大小已修剪至 {MAX_BUFFER_SIZE}")

                if len(episode_buffer) >= self.batch_size:
                    train_data = episode_buffer[:self.batch_size]

                    if len(episode_buffer) > self.batch_size:
                        next_state = episode_buffer[self.batch_size][0]
                        next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
                        _, next_value, _ = self.local_network.get_policy_value(next_state_tensor, next_state_tensor)
                        bootstrap_value = next_value[0, 0].numpy()
                    else:
                        bootstrap_value = 0.0

                    v_l, p_l, e_l = self.train(
                        train_data, global_network, bootstrap_value)

                    mean_value_loss += v_l
                    mean_policy_loss += p_l
                    mean_entropy += e_l

                    del episode_buffer[:self.batch_size]
                    self.update_target_graph(global_network, self.local_network)

            if len(episode_buffer) > 0:
                v_l, p_l, e_l = self.train(episode_buffer, global_network, 0.0)
                mean_value_loss += v_l
                mean_policy_loss += p_l
                mean_entropy += e_l
                episode_buffer = []

            bp = num_blocks / self.episode_size
            self.episode_blocking.append(bp)
            self.episode_rewards.append(episode_reward)
            avg_value = np.mean(episode_values) if episode_values else 0.0
            self.episode_mean_values.append(avg_value)

            if self.name == 'agent_0':
                print(
                    f"[Agent_0 | 回合 {episode_count}] BP: {bp:.4f} | "
                    f"奖励: {episode_reward:.2f} | V值: {avg_value:.4f} | "
                    f"V损失: {mean_value_loss:.4f} | P损失: {mean_policy_loss:.4f} | "
                    f"熵: {mean_entropy:.4f} | Epsilon: {epsilon:.4f}"
                )

                with safe_file_write(os.path.join(self.model_path, 'BP.dat')) as fp:
                    if fp: fp.write(f'{bp}\n')
                with safe_file_write(os.path.join(self.model_path, 'value.dat')) as fp:
                    if fp: fp.write(f'{avg_value}\n')
                with safe_file_write(os.path.join(self.model_path, 'value_loss.dat')) as fp:
                    if fp: fp.write(f'{float(mean_value_loss)}\n')
                with safe_file_write(os.path.join(self.model_path, 'policy_loss.dat')) as fp:
                    if fp: fp.write(f'{float(mean_policy_loss)}\n')
                with safe_file_write(os.path.join(self.model_path, 'entropy.dat')) as fp:
                    if fp: fp.write(f'{float(mean_entropy)}\n')

            sample_step = int(1000 / self.episode_size)
            if episode_count > 0 and episode_count % sample_step == 0:
                if episode_count % (100 * sample_step) == 0 and self.name == 'agent_0':
                    checkpoint.save(file_prefix=os.path.join(self.model_path, 'model'))
                    logging.info(f"模型已于回合 {episode_count} 保存")

            with self._episode_lock:
                if self.name == 'agent_0':
                    self.global_episodes.assign_add(1)

