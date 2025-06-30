# GeAgent.py (已应用所有逻辑与显示优化)

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
from GeACNet import AC_Net
import threading
import multiprocessing
import tensorflow as tf
import scipy.signal
from random import choice
from time import sleep
from time import time
import os


# Helper function for discounting rewards, similar to the original scipy.signal.lfilter usage
def discount_rewards(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class DeepRMSA_Agent():
    # 🟢 代码清晰度优化: 将魔法数字定义为具名常量
    # 这些值通常来自对数据集的统计分析（例如，均值和标准差）
    FS_MEAN = 5.5
    FS_STD = 3.5
    SLOT_CONTINUE_MEAN = 8.0
    SLOT_CONTINUE_STD = 8.0
    SLOT_CONTINUE_MEAN_GLOBAL = 4.0
    SLOT_CONTINUE_STD_GLOBAL = 4.0

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
                 # 1. 在构造函数中接收 output_dir 参数
                 output_dir):
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

        # 2. 保存并使用 output_dir
        self.output_dir = output_dir

        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_blocking = []
        self.episode_mean_values = []

        # 3. 修改 TensorBoard summary 的保存路径
        summary_path = os.path.join(self.output_dir, 'summaries', "train_" + self.name)
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(summary_path)

        self.x_dim_p = x_dim_p
        self.x_dim_v = x_dim_v
        self.n_actions = n_actions

        self.local_network = AC_Net(scope=self.name,
                                    x_dim_p=self.x_dim_p,
                                    x_dim_v=self.x_dim_v,
                                    n_actions=self.n_actions,
                                    num_layers=num_layers,
                                    layer_size=layer_size,
                                    regu_scalar=regu_scalar)

        self.local_network((np.zeros((1, self.x_dim_p)), np.zeros((1, self.x_dim_v))))

        self.slot_map = [[1 for x in range(self.SLOT_TOTAL)] for y in
                         range(self.LINK_NUM)]
        self.slot_map_t = [[0 for x in range(self.SLOT_TOTAL)] for y in
                           range(self.LINK_NUM)]
        self.service_time = self.lambda_time[np.random.randint(0, self.len_lambda_time)]
        self.lambda_intervals = 1 / self.lambda_req
        self.request_set = {}
        self.his_slotmap = []

    def update_local_network_from_global(self, global_network):
        global_weights = global_network.get_weights()
        self.local_network.set_weights(global_weights)

    def _get_path(self, src, dst, Candidate_Paths, k):
        if src == dst:
            print('error: _get_path()')
            path = []
        else:
            path = Candidate_Paths[src][dst][k]
            if path is None:
                return None
        return path

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
            except ValueError:
                break

            ii += a
            flag += 1
            slotflag.append(ii)

            try:
                n = tempvector[a + 1:].index(1 - default)
                slotcontinue = n + 1
                slotscontinue.append(slotcontinue)
                ii += slotcontinue
            except ValueError:
                slotscontinue.append(le - ii)
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

    # 🟢 命名优化: 将 current_fe 重命名为 current_fe_slot 以提升代码可读性
    def update_slot_map_for_committing_wp(self, slot_map, current_wp_link, current_fs, current_fe_slot, slot_map_t,
                                          current_TTL):
        for ll in current_wp_link:
            for s in range(current_fs, current_fe_slot + 1):
                assert slot_map[ll][s] == 1
                slot_map[ll][s] = 0
        return slot_map, slot_map_t

    def update_slot_map_for_releasing_wp(self, slot_map, current_wp_link, current_fs, current_fe):
        for ll in current_wp_link:
            for s in range(current_fs, current_fe + 1):
                assert slot_map[ll][s] == 0
                slot_map[ll][s] = 1
        return slot_map

    def release(self, slot_map, request_set, slot_map_t, time_to):
        if request_set:
            del_id = []
            for rr in request_set:
                request_set[rr][3] -= time_to
                if request_set[rr][3] <= 0:
                    current_wp_link = request_set[rr][0]
                    fs_wp = request_set[rr][1]
                    fe_wp = request_set[rr][2]
                    slot_map = self.update_slot_map_for_releasing_wp(slot_map, current_wp_link, fs_wp, fe_wp)
                    del_id.append(rr)
            for ii in del_id:
                del request_set[ii]
        return slot_map, request_set, slot_map_t

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
        return discount_rewards(x, self.gamma)

    @tf.function
    def train(self, inputs_p, inputs_v, actions, discounted_rewards, advantages, global_network_ref):
        with tf.GradientTape() as tape:
            policy_output, value_output = self.local_network((inputs_p, inputs_v))
            target_v = tf.stop_gradient(discounted_rewards)
            value_output_squeezed = tf.squeeze(value_output)
            loss_value_net = tf.keras.losses.MeanSquaredError()(target_v, value_output_squeezed)

            actions_onehot = tf.one_hot(actions, self.n_actions, dtype=tf.float32)
            responsible_outputs = tf.reduce_sum(policy_output * actions_onehot, axis=1)

            log_policy = tf.math.log(responsible_outputs + 1e-6)
            lost_policy_net = -tf.reduce_mean(log_policy * tf.stop_gradient(advantages))

            entropy = -tf.reduce_mean(tf.reduce_sum(policy_output * tf.math.log(policy_output + 1e-6), axis=1))

            regu_loss = sum(self.local_network.losses)
            loss_policy = lost_policy_net - entropy * 0.01
            total_loss = loss_policy + 0.5 * loss_value_net + regu_loss

        local_vars = self.local_network.trainable_variables
        gradients = tape.gradient(total_loss, local_vars)
        grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

        global_vars = global_network_ref.trainable_variables
        self.trainer.apply_gradients(zip(grads, global_vars))

        return loss_value_net, loss_policy, entropy, grad_norms, tf.linalg.global_norm(global_vars)

    def rmsa(self, global_network, coord, saver):
        req_id = 0
        episode_count = self.global_episodes.numpy()
        total_steps = 0
        episode_buffer = []

        action_onehot = [x for x in range(self.n_actions)]
        sd_onehot = [x for x in range(self.num_src_dest_pair)]
        node_onehot = np.diag([1 for x in range(self.NODE_NUM)]).tolist()
        all_zeros = [0 for ii in range(3 + 2 * self.M)]
        all_nega_ones = [-1 for ii in range(3 + 2 * self.M)]

        self.update_local_network_from_global(global_network)
        epsilon = 1.0

        print(f"🚀 智能体 {self.name} 已启动，开始执行训练任务...")

        while not coord.should_stop():
            episode_values = []
            episode_reward = 0
            episode_step_count = 0
            actionss = []
            num_blocks = 0
            resource_util = []
            mean_value_loss, mean_policy_loss, mean_entropy = 0.0, 0.0, 0.0

            while episode_step_count < self.episode_size:

                # 🔴 逻辑修正: 保证正确的事件处理顺序
                # 1. 首先，计算自上一事件以来的时间间隔
                time_to = 0
                while time_to == 0:
                    time_to = np.random.exponential(self.lambda_intervals)

                # 2. 其次，根据该时间间隔释放已到期的业务
                (self.slot_map, self.request_set, self.slot_map_t) = self.release(self.slot_map, self.request_set,
                                                                                  self.slot_map_t, time_to)

                # 3. 最后，处理新到达的请求
                req_id += 1

                if self.nonuniform is True:
                    sd_id = np.random.choice(sd_onehot, p=self.prob_arr)
                    temp = self.Src_Dest_Pair[sd_id]
                else:
                    temp = self.Src_Dest_Pair[np.random.randint(0, self.num_src_dest_pair)]

                current_src = temp[0]
                current_dst = temp[1]
                current_bandwidth = np.random.randint(25, 101)
                current_TTL = 0
                while current_TTL == 0 or current_TTL >= self.service_time * 2:
                    current_TTL = np.random.exponential(self.service_time)

                src_onehot_list = node_onehot[current_src - 1]
                dst_onehot_list = node_onehot[current_dst - 1]
                Input_feature = []
                Input_feature += src_onehot_list
                Input_feature += dst_onehot_list

                if self.model2_flag > 0:
                    slot_map_fur = []
                    slot_map_tmp = copy.deepcopy(self.slot_map)
                    request_set_tmp = copy.deepcopy(self.request_set)
                    slot_map_t_tmp = copy.deepcopy(self.slot_map_t)
                    for ii in range(self.model2_flag):
                        (slot_map_tmp, request_set_tmp, slot_map_t_tmp) = self.release(slot_map_tmp, request_set_tmp,
                                                                                       slot_map_t_tmp,
                                                                                       5 * self.lambda_intervals)
                        slot_map_fur.append(slot_map_tmp)

                for x in range(self.k_path):
                    path = self._get_path(current_src, current_dst, self.Candidate_Paths, x)
                    if len(path) == 0:
                        Input_feature += all_nega_ones
                        for ii in range(self.model2_flag):
                            Input_feature += [0, 0, 0]
                    else:
                        path_len = self.cal_len(path)
                        num_FS = self.cal_FS(current_bandwidth, path_len)
                        slot_temp = [1] * self.SLOT_TOTAL
                        path_links = self.calclink(path)
                        slot_temp = self.get_new_slot_temp(slot_temp, path_links, self.slot_map)
                        (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)
                        if flag == 0 or np.max(slotscontinue) < num_FS:
                            Input_feature += all_nega_ones
                        else:
                            # 🟢 使用具名常量进行特征缩放
                            Input_feature.append((num_FS - self.FS_MEAN) / self.FS_STD)
                            slotscontinue_array = np.array(slotscontinue)
                            idx = np.where(slotscontinue_array >= num_FS)[0]
                            for jj in range(self.M):
                                if len(idx) > jj:
                                    Input_feature.append(
                                        2 * (slotflag[idx[jj]] - 0.5 * self.SLOT_TOTAL) / self.SLOT_TOTAL)
                                    Input_feature.append(
                                        (slotscontinue[idx[jj]] - self.SLOT_CONTINUE_MEAN) / self.SLOT_CONTINUE_STD)
                                else:
                                    Input_feature += [-1, -1]
                            Input_feature.append(2 * (sum(slotscontinue) - 0.5 * self.SLOT_TOTAL) / self.SLOT_TOTAL)
                            Input_feature.append((np.mean(
                                slotscontinue) - self.SLOT_CONTINUE_MEAN_GLOBAL) / self.SLOT_CONTINUE_STD_GLOBAL)

                        for ii in range(self.model2_flag):
                            slot_temp = [1] * self.SLOT_TOTAL
                            slot_temp = self.get_new_slot_temp(slot_temp, path_links, slot_map_fur[ii])
                            (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)
                            if flag == 0:
                                Input_feature += [0, 0, 0]
                            else:
                                Input_feature.append(sum(slotscontinue) / self.SLOT_TOTAL)
                                Input_feature.append(np.mean(slotscontinue) / 8)
                                Input_feature.append(np.max(slotscontinue) / 8)

                Input_feature_np = np.array(Input_feature, dtype=np.float32)
                Input_feature_reshaped_p = np.reshape(Input_feature_np, (1, self.x_dim_p))
                Input_feature_reshaped_v = np.reshape(Input_feature_np, (1, self.x_dim_v))

                blocking = 0
                prob_dist_tensor, value_tensor = self.local_network(
                    (Input_feature_reshaped_p, Input_feature_reshaped_v))

                pp = prob_dist_tensor.numpy()[0]
                value = value_tensor.numpy()[0, 0]

                if random.random() > epsilon:
                    action_id = np.argmax(pp)
                else:
                    action_id = np.random.choice(action_onehot)

                path_id = action_id // self.M
                FS_id = action_id % self.M
                path = self._get_path(current_src, current_dst, self.Candidate_Paths, path_id)
                actionss.append(action_id)

                if len(path) == 0:
                    blocking = 1
                else:
                    path_len = self.cal_len(path)
                    num_FS = self.cal_FS(current_bandwidth, path_len)
                    slot_temp = [1] * self.SLOT_TOTAL
                    path_links = self.calclink(path)
                    slot_temp = self.get_new_slot_temp(slot_temp, path_links, self.slot_map)
                    (flag, fs_start, fs_end) = self.judge_availability(slot_temp, num_FS, FS_id)
                    if flag == 1:
                        # 🟢 调用时使用对应的参数，尽管调用侧代码不变，但函数定义已修改
                        self.slot_map, self.slot_map_t = self.update_slot_map_for_committing_wp(self.slot_map,
                                                                                                path_links, fs_start,
                                                                                                fs_end, self.slot_map_t,
                                                                                                current_TTL)
                        temp_ = []
                        temp_.append(list(path_links))
                        temp_.append(fs_start)
                        temp_.append(fs_end)
                        temp_.append(current_TTL)
                        self.request_set[req_id] = temp_
                    else:
                        blocking = 1

                r_t = 1 - 2 * blocking
                num_blocks += blocking
                episode_reward += r_t
                total_steps += 1
                episode_step_count += 1
                resource_util.append(1 - np.sum(self.slot_map) / (self.LINK_NUM * self.SLOT_TOTAL))

                if episode_count < (3000 / self.episode_size):
                    continue

                episode_buffer.append([Input_feature_reshaped_p, Input_feature_reshaped_v, action_id, r_t, value])
                episode_values.append(value)

                if len(episode_buffer) >= self.batch_size:
                    batch_data = np.array(episode_buffer, dtype=object)
                    inputs_p = np.vstack(batch_data[:, 0])
                    inputs_v = np.vstack(batch_data[:, 1])
                    actions = batch_data[:, 2].astype(np.int32)
                    rewards = batch_data[:, 3].astype(np.float32)
                    values = batch_data[:, 4].astype(np.float32)

                    discounted_rewards = self.discount(rewards)
                    advantages = discounted_rewards - values

                    mean_value_loss, mean_policy_loss, mean_entropy, grad_norms, var_norms = \
                        self.train(tf.constant(inputs_p, dtype=tf.float32),
                                   tf.constant(inputs_v, dtype=tf.float32),
                                   tf.constant(actions, dtype=tf.int32),
                                   tf.constant(discounted_rewards, dtype=tf.float32),
                                   tf.constant(advantages, dtype=tf.float32),
                                   global_network)

                    episode_buffer = []
                    self.update_local_network_from_global(global_network)
                    epsilon = np.max([epsilon * 0.9995, 0.1])

            # 回合结束
            self.global_episodes.assign_add(1)
            episode_count = self.global_episodes.numpy()

            if episode_count <= (3000 / self.episode_size):
                continue

            # 为性能记录做准备
            bp = num_blocks / self.episode_size
            self.episode_blocking.append(bp)
            self.episode_rewards.append(episode_reward)
            self.episode_mean_values.append(np.mean(episode_values) if episode_values else 0)

            sample_step = int(1000 / self.episode_size)
            if episode_count % sample_step == 0 and episode_count != 0:
                if episode_count % (100 * sample_step) == 0 and self.name == 'agent_0':
                    if saver is not None:
                        save_path = saver.save()
                        print(f"\n💾 ---- 模型检查点已保存 (回合 {episode_count}) ---- 💾")
                        print(f"   路径: {save_path}\n")

                if self.name == 'agent_0':
                    # 🟢 优化: 计算最近 sample_step 个回合的平均性能
                    mean_reward = np.mean(self.episode_rewards[-sample_step:])
                    mean_value = np.mean(self.episode_mean_values[-sample_step:])
                    mean_blocking = np.mean(self.episode_blocking[-sample_step:])

                    # 🟢 优化: 在此处集中打印性能摘要，而不是每个回合都打印
                    print(f"\n[📊 Agent_0 | 回合 {episode_count}] 最近 {sample_step} 回合性能摘要:")
                    print(f"  - 平均阻塞率 (BP)      : {mean_blocking:.6f}")
                    print(f"  - 平均评估价值 (V)     : {mean_value:.6f}")
                    print(f"  - 平均回合奖励 (R)     : {mean_reward / self.episode_size:.6f}")  # 平均每请求奖励
                    print(f"  - 最近批次价值损失      : {float(mean_value_loss):.6f}")
                    print(f"  - 最近批次策略损失      : {float(mean_policy_loss):.6f}")
                    print(f"  - 最近批次熵          : {float(mean_entropy):.6f}")

                    # 🟢 优化: 将性能数据文件保存到 metrics 子目录，并记录平滑后的值
                    metrics_path = os.path.join(self.output_dir, 'metrics')
                    os.makedirs(metrics_path, exist_ok=True)
                    with open(os.path.join(metrics_path, 'BP.dat'), 'a') as fp:
                        fp.write('%f\n' % mean_blocking)
                    with open(os.path.join(metrics_path, 'value.dat'), 'a') as fp:
                        fp.write('%f\n' % mean_value)
                    with open(os.path.join(metrics_path, 'reward.dat'), 'a') as fp:
                        fp.write('%f\n' % (mean_reward / self.episode_size))
                    # 注意: AU 和 Loss 值通常记录瞬时值或批次值
                    with open(os.path.join(metrics_path, 'AU.dat'), 'a') as fp:
                        fp.write('%f\n' % np.mean(resource_util))  # 记录当前回合的AU
                    with open(os.path.join(metrics_path, 'value_loss.dat'), 'a') as fp:
                        fp.write('%f\n' % float(mean_value_loss))
                    with open(os.path.join(metrics_path, 'policy_loss.dat'), 'a') as fp:
                        fp.write('%f\n' % float(mean_policy_loss))
                    with open(os.path.join(metrics_path, 'entropy.dat'), 'a') as fp:
                        fp.write('%f\n' % float(mean_entropy))

                    with self.summary_writer.as_default():
                        tf.summary.scalar('Perf/Reward_Mean', float(mean_reward / self.episode_size),
                                          step=episode_count)
                        tf.summary.scalar('Perf/Value_Mean', float(mean_value), step=episode_count)
                        tf.summary.scalar('Perf/Blocking_Mean', float(mean_blocking), step=episode_count)
                        tf.summary.scalar('Losses/Value_Loss', float(mean_value_loss), step=episode_count)
                        tf.summary.scalar('Losses/Policy_Loss', float(mean_policy_loss), step=episode_count)
                        tf.summary.scalar('Losses/Entropy', float(mean_entropy), step=episode_count)
                        tf.summary.scalar('Losses/Grad_Norm', float(grad_norms), step=episode_count)
                        tf.summary.scalar('Losses/Var_Norm', float(var_norms), step=episode_count)
                    self.summary_writer.flush()