# GeAgent.py (已修改并整合)

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
import os  # 确保导入os模块


# Helper function for discounting rewards, similar to the original scipy.signal.lfilter usage
def discount_rewards(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class DeepRMSA_Agent():

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
                 # ==================== 修改开始 ====================
                 # 1. 在构造函数中接收 output_dir 参数
                 # 详细注释: 添加 `output_dir` 参数，允许从外部（GeA3C.py）传入统一的输出路径。
                 output_dir):  # 添加新参数
        # ==================== 修改结束 ====================
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

        # ==================== 修改开始 ====================
        # 2. 保存并使用 output_dir
        # 详细注释: 将传入的 `output_dir` 保存为实例属性，以供后续文件操作使用。
        self.output_dir = output_dir
        # ==================== 修改结束 ====================

        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_blocking = []
        self.episode_mean_values = []

        # ==================== 修改开始 ====================
        # 3. 修改 TensorBoard summary 的保存路径
        # 详细注释: 将 TensorBoard 的日志文件保存到指定的输出目录下的 'summaries' 子文件夹中。
        # 这样做可以更好地组织和分离不同类型的输出文件。
        summary_path = os.path.join(self.output_dir, 'summaries', "train_" + self.name)
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)  # 确保目录存在
        self.summary_writer = tf.summary.create_file_writer(summary_path)
        # ==================== 修改结束 ====================

        self.x_dim_p = x_dim_p
        self.x_dim_v = x_dim_v
        self.n_actions = n_actions

        # AC_Net is now a tf.keras.Model.
        self.local_network = AC_Net(scope=self.name,
                                    x_dim_p=self.x_dim_p,
                                    x_dim_v=self.x_dim_v,
                                    n_actions=self.n_actions,
                                    num_layers=num_layers,
                                    layer_size=layer_size,
                                    regu_scalar=regu_scalar)

        # 显式构建本地网络以确保其权重已初始化。
        # 这是在多线程环境中安全地复制权重的关键步骤。
        self.local_network((np.zeros((1, self.x_dim_p)), np.zeros((1, self.x_dim_v))))

        self.slot_map = [[1 for x in range(self.SLOT_TOTAL)] for y in
                         range(self.LINK_NUM)]
        self.slot_map_t = [[0 for x in range(self.SLOT_TOTAL)] for y in
                           range(self.LINK_NUM)]
        self.service_time = self.lambda_time[np.random.randint(0, self.len_lambda_time)]
        self.lambda_intervals = 1 / self.lambda_req
        self.request_set = {}
        self.his_slotmap = []

    # 此方法在Eager模式下运行，无需@tf.function。
    # 它在Python端处理Numpy数组，效率足够。
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

    def update_slot_map_for_committing_wp(self, slot_map, current_wp_link, current_fs, current_fe, slot_map_t,
                                          current_TTL):
        for ll in current_wp_link:
            for s in range(current_fs, current_fe + 1):
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

    # 这是核心训练逻辑，被重构以适应TF2.x。
    @tf.function
    def train(self, inputs_p, inputs_v, actions, discounted_rewards, advantages, global_network_ref):
        with tf.GradientTape() as tape:
            # 1. 使用本地网络进行前向传播
            policy_output, value_output = self.local_network((inputs_p, inputs_v))

            # 2. 计算价值损失 (Value Loss)
            # 价值目标是折扣后的奖励。
            target_v = tf.stop_gradient(discounted_rewards)
            value_output_squeezed = tf.squeeze(value_output)
            # 实例化 MeanSquaredError 类，然后传入参数进行调用
            loss_value_net = tf.keras.losses.MeanSquaredError()(target_v, value_output_squeezed)

            # 3. 计算策略损失 (Policy Loss)
            actions_onehot = tf.one_hot(actions, self.n_actions, dtype=tf.float32)
            responsible_outputs = tf.reduce_sum(policy_output * actions_onehot, axis=1)

            # 使用stop_gradient确保优势值不参与梯度计算
            log_policy = tf.math.log(responsible_outputs + 1e-6)
            lost_policy_net = -tf.reduce_mean(log_policy * tf.stop_gradient(advantages))

            # 4. 计算熵奖励（鼓励探索）
            entropy = -tf.reduce_mean(tf.reduce_sum(policy_output * tf.math.log(policy_output + 1e-6), axis=1))

            # 5. 计算总损失（包含正则化损失）
            # Keras模型会自动收集正则化损失。
            regu_loss = sum(self.local_network.losses)
            loss_policy = lost_policy_net - entropy * 0.01
            total_loss = loss_policy + 0.5 * loss_value_net + regu_loss

        # 6. 计算梯度
        local_vars = self.local_network.trainable_variables
        gradients = tape.gradient(total_loss, local_vars)
        grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

        # 7. 将梯度应用到全局网络
        global_vars = global_network_ref.trainable_variables
        self.trainer.apply_gradients(zip(grads, global_vars))

        return loss_value_net, loss_policy, entropy, grad_norms, tf.linalg.global_norm(global_vars)

    def rmsa(self, global_network, coord, saver):
        time_to = 0
        req_id = 0
        episode_count = self.global_episodes.numpy()
        total_steps = 0
        episode_buffer = []

        action_onehot = [x for x in range(self.n_actions)]
        sd_onehot = [x for x in range(self.num_src_dest_pair)]
        node_onehot = np.diag([1 for x in range(self.NODE_NUM)]).tolist()
        all_zeros = [0 for ii in range(3 + 2 * self.M)]
        all_nega_ones = [-1 for ii in range(3 + 2 * self.M)]

        # 在每个 episode 开始时，从全局网络同步权重到本地网络。
        self.update_local_network_from_global(global_network)

        epsilon = 1.0

        # ==================== 修改开始 ====================
        # 4. 优化智能体启动时的控制台输出
        # 详细注释: 打印更清晰的启动信息，告知用户哪个智能体已经开始工作。
        print(f"🚀 智能体 {self.name} 已启动，开始执行训练任务...")
        # ==================== 修改结束 ====================

        while not coord.should_stop():
            episode_values = []
            episode_reward = 0
            episode_step_count = 0
            actionss = []
            num_blocks = 0
            resource_util = []
            mean_value_loss, mean_policy_loss, mean_entropy = 0.0, 0.0, 0.0

            while episode_step_count < self.episode_size:
                req_id += 1
                (self.slot_map, self.request_set, self.slot_map_t) = self.release(self.slot_map, self.request_set,
                                                                                  self.slot_map_t, time_to)
                time_to = 0
                while time_to == 0:
                    time_to = np.random.exponential(self.lambda_intervals)

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
                            Input_feature.append((num_FS - 5.5) / 3.5)
                            slotscontinue_array = np.array(slotscontinue)
                            idx = np.where(slotscontinue_array >= num_FS)[0]
                            for jj in range(self.M):
                                if len(idx) > jj:
                                    Input_feature.append(
                                        2 * (slotflag[idx[jj]] - 0.5 * self.SLOT_TOTAL) / self.SLOT_TOTAL)
                                    Input_feature.append((slotscontinue[idx[jj]] - 8) / 8)
                                else:
                                    Input_feature += [-1, -1]
                            Input_feature.append(2 * (sum(slotscontinue) - 0.5 * self.SLOT_TOTAL) / self.SLOT_TOTAL)
                            Input_feature.append((np.mean(slotscontinue) - 4) / 4)

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

                # 直接调用本地网络实例获取策略和价值。
                prob_dist_tensor, value_tensor = self.local_network(
                    (Input_feature_reshaped_p, Input_feature_reshaped_v))

                pp = prob_dist_tensor.numpy()[0]
                value = value_tensor.numpy()[0, 0]

                if random.random() > epsilon:  # Epsilon-greedy, random() > epsilon 时使用模型预测
                    action_id = np.argmax(pp)
                else:
                    action_id = np.random.choice(action_onehot)

                path_id = action_id // self.M
                # 使用整数取模 '%' 替代 math.fmod
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
                        self.slot_map, self.slot_map_t = self.update_slot_map_for_committing_wp(self.slot_map,
                                                                                                path_links, fs_start,
                                                                                                fs_end, self.slot_map_t,
                                                                                                current_TTL)
                        temp_ = []
                        temp_.append(list(path_links))
                        temp_.append(fs_start)
                        # 修正拼写错误 fe_end -> fs_end (原代码存在此错误，此处保持修正)
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

                # 简化了训练触发逻辑。当缓冲区大小足够时进行训练。
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
                    epsilon = np.max([epsilon * 0.9995, 0.1])  # Epsilon decay

            # end of an episode
            self.global_episodes.assign_add(1)
            episode_count = self.global_episodes.numpy()

            if episode_count <= (3000 / self.episode_size):
                continue

            bp = num_blocks / self.episode_size
            if self.name == 'agent_0':
                print(f"\n[📊 Agent_0 | 回合 {episode_count}] 性能摘要:")
                print(f"  - 阻塞率 (BP)      : {bp:.6f}")
                print(f"  - 平均资源利用率(AU)   : {np.mean(resource_util):.6f}")
                print(f"  - 平均评估价值 (V) : {np.mean(episode_values) if episode_values else 0:.6f}")
                print(f"  - 平均每请求奖励 (R) : {episode_reward / self.episode_size:.6f}")

                with open(os.path.join(self.output_dir, 'BP.dat'), 'a') as fp:
                    fp.write('%f\n' % bp)
                with open(os.path.join(self.output_dir, 'AU.dat'), 'a') as fp:
                    fp.write('%f\n' % np.mean(resource_util))
                with open(os.path.join(self.output_dir, 'value.dat'), 'a') as fp:
                    fp.write('%f\n' % (np.mean(episode_values) if episode_values else 0))
                with open(os.path.join(self.output_dir, 'reward.dat'), 'a') as fp:
                    fp.write('%f\n' % (episode_reward / self.episode_size))

                with open(os.path.join(self.output_dir, 'value_loss.dat'), 'a') as fp:
                    fp.write('%f\n' % float(mean_value_loss))
                with open(os.path.join(self.output_dir, 'policy_loss.dat'), 'a') as fp:
                    fp.write('%f\n' % float(mean_policy_loss))
                with open(os.path.join(self.output_dir, 'entropy.dat'), 'a') as fp:
                    fp.write('%f\n' % float(mean_entropy))

                # ==================== 修改结束 ====================

            self.episode_blocking.append(bp)
            self.episode_rewards.append(episode_reward)
            self.episode_mean_values.append(np.mean(episode_values) if episode_values else 0)

            sample_step = int(1000 / self.episode_size)
            if episode_count % sample_step == 0 and episode_count != 0:
                if episode_count % (100 * sample_step) == 0 and self.name == 'agent_0':
                    if saver is not None:
                        save_path = saver.save()
                        # ==================== 修改开始 ====================
                        # 7. 优化模型保存时的控制台输出
                        # 详细注释: 提供更醒目的保存成功信息，并清晰地指出模型检查点保存的位置。
                        print(f"\n💾 ---- 模型检查点已保存 (回合 {episode_count}) ---- 💾")
                        print(f"   路径: {save_path}\n")
                        # ==================== 修改结束 ====================

                if self.name == 'agent_0':
                    mean_reward = np.mean(self.episode_rewards[-sample_step:])
                    mean_value = np.mean(self.episode_mean_values[-sample_step:])
                    mean_blocking = np.mean(self.episode_blocking[-sample_step:])

                    with self.summary_writer.as_default():
                        tf.summary.scalar('Perf/Reward', float(mean_reward), step=episode_count)
                        tf.summary.scalar('Perf/Value', float(mean_value), step=episode_count)
                        tf.summary.scalar('Perf/Blocking', float(mean_blocking), step=episode_count)
                        tf.summary.scalar('Losses/Value_Loss', float(mean_value_loss), step=episode_count)
                        tf.summary.scalar('Losses/Policy_Loss', float(mean_policy_loss), step=episode_count)
                        tf.summary.scalar('Losses/Entropy', float(mean_entropy), step=episode_count)
                        tf.summary.scalar('Losses/Grad_Norm', float(grad_norms), step=episode_count)
                        tf.summary.scalar('Losses/Var_Norm', float(var_norms), step=episode_count)
                    self.summary_writer.flush()