from __future__ import division
# 自定义
from ACNet627 import AC_Net
from collections import deque, defaultdict
import numpy as np
import math
import random
import threading
import tensorflow as tf
import scipy.signal
from time import sleep, time
import os
import logging
from contextlib import contextmanager
import json
import bisect

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
        self.min_val = np.inf
        self.max_val = -np.inf

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

    def get_stats(self):
        if self.n < 2:
            return self.mean, np.ones_like(self.mean)
        else:
            variance = self.M2 / (self.n - 1)
            return self.mean, np.sqrt(variance + 1e-8)

    def normalize(self, value):
        """归一化值"""
        if self.n < 2:
            # 初始阶段使用min-max归一化到[0,1]
            if self.max_val > self.min_val:
                return (value - self.min_val) / (self.max_val - self.min_val + 1e-8)
            return 0.5
        else:
            mean, std = self.get_stats()
            return (value - mean) / (std + 1e-8)


class AdaptiveNormalizer:
    """自适应特征归一化器"""

    def __init__(self, warmup_samples=100):
        self.warmup_samples = warmup_samples
        self.samples = deque(maxlen=warmup_samples)
        self.running_stats = RunningStats()

    def update_and_normalize(self, value):
        """更新统计并归一化"""
        self.samples.append(value)
        self.running_stats.update(value)

        if len(self.samples) < self.warmup_samples:
            # 预热阶段：使用分位数归一化
            sorted_samples = sorted(self.samples)
            rank = bisect.bisect_left(sorted_samples, value)
            return rank / len(self.samples)
        else:
            # 正常阶段：使用运行统计
            mean, std = self.running_stats.get_stats()
            if std > 0:
                return np.clip((value - mean) / std, -3, 3) / 3
            else:
                return 0.0


class ExperienceBuffer:
    """高效的经验回放缓冲区"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, experience):
        with self._lock:
            self.buffer.append(experience)

    def get_batch(self, batch_size):
        """获取一批数据（不移除）"""
        with self._lock:
            buffer_list = list(self.buffer)
            if len(buffer_list) < batch_size:
                return buffer_list
            return random.sample(buffer_list, batch_size)

    def clear(self):
        with self._lock:
            self.buffer.clear()

    def __len__(self):
        with self._lock:
            return len(self.buffer)


class SlotMapManager:
    """线程安全的频谱图管理器"""

    def __init__(self, link_num, slot_total):
        self.link_num = link_num
        self.slot_total = slot_total
        self.slot_map = np.ones((link_num, slot_total), dtype=np.int8)
        self.slot_map_t = np.zeros((link_num, slot_total), dtype=np.float32)

        # 使用细粒度锁
        self.link_locks = [threading.RLock() for _ in range(link_num)]

    @contextmanager
    def lock_links(self, link_ids):
        """锁定多个链路"""
        # 按顺序锁定以避免死锁
        sorted_links = sorted(set(link_ids))
        locks = [self.link_locks[lid] for lid in sorted_links if 0 <= lid < self.link_num]

        for lock in locks:
            lock.acquire()
        try:
            yield
        finally:
            for lock in reversed(locks):
                lock.release()

    def update_slots(self, link_ids, start, end, value, ttl=None):
        """线程安全地更新频谱槽"""
        with self.lock_links(link_ids):
            for link_id in link_ids:
                if 0 <= link_id < self.link_num and end < self.slot_total:
                    self.slot_map[link_id, start:end + 1] = value
                    if ttl is not None:
                        self.slot_map_t[link_id, start:end + 1] = ttl

    def get_slot_temp(self, path_link):
        """线程安全地获取路径上的可用频谱槽"""
        with self.lock_links(path_link):
            slot_temp = np.ones(self.slot_total, dtype=np.int8)
            for link_id in path_link:
                if 0 <= link_id < self.link_num:
                    slot_temp &= self.slot_map[link_id]
            return slot_temp.tolist()

    def release_expired(self, time_to):
        """释放过期的频谱槽"""
        # 获取所有锁
        for lock in self.link_locks:
            lock.acquire()

        try:
            # 就地更新TTL
            occupied_mask = (self.slot_map == 0)
            self.slot_map_t[occupied_mask] = np.maximum(0, self.slot_map_t[occupied_mask] - time_to)

            # 清理过期槽
            expired_mask = (self.slot_map_t <= 0) & (self.slot_map == 0)
            self.slot_map[expired_mask] = 1
            self.slot_map_t[expired_mask] = 0

            return expired_mask

        finally:
            for lock in reversed(self.link_locks):
                lock.release()


class RequestManager:
    """改进的请求管理器，防止内存泄漏"""

    def __init__(self, max_requests=10000):
        self.requests = {}
        self.max_requests = max_requests
        self.lock = threading.Lock()
        self.request_counter = 0

    def add_request(self, path_links, fs_start, fs_end, ttl):
        """添加请求，自动清理旧请求"""
        with self.lock:
            # 如果请求数超过限制，清理最旧的请求
            if len(self.requests) >= self.max_requests:
                oldest_key = min(self.requests.keys())
                del self.requests[oldest_key]

            request_id = f"req_{self.request_counter}"
            self.request_counter += 1

            self.requests[request_id] = {
                'links': path_links,
                'fs_start': fs_start,
                'fs_end': fs_end,
                'ttl': ttl,
                'created_at': time()
            }
            return request_id

    def remove_expired(self, time_delta):
        """移除过期请求"""
        with self.lock:
            expired = []
            for req_id, req in self.requests.items():
                req['ttl'] -= time_delta
                if req['ttl'] <= 0:
                    expired.append((req_id, req))

            for req_id, _ in expired:
                del self.requests[req_id]

            return expired

    def get_all_requests(self):
        """获取所有请求的副本"""
        with self.lock:
            return dict(self.requests)


class DeepRMSA_Agent():
    _episode_lock = threading.Lock()
    _file_lock = threading.Lock()

    # 类级别的配置
    MAX_RETRIES = 5
    RETRY_DELAY = 2.0
    PPO_CLIP_RATIO = (0.8, 1.2)
    ENTROPY_WEIGHT_INITIAL = 0.01
    ENTROPY_WEIGHT_FINAL = 0.001
    ENTROPY_DECAY_EPISODES = 10000

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

        # 使用deque限制内存使用
        self.episode_rewards = deque(maxlen=1000)
        self.episode_blocking = deque(maxlen=1000)
        self.episode_mean_values = deque(maxlen=1000)

        # TensorBoard writer
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.model_path, "train_" + self.name)
        )

        self.x_dim_p = x_dim_p
        self.x_dim_v = x_dim_v
        self.n_actions = n_actions

        # 改进的epsilon衰减策略
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.05
        self.epsilon_decay_steps = 200000
        self.epsilon_decay_rate = 0.995

        # 创建本地网络
        self.local_network = AC_Net(
            scope=self.name,
            trainer=self.trainer,
            x_dim_p=self.x_dim_p,
            x_dim_v=self.x_dim_v,
            n_actions=self.n_actions,
            num_layers=num_layers,
            layer_size=layer_size,
            regu_scalar=regu_scalar
        )

        # 使用线程安全的频谱图管理器
        self.slot_manager = SlotMapManager(self.LINK_NUM, self.SLOT_TOTAL)

        # 使用改进的请求管理器
        self.request_manager = RequestManager(max_requests=10000)

        self.service_time = self.lambda_time[np.random.randint(0, self.len_lambda_time)]
        self.lambda_intervals = 1 / self.lambda_req

        # 使用自适应归一化器
        self.feature_normalizers = {
            'num_fs': AdaptiveNormalizer(),
            'slot_position': AdaptiveNormalizer(),
            'slot_continue': AdaptiveNormalizer(),
            'total_available': AdaptiveNormalizer(),
            'mean_size': AdaptiveNormalizer()
        }

        # 使用改进的经验缓冲区
        self.MAX_BUFFER_SIZE = 2000
        self.episode_buffer = ExperienceBuffer(self.MAX_BUFFER_SIZE)

        # JSON日志缓冲
        self.json_log_buffer = []
        self.json_log_buffer_size = 10

        # 优化：预处理路径数据
        self._preprocess_paths()

        # 错误重试计数器
        self.error_count = 0
        self.consecutive_errors = 0

        # PPO更新计数
        self.ppo_update_interval = 500  # 每500步更新一次旧策略

    def _preprocess_paths(self):
        """预处理路径数据以提高访问效率"""
        self.path_cache = {}
        for src in range(1, self.NODE_NUM + 1):
            for dst in range(1, self.NODE_NUM + 1):
                if src != dst:
                    for k in range(self.k_path):
                        path = self._get_path(src, dst, self.Candidate_Paths, k)
                        if path:
                            key = (src, dst, k)
                            self.path_cache[key] = {
                                'path': path,
                                'links': self.calclink(path),
                                'length': self.cal_len(path)
                            }

    def _get_path_cached(self, src, dst, k):
        """从缓存获取路径信息"""
        return self.path_cache.get((src, dst, k), None)

    def _get_path(self, src, dst, Candidate_Paths, k):
        """获取从源到目的地的第k条路径 - 安全版本"""
        if src == dst:
            return []

        try:
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
                    if np.all(np.array(slot_temp[fs:fe + 1]) == 1):
                        valid_blocks.append((i, fs, slotscontinue[i]))

        if not valid_blocks:
            return 0, -1, -1

        # 使用改进的选择策略（考虑频谱碎片）
        if FS_id >= len(valid_blocks):
            # 选择最佳适配而不是简单的模运算
            best_idx = min(range(len(valid_blocks)),
                           key=lambda i: valid_blocks[i][2] - current_slots)
            _, fs, _ = valid_blocks[best_idx]
        else:
            _, fs, _ = valid_blocks[FS_id]

        fe = fs + current_slots - 1

        return 1, fs, fe

    def release(self, time_to):
        """释放过期的请求 - 使用新的管理器"""
        # 释放频谱图中的过期槽
        expired_mask = self.slot_manager.release_expired(time_to)

        # 从请求管理器中移除过期请求
        expired_requests = self.request_manager.remove_expired(time_to)

        # 对于每个过期的请求，确保释放其占用的频谱
        for req_id, req in expired_requests:
            self.slot_manager.update_slots(
                req['links'],
                req['fs_start'],
                req['fs_end'],
                1,  # 释放频谱
                0  # 清零TTL
            )

        return len(expired_requests)

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
            return 1  # 返回最小值而不是0

        if path_len <= 0:
            logging.warning(f"无效的路径长度: {path_len}")
            return 1

        # 计算频谱槽数
        if path_len <= 625:
            num_FS = math.ceil(bandwidth / (4 * 12.5)) + 1
        elif path_len <= 1250:
            num_FS = math.ceil(bandwidth / (3 * 12.5)) + 1
        elif path_len <= 2500:
            num_FS = math.ceil(bandwidth / (2 * 12.5)) + 1
        else:
            num_FS = math.ceil(bandwidth / (1 * 12.5)) + 1

        return max(1, int(num_FS))  # 确保至少返回1

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

    def get_entropy_weight(self):
        """获取当前的熵权重（随训练进度衰减）"""
        episodes = self.global_episodes.numpy()
        if episodes >= self.ENTROPY_DECAY_EPISODES:
            return self.ENTROPY_WEIGHT_FINAL
        else:
            ratio = episodes / self.ENTROPY_DECAY_EPISODES
            return self.ENTROPY_WEIGHT_FINAL + (self.ENTROPY_WEIGHT_INITIAL - self.ENTROPY_WEIGHT_FINAL) * (1 - ratio)

    @tf.function(reduce_retracing=True)
    def train_step(self, local_network, input_p, input_v, actions, advantages, target_v):
        """单步训练"""
        with tf.GradientTape() as tape:
            policy_total_loss, policy_loss, entropy = local_network.get_policy_loss(
                input_p, actions, advantages
            )
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
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                if adv_std > 0:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

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

    def save_json_logs(self, force=False):
        """批量保存JSON日志"""
        if not self.json_log_buffer and not force:
            return

        if len(self.json_log_buffer) < self.json_log_buffer_size and not force:
            return

        try:
            log_file = os.path.join(self.model_path, 'training_log.jsonl')
            with self._file_lock:
                with safe_file_write(log_file, 'a') as fp:
                    if fp:
                        for log_data in self.json_log_buffer:
                            fp.write(json.dumps(log_data) + '\n')
                self.json_log_buffer.clear()
        except Exception as e:
            logging.error(f"保存JSON日志时出错: {e}")

    def rmsa(self, coord, checkpoint_manager, global_network):
        """主训练循环 - 使用改进的管理器"""
        total_steps = 0
        action_onehot = list(range(self.n_actions))
        sd_onehot = list(range(self.num_src_dest_pair))
        node_onehot = np.eye(self.NODE_NUM).tolist()
        all_nega_ones = [-1] * (3 + 2 * self.M)

        print(f'Starting {self.name}')
        episode_times = deque(maxlen=100)

        try:
            while not coord.should_stop():
                try:
                    # 在开始新episode前检查停止信号
                    if coord.should_stop():
                        break

                    episode_start_time = time()
                    self.update_target_graph(global_network, self.local_network)

                    episode_count = self.global_episodes.numpy()
                    mean_value_loss, mean_policy_loss, mean_entropy = 0.0, 0.0, 0.0

                    episode_values = []
                    episode_reward = 0
                    episode_step_count = 0
                    num_blocks = 0

                    self.episode_buffer.clear()

                    # 定期更新PPO旧策略
                    if total_steps % self.ppo_update_interval == 0:
                        self.local_network.update_old_policy()

                    while episode_step_count < self.episode_size:
                        # 检查停止信号
                        if coord.should_stop():
                            break

                        # 释放过期请求
                        time_to = np.random.exponential(self.lambda_intervals)
                        num_released = self.release(time_to)

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
                            # 使用缓存的路径信息
                            path_info = self._get_path_cached(current_src, current_dst, x)

                            if not path_info:
                                Input_feature += all_nega_ones
                                continue

                            path = path_info['path']
                            path_links = path_info['links']
                            path_len = path_info['length']

                            if path_len == 0:
                                Input_feature += all_nega_ones
                                continue

                            num_FS = self.cal_FS(current_bandwidth, path_len)
                            if num_FS == 0:
                                Input_feature += all_nega_ones
                                continue

                            # 使用线程安全的方法获取频谱状态
                            slot_temp = self.slot_manager.get_slot_temp(path_links)

                            flag, slotflag, slotscontinue = self.mark_vector(slot_temp, 1)

                            if flag == 0 or not slotscontinue or max(slotscontinue) < num_FS:
                                Input_feature += all_nega_ones
                            else:
                                # 使用自适应归一化器
                                Input_feature.append(
                                    self.feature_normalizers['num_fs'].update_and_normalize(num_FS)
                                )

                                slotscontinue_array = np.array(slotscontinue)
                                idx = np.where(slotscontinue_array >= num_FS)[0]

                                for jj in range(self.M):
                                    if len(idx) > jj:
                                        pos = slotflag[idx[jj]]
                                        cont = slotscontinue[idx[jj]]

                                        Input_feature.append(
                                            self.feature_normalizers['slot_position'].update_and_normalize(pos)
                                        )
                                        Input_feature.append(
                                            self.feature_normalizers['slot_continue'].update_and_normalize(cont)
                                        )
                                    else:
                                        Input_feature += [-1, -1]

                                total_avail = sum(slotscontinue)
                                mean_avail = np.mean(slotscontinue) if len(slotscontinue) > 0 else 0

                                Input_feature.append(
                                    self.feature_normalizers['total_available'].update_and_normalize(total_avail)
                                )
                                Input_feature.append(
                                    self.feature_normalizers['mean_size'].update_and_normalize(mean_avail)
                                )

                        Input_feature = np.reshape(np.array(Input_feature), (1, self.x_dim_p))
                        Input_feature_tensor = tf.convert_to_tensor(Input_feature, dtype=tf.float32)

                        # 获取策略和价值
                        prob_dist, value, entro = self.local_network.get_policy_value(
                            Input_feature_tensor, Input_feature_tensor, training=False
                        )

                        # Epsilon-greedy动作选择
                        epsilon = self.get_epsilon()
                        if random.random() < epsilon:
                            action_id = np.random.choice(action_onehot)
                        else:
                            action_id = np.argmax(prob_dist[0].numpy())

                        # 执行动作
                        path_id = action_id // self.M
                        FS_id = action_id % self.M

                        # 使用缓存
                        path_info = self._get_path_cached(current_src, current_dst, path_id)

                        blocking = 0
                        if not path_info:
                            blocking = 1
                        else:
                            path = path_info['path']
                            path_links = path_info['links']
                            path_len = path_info['length']

                            if path_len > 0:
                                num_FS = self.cal_FS(current_bandwidth, path_len)
                                if num_FS > 0:
                                    # 使用线程安全的方法
                                    slot_temp = self.slot_manager.get_slot_temp(path_links)

                                    if path_links:
                                        flag, fs_start, fs_end = self.judge_availability(slot_temp, num_FS, FS_id)

                                        if flag == 1 and fs_start >= 0 and fs_end >= 0:
                                            # 使用线程安全的方法更新频谱图
                                            self.slot_manager.update_slots(
                                                path_links, fs_start, fs_end, 0, current_TTL
                                            )

                                            # 使用请求管理器
                                            request_id = self.request_manager.add_request(
                                                path_links, fs_start, fs_end, current_TTL
                                            )
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
                        self.episode_buffer.push(
                            [Input_feature[0], Input_feature[0], action_id, r_t, value[0, 0].numpy()]
                        )

                        # 批量训练
                        if len(self.episode_buffer) >= self.batch_size and episode_step_count % 10 == 0:
                            # 在训练前检查停止信号
                            if coord.should_stop():
                                break

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

                    # 如果因为停止信号而退出内部循环，也退出外部循环
                    if coord.should_stop():
                        break

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
                    bp = num_blocks / self.episode_size if self.episode_size > 0 else 0
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

                    # 获取当前熵权重
                    current_entropy_weight = self.get_entropy_weight()

                    # 衰减探索参数
                    if current_episode % 100 == 0:
                        self.local_network.decay_exploration_params(decay_rate=0.995)

                    # 更新监控器
                    if self.monitor:
                        self.monitor.update(current_episode, bp, episode_reward, episode_time)

                    # 早停检查
                    if len(self.episode_blocking) > 100:
                        recent_bp = np.mean(list(self.episode_blocking)[-100:])
                        recent_reward = np.mean(list(self.episode_rewards)[-100:])

                        if self.monitor and self.monitor.should_early_stop(recent_bp, recent_reward):
                            print(f"🎯 性能目标达成！BP: {recent_bp:.4f}, Reward: {recent_reward:.2f}")
                            coord.request_stop()
                            continue

                    # Agent 0 负责日志记录和输出
                    if self.name == 'agent_0':
                        avg_episode_time = np.mean(list(episode_times))
                        print(
                            f"[Episode {current_episode}] BP: {bp:.4f} | "
                            f"Reward: {episode_reward:.2f} | Value: {avg_value:.4f} | "
                            f"V_Loss: {mean_value_loss:.4f} | P_Loss: {mean_policy_loss:.4f} | "
                            f"Entropy: {mean_entropy:.4f} | Epsilon: {epsilon:.4f} | "
                            f"LR: {current_lr:.6f} | Time: {episode_time:.2f}s"
                        )

                        # JSON日志记录（使用缓冲）
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
                            'entropy_weight': float(current_entropy_weight),
                            'episode_time': float(episode_time),
                            'temperature': float(self.local_network.temperature.numpy()),
                            'noise_scale': float(self.local_network.noise_scale.numpy()),
                            'timestamp': time()
                        }

                        self.json_log_buffer.append(log_data)
                        self.save_json_logs()

                        # TensorBoard日志（定期写入）
                        if current_episode % 10 == 0:
                            try:
                                with self.summary_writer.as_default():
                                    tf.summary.scalar('blocking_probability', bp, step=current_episode)
                                    tf.summary.scalar('episode_reward', episode_reward, step=current_episode)
                                    tf.summary.scalar('average_value', avg_value, step=current_episode)
                                    tf.summary.scalar('value_loss', mean_value_loss, step=current_episode)
                                    tf.summary.scalar('policy_loss', mean_policy_loss, step=current_episode)
                                    tf.summary.scalar('entropy', mean_entropy, step=current_episode)
                                    tf.summary.scalar('epsilon', epsilon, step=current_episode)
                                    tf.summary.scalar('learning_rate', current_lr, step=current_episode)
                                    tf.summary.scalar('entropy_weight', current_entropy_weight, step=current_episode)
                                    tf.summary.scalar('temperature', self.local_network.temperature.numpy(),
                                                      step=current_episode)
                                    tf.summary.scalar('noise_scale', self.local_network.noise_scale.numpy(),
                                                      step=current_episode)

                                    # 添加更多有用的指标
                                    if len(self.episode_blocking) >= 100:
                                        tf.summary.scalar('recent_avg_bp', recent_bp, step=current_episode)
                                        tf.summary.scalar('recent_avg_reward', recent_reward, step=current_episode)

                                self.summary_writer.flush()
                            except Exception as e:
                                logging.debug(f"写入TensorBoard时出错: {e}")

                        # 定期保存模型
                        if current_episode % 100 == 0 and checkpoint_manager:
                            if not coord.should_stop():
                                try:
                                    save_path = checkpoint_manager.save()
                                    print(f"✅ 模型已保存 (Episode {current_episode}): {save_path}")

                                    # 保存网络配置
                                    config_path = os.path.join(
                                        self.model_path,
                                        f'network_config_ep{current_episode}.json'
                                    )
                                    with open(config_path, 'w') as f:
                                        json.dump(self.local_network.get_config(), f, indent=2)
                                except Exception as e:
                                    logging.error(f"保存模型时出错: {e}")

                    # 成功完成episode，重置错误计数
                    self.consecutive_errors = 0

                except Exception as e:
                    self.error_count += 1
                    self.consecutive_errors += 1

                    logging.error(f"{self.name} 训练循环中出错 (错误 #{self.error_count}): {e}")
                    import traceback
                    traceback.print_exc()

                    # 如果遇到错误，检查是否应该停止
                    if coord and coord.should_stop():
                        break

                    # 检查连续错误次数
                    if self.consecutive_errors >= self.MAX_RETRIES:
                        logging.error(f"{self.name} 连续错误次数过多 ({self.consecutive_errors})，停止训练")
                        coord.request_stop()
                        break

                    # 尝试恢复
                    if coord and not coord.should_stop():
                        logging.info(f"{self.name} 尝试恢复训练...")
                        sleep(self.RETRY_DELAY)

                        # 重置状态
                        try:
                            self.slot_manager = SlotMapManager(self.LINK_NUM, self.SLOT_TOTAL)
                            self.request_manager = RequestManager(max_requests=10000)
                            self.episode_buffer.clear()
                        except Exception as reset_error:
                            logging.error(f"重置状态时出错: {reset_error}")

                        continue
                    else:
                        break

        finally:
            # 确保清理资源
            print(f"{self.name} 正在退出...")

            # 保存剩余的日志
            self.save_json_logs(force=True)

            # 清理资源
            try:
                if hasattr(self, 'summary_writer'):
                    self.summary_writer.close()
            except Exception as e:
                logging.debug(f"关闭summary writer时出错: {e}")

            # 清理缓冲区
            try:
                self.episode_buffer.clear()
            except Exception as e:
                logging.debug(f"清理缓冲区时出错: {e}")

            print(f"{self.name} 已安全退出")