from __future__ import division
import numpy as np
import tensorflow as tf
from collections import deque
import random
import string
import os
from typing import Tuple, List, Optional, Dict
import logging

np.random.seed(1)
tf.random.set_seed(1)

logger = logging.getLogger(__name__)


class AC_Net(tf.keras.Model):
    """Actor-Critic Network for Deep RMSA - 改进版"""

    def __init__(self, scope: str, trainer, x_dim_p: int, x_dim_v: int,
                 n_actions: int, num_layers: int, layer_size: int,
                 regu_scalar: float):
        super(AC_Net, self).__init__()
        self.scope = scope
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.x_dim_p = x_dim_p
        self.n_actions = n_actions
        self.x_dim_v = x_dim_v
        self.regu_scalar = regu_scalar
        self.trainer = trainer

        # Build policy network layers with batch normalization
        self.policy_layers = []
        self.policy_batch_norms = []
        for i in range(num_layers):
            self.policy_layers.append(
                tf.keras.layers.Dense(
                    layer_size,
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.l2(regu_scalar),
                    kernel_initializer=tf.keras.initializers.he_normal(),
                    bias_initializer=tf.keras.initializers.zeros(),
                    name=f'{scope}_policy_hidden_{i}'
                )
            )
            self.policy_batch_norms.append(
                tf.keras.layers.BatchNormalization(
                    name=f'{scope}_policy_bn_{i}',
                    momentum=0.99,
                    epsilon=1e-5
                )
            )

        # Policy output layer
        self.policy_output = tf.keras.layers.Dense(
            n_actions,
            activation=None,
            kernel_initializer=self.normalized_columns_initializer(0.01),
            bias_initializer=tf.keras.initializers.zeros(),
            name=f'{scope}_policy_output'
        )

        # Build value network layers with batch normalization
        self.value_layers = []
        self.value_batch_norms = []
        for i in range(num_layers):
            self.value_layers.append(
                tf.keras.layers.Dense(
                    layer_size,
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.l2(regu_scalar),
                    kernel_initializer=tf.keras.initializers.he_normal(),
                    bias_initializer=tf.keras.initializers.zeros(),
                    name=f'{scope}_value_hidden_{i}'
                )
            )
            self.value_batch_norms.append(
                tf.keras.layers.BatchNormalization(
                    name=f'{scope}_value_bn_{i}',
                    momentum=0.99,
                    epsilon=1e-5
                )
            )

        # Value output layer
        self.value_output = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=self.normalized_columns_initializer(1.0),
            bias_initializer=tf.keras.initializers.zeros(),
            name=f'{scope}_value_output'
        )

        # Dropout层用于防止过拟合
        self.dropout_rate = 0.1
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # 激活函数
        self.activation = tf.keras.layers.ELU()

        # 温度参数（可调节的探索参数）
        self.temperature = tf.Variable(1.0, trainable=False, name=f'{scope}_temperature')

        # 噪声层用于探索（可选）
        self.noise_scale = tf.Variable(0.1, trainable=False, name=f'{scope}_noise_scale')

        # 🔧 改进：自适应梯度裁剪
        self.gradient_clip_norm = tf.Variable(5.0, trainable=False, name=f'{scope}_grad_clip')
        self.gradient_clip_min = 1.0
        self.gradient_clip_max = 10.0
        self.gradient_adaptation_rate = 0.01

        # 添加梯度监控
        self.gradient_history = deque(maxlen=100)
        self.gradient_warning_threshold = 10.0
        self.gradient_anomaly_count = 0

        # 🔧 改进：PPO参数优化
        self.ppo_clip_ratio = 0.2  # 标准PPO裁剪范围
        self.ppo_epochs = 4  # PPO多次更新epochs
        self.ppo_minibatch_size = 64  # PPO小批量大小

        # 🔧 新增：PPO更新缓冲区
        self.ppo_buffer = {
            'states_p': [],
            'states_v': [],
            'actions': [],
            'advantages': [],
            'returns': []
        }
        self.ppo_buffer_size = 2048  # 缓冲区大小
        self.ppo_update_interval = 500  # 保持原有命名，但会在收集足够数据后更新

        # 🔧 新增：数值稳定性参数
        self.logits_clip_min = -50.0
        self.logits_clip_max = 50.0
        self.logits_clip_decay = 0.999  # 逐渐放宽裁剪范围

        # 初始化网络：执行一次前向传播以创建变量
        self._build_network()

        # 添加旧策略网络（用于PPO）
        self.build_old_policy_network()

    def build_old_policy_network(self):
        """构建用于PPO的旧策略网络"""
        self.old_policy_layers = []
        self.old_policy_batch_norms = []

        for i in range(self.num_layers):
            self.old_policy_layers.append(
                tf.keras.layers.Dense(
                    self.layer_size,
                    activation=None,
                    trainable=False,  # 不可训练
                    name=f'{self.scope}_old_policy_hidden_{i}'
                )
            )
            self.old_policy_batch_norms.append(
                tf.keras.layers.BatchNormalization(
                    trainable=False,
                    name=f'{self.scope}_old_policy_bn_{i}'
                )
            )

        self.old_policy_output = tf.keras.layers.Dense(
            self.n_actions,
            activation=None,
            trainable=False,
            name=f'{self.scope}_old_policy_output'
        )

        # 初始化旧策略网络
        dummy_input = tf.zeros((1, self.x_dim_p))
        _ = self.call_old_policy(dummy_input, training=False)

    def update_old_policy(self):
        """更新旧策略网络权重"""
        for i in range(self.num_layers):
            self.old_policy_layers[i].set_weights(
                self.policy_layers[i].get_weights()
            )
            self.old_policy_batch_norms[i].set_weights(
                self.policy_batch_norms[i].get_weights()
            )
        self.old_policy_output.set_weights(
            self.policy_output.get_weights()
        )

    @tf.function
    def call_old_policy(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """前向传播旧策略网络"""
        x = inputs

        # 隐藏层
        for i in range(self.num_layers):
            x = self.old_policy_layers[i](x)
            x = self.old_policy_batch_norms[i](x, training=False)  # 始终使用推理模式
            x = self.activation(x)

        # 输出层
        logits = self.old_policy_output(x)

        # 使用稳定的softmax
        policy = self.stable_softmax(logits, self.temperature)

        return policy

    @tf.function
    def stable_softmax(self, logits: tf.Tensor, temperature: float = 1.0) -> tf.Tensor:
        """数值稳定的softmax实现 - 改进版"""
        # 确保温度不会太小
        temperature = tf.maximum(temperature, 0.01)

        # 🔧 改进：使用自适应的logits裁剪
        # 动态调整裁剪范围，避免过于激进
        logits_std = tf.math.reduce_std(logits)
        dynamic_clip_value = tf.minimum(
            self.logits_clip_max,
            tf.maximum(self.logits_clip_min, logits_std * 3.0)
        )

        # 软裁剪：使用tanh来平滑裁剪
        scale = dynamic_clip_value / 3.0
        logits = scale * tf.nn.tanh(logits / scale)

        # 缩放logits
        scaled_logits = logits / temperature

        # 🔧 改进：增强的数值稳定性
        # 使用双精度计算关键步骤
        scaled_logits_f64 = tf.cast(scaled_logits, tf.float64)
        max_logits = tf.reduce_max(scaled_logits_f64, axis=-1, keepdims=True)

        # 计算exp时使用更稳定的方法
        exp_logits = tf.exp(scaled_logits_f64 - max_logits)
        sum_exp = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)

        # 添加动态epsilon
        epsilon = tf.cast(1e-10 * tf.maximum(tf.constant(1.0, dtype=tf.float64), tf.reduce_max(tf.abs(sum_exp))), tf.float64)
        probs = exp_logits / (sum_exp + epsilon)

        # 转回float32
        probs = tf.cast(probs, tf.float32)

        # 确保概率和为1（使用更稳定的归一化）
        probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)

        # 最终裁剪
        probs = tf.clip_by_value(probs, 1e-10, 1.0 - 1e-10)

        return probs

    def _build_network(self):
        """通过执行一次前向传播来初始化所有层的变量"""
        # 创建虚拟输入
        dummy_input_p = tf.zeros((1, self.x_dim_p))
        dummy_input_v = tf.zeros((1, self.x_dim_v))

        # 执行前向传播以初始化变量
        _ = self.call_policy(dummy_input_p, training=False)
        _ = self.call_value(dummy_input_v, training=False)

        logger.debug(f"网络 {self.scope} 初始化完成")

    @tf.function
    def call_policy(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """前向传播策略网络"""
        x = inputs

        # 隐藏层
        for i in range(self.num_layers):
            x = self.policy_layers[i](x)
            x = self.policy_batch_norms[i](x, training=training)
            x = self.activation(x)
            if training:
                x = self.dropout(x, training=training)
                # 🔧 改进：使用更温和的噪声添加
                if self.noise_scale > 0:
                    # 使用渐进式噪声
                    noise = tf.random.normal(tf.shape(x), stddev=self.noise_scale)
                    noise_mask = tf.cast(tf.random.uniform(tf.shape(x)) < 0.1, tf.float32)
                    x = x + noise * noise_mask

        # 输出层
        logits = self.policy_output(x)

        # 使用稳定的softmax
        policy = self.stable_softmax(logits, self.temperature)

        return policy

    @tf.function
    def call_value(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """前向传播价值网络"""
        x = inputs

        # 隐藏层
        for i in range(self.num_layers):
            x = self.value_layers[i](x)
            x = self.value_batch_norms[i](x, training=training)
            x = self.activation(x)
            if training:
                x = self.dropout(x, training=training)

        # 输出层
        value = self.value_output(x)

        return value

    def get_policy_value(self, input_p: tf.Tensor, input_v: tf.Tensor,
                         training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """获取策略分布、价值和熵"""
        policy = self.call_policy(input_p, training=training)
        value = self.call_value(input_v, training=training)

        # 计算熵（用于鼓励探索）
        log_policy = tf.math.log(policy + 1e-10)
        entropy = -tf.reduce_sum(policy * log_policy, axis=1)
        entropy = tf.reduce_mean(entropy)

        return policy, value, entropy

    def normalized_columns_initializer(self, std: float = 1.0):
        """归一化列初始化器"""

        def _initializer(shape, dtype=None):
            out = np.random.randn(*shape).astype(np.float32)
            # 沿着第0维度归一化
            if len(shape) >= 2:
                out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True) + 1e-8)
            else:
                out *= std / np.sqrt(np.square(out).sum() + 1e-8)
            return tf.constant(out)

        return _initializer

    @tf.function
    def get_policy_loss(self, input_p: tf.Tensor, actions: tf.Tensor,
                        advantages: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """计算策略损失 - 改进的PPO实现"""
        # 获取当前策略分布
        policy = self.call_policy(input_p, training=training)

        # 获取旧策略分布
        old_policy = self.call_old_policy(input_p, training=False)

        # One-hot编码动作
        actions_onehot = tf.one_hot(actions, self.n_actions, dtype=tf.float32)

        # 计算选择的动作的概率
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, axis=1)
        old_responsible_outputs = tf.reduce_sum(old_policy * actions_onehot, axis=1)

        # 防止log(0)和除零
        epsilon = 1e-10
        responsible_outputs = tf.clip_by_value(responsible_outputs, epsilon, 1.0)
        old_responsible_outputs = tf.clip_by_value(old_responsible_outputs, epsilon, 1.0)

        # 🔧 改进：使用log空间计算比率，提高数值稳定性
        log_ratio = tf.math.log(responsible_outputs) - tf.math.log(old_responsible_outputs)
        ratio = tf.exp(tf.clip_by_value(log_ratio, -2.0, 2.0))  # 防止exp溢出

        # PPO裁剪
        clipped_ratio = tf.clip_by_value(
            ratio,
            1.0 - self.ppo_clip_ratio,
            1.0 + self.ppo_clip_ratio
        )

        # 🔧 改进：使用加权优势，减少极端值影响
        advantages_normalized = tf.stop_gradient(advantages)
        advantages_std = tf.math.reduce_std(advantages_normalized) + epsilon
        advantages_normalized = advantages_normalized / advantages_std

        # 软裁剪优势值
        advantages_normalized = tf.tanh(advantages_normalized / 2.0) * 2.0

        # 计算两种损失
        surrogate1 = ratio * advantages_normalized
        surrogate2 = clipped_ratio * advantages_normalized

        # 使用较小的损失（更保守）
        policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        # 计算熵 - 自适应熵权重
        log_policy = tf.math.log(policy + epsilon)
        entropy = -tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=1))

        # 🔧 改进：动态熵权重，随训练进度递减
        entropy_weight = 0.01 * tf.maximum(0.1, self.temperature)
        total_loss = policy_loss - entropy * entropy_weight

        # 添加PPO裁剪分数作为监控指标
        clip_fraction = tf.reduce_mean(tf.cast(tf.not_equal(ratio, clipped_ratio), tf.float32))

        return total_loss, policy_loss, entropy

    @tf.function
    def get_value_loss(self, input_v: tf.Tensor, target_v: tf.Tensor,
                       training: bool = True) -> tf.Tensor:
        """计算价值损失 - 使用改进的Huber损失"""
        # 获取价值预测
        value = self.call_value(input_v, training=training)
        value = tf.reshape(value, [-1])

        # 确保target_v也是一维的
        target_v = tf.reshape(target_v, [-1])

        # 🔧 改进：使用自适应的价值裁剪
        value_range = tf.reduce_max(tf.abs(target_v))
        clip_value = tf.maximum(100.0, value_range * 1.5)

        value_clipped = tf.clip_by_value(value, -clip_value, clip_value)
        target_v_clipped = tf.clip_by_value(target_v, -clip_value, clip_value)

        # 🔧 改进：自适应Huber损失
        # 根据误差分布动态调整delta
        error = target_v_clipped - value_clipped
        error_std = tf.math.reduce_std(error) + 1e-8
        delta = tf.clip_by_value(error_std, 0.5, 2.0)

        abs_error = tf.abs(error)

        # 分段损失函数
        is_small_error = abs_error <= delta
        small_error_loss = tf.square(error) / 2
        large_error_loss = delta * abs_error - tf.square(delta) / 2

        huber_loss = tf.where(is_small_error, small_error_loss, large_error_loss)
        value_loss = tf.reduce_mean(huber_loss)

        return value_loss

    @tf.function
    def compute_gradients(self, inputs_p: tf.Tensor, inputs_v: tf.Tensor,
                          actions: tf.Tensor, advantages: tf.Tensor,
                          target_v: tf.Tensor) -> Tuple[List[tf.Tensor], Dict[str, tf.Tensor]]:
        """计算梯度并进行裁剪 - 改进版"""
        with tf.GradientTape() as tape:
            policy_total_loss, policy_loss, entropy = self.get_policy_loss(
                inputs_p, actions, advantages, training=True
            )
            value_loss = self.get_value_loss(inputs_v, target_v, training=True)

            # 🔧 改进：自适应的价值损失权重
            value_loss_weight = tf.clip_by_value(0.5 / (1.0 + value_loss), 0.1, 0.5)
            total_loss = policy_total_loss + value_loss * value_loss_weight

            # 添加正则化损失
            if self.losses:
                regularization_loss = tf.add_n(self.losses)
                total_loss += regularization_loss

        # 计算梯度
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # 🔧 改进：更智能的梯度处理
        valid_gradients = []
        grad_norms = []

        for i, grad in enumerate(gradients):
            if grad is not None:
                # 计算每个梯度的范数
                grad_norm = tf.norm(grad)
                grad_norms.append(grad_norm)

                # 使用层级别的梯度裁剪
                if grad_norm > self.gradient_clip_norm * 2:
                    # 对异常大的梯度进行更严格的裁剪
                    grad = grad * (self.gradient_clip_norm / (grad_norm + 1e-8))

                valid_gradients.append(grad)
            else:
                # 使用零梯度替代None
                valid_gradients.append(tf.zeros_like(self.trainable_variables[i]))

        # 🔧 改进：使用自适应的全局梯度裁剪
        clipped_gradients, global_grad_norm = tf.clip_by_global_norm(
            valid_gradients, self.gradient_clip_norm
        )

        # 🔧 新增：自适应调整梯度裁剪范数
        if global_grad_norm > self.gradient_clip_norm * 1.5:
            # 如果梯度过大，略微增加裁剪范数
            new_clip_norm = tf.minimum(
                self.gradient_clip_max,
                self.gradient_clip_norm * (1 + self.gradient_adaptation_rate)
            )
            self.gradient_clip_norm.assign(new_clip_norm)
        elif global_grad_norm < self.gradient_clip_norm * 0.5:
            # 如果梯度较小，略微减小裁剪范数
            new_clip_norm = tf.maximum(
                self.gradient_clip_min,
                self.gradient_clip_norm * (1 - self.gradient_adaptation_rate)
            )
            self.gradient_clip_norm.assign(new_clip_norm)

        # 返回梯度和损失信息
        losses = {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'grad_norm': global_grad_norm,
            'grad_clip_norm': self.gradient_clip_norm
        }

        return clipped_gradients, losses

    def monitor_gradients(self, grad_norm: float) -> bool:
        """监控梯度健康状态 - 增强版"""
        self.gradient_history.append(grad_norm)

        # 计算梯度统计
        if len(self.gradient_history) > 10:
            recent_grads = list(self.gradient_history)[-10:]
            grad_mean = np.mean(recent_grads)
            grad_std = np.std(recent_grads)

            # 检测梯度异常模式
            if grad_norm > grad_mean + 3 * grad_std:
                self.gradient_anomaly_count += 1
                logger.warning(f"梯度异常: {grad_norm:.4f} (均值: {grad_mean:.4f}, 标准差: {grad_std:.4f})")

                # 连续异常检测
                if self.gradient_anomaly_count > 3:
                    logger.error("检测到连续梯度异常，建议检查以下内容：")
                    logger.error("1. 降低学习率")
                    logger.error("2. 检查输入数据是否有异常值")
                    logger.error("3. 增加批量大小")
                    logger.error(f"4. 当前梯度裁剪范数: {self.gradient_clip_norm.numpy():.4f}")
                    return False
            else:
                # 重置异常计数
                if self.gradient_anomaly_count > 0:
                    self.gradient_anomaly_count = max(0, self.gradient_anomaly_count - 1)

        return True

    def add_to_ppo_buffer(self, state_p: np.ndarray, state_v: np.ndarray,
                          action: int, advantage: float, return_: float):
        """添加数据到PPO缓冲区"""
        self.ppo_buffer['states_p'].append(state_p)
        self.ppo_buffer['states_v'].append(state_v)
        self.ppo_buffer['actions'].append(action)
        self.ppo_buffer['advantages'].append(advantage)
        self.ppo_buffer['returns'].append(return_)

        # 如果缓冲区满了，触发PPO更新
        if len(self.ppo_buffer['states_p']) >= self.ppo_buffer_size:
            return True
        return False

    def ppo_update(self, optimizer) -> Dict[str, float]:
        """执行PPO更新 - 完整实现"""
        if len(self.ppo_buffer['states_p']) == 0:
            return {}

        # 转换缓冲区数据为张量
        states_p = tf.convert_to_tensor(np.array(self.ppo_buffer['states_p']), dtype=tf.float32)
        states_v = tf.convert_to_tensor(np.array(self.ppo_buffer['states_v']), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(self.ppo_buffer['actions']), dtype=tf.int32)
        advantages = tf.convert_to_tensor(np.array(self.ppo_buffer['advantages']), dtype=tf.float32)
        returns = tf.convert_to_tensor(np.array(self.ppo_buffer['returns']), dtype=tf.float32)

        # 更新旧策略
        self.update_old_policy()

        # 收集所有epoch的损失
        all_losses = []

        # PPO多轮更新
        dataset_size = len(self.ppo_buffer['states_p'])
        for epoch in range(self.ppo_epochs):
            # 打乱数据
            indices = np.random.permutation(dataset_size)

            # 小批量更新
            for start in range(0, dataset_size, self.ppo_minibatch_size):
                end = min(start + self.ppo_minibatch_size, dataset_size)
                batch_indices = indices[start:end]

                # 获取小批量数据
                batch_states_p = tf.gather(states_p, batch_indices)
                batch_states_v = tf.gather(states_v, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)

                # 计算梯度并更新
                gradients, losses = self.compute_gradients(
                    batch_states_p, batch_states_v,
                    batch_actions, batch_advantages, batch_returns
                )

                # 应用梯度
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                # 监控梯度
                self.monitor_gradients(losses['grad_norm'].numpy())

                all_losses.append({k: v.numpy() for k, v in losses.items()})

        # 清空缓冲区
        for key in self.ppo_buffer:
            self.ppo_buffer[key].clear()

        # 计算平均损失
        avg_losses = {}
        if all_losses:
            for key in all_losses[0]:
                avg_losses[key] = np.mean([loss[key] for loss in all_losses])

        # 衰减探索参数
        self.decay_exploration_params()

        # 衰减logits裁剪范围
        self.logits_clip_min *= self.logits_clip_decay
        self.logits_clip_max *= self.logits_clip_decay

        return avg_losses

    def get_policy_variables(self) -> List[tf.Variable]:
        """获取策略网络的可训练变量"""
        variables = []
        for i in range(self.num_layers):
            variables.extend(self.policy_layers[i].trainable_variables)
            variables.extend(self.policy_batch_norms[i].trainable_variables)
        variables.extend(self.policy_output.trainable_variables)
        return variables

    def get_value_variables(self) -> List[tf.Variable]:
        """获取价值网络的可训练变量"""
        variables = []
        for i in range(self.num_layers):
            variables.extend(self.value_layers[i].trainable_variables)
            variables.extend(self.value_batch_norms[i].trainable_variables)
        variables.extend(self.value_output.trainable_variables)
        return variables

    def set_temperature(self, temperature: float):
        """设置温度参数"""
        self.temperature.assign(tf.maximum(temperature, 0.1))

    def set_noise_scale(self, noise_scale: float):
        """设置噪声缩放参数"""
        self.noise_scale.assign(tf.maximum(noise_scale, 0.0))

    def decay_exploration_params(self, decay_rate: float = 0.995):
        """衰减探索参数"""
        self.temperature.assign(self.temperature * decay_rate)
        self.noise_scale.assign(self.noise_scale * decay_rate)

    def save_weights(self, filepath: str) -> bool:
        """保存模型权重"""
        try:
            # 保存权重
            super().save_weights(filepath)

            # 保存额外的参数
            extra_params = {
                'temperature': float(self.temperature.numpy()),
                'noise_scale': float(self.noise_scale.numpy()),
                'dropout_rate': self.dropout_rate,
                'gradient_clip_norm': float(self.gradient_clip_norm.numpy()),
                'ppo_clip_ratio': self.ppo_clip_ratio,
                'gradient_history': list(self.gradient_history)[-20:],
                'gradient_anomaly_count': self.gradient_anomaly_count,
                'logits_clip_min': self.logits_clip_min,
                'logits_clip_max': self.logits_clip_max,
                'ppo_buffer_size': len(self.ppo_buffer['states_p'])
            }

            params_path = filepath + '_params.npy'
            np.save(params_path, extra_params)

            logger.info(f"模型权重已保存至: {filepath}")
            return True

        except Exception as e:
            logger.error(f"保存权重时出错: {e}")
            return False

    def load_weights(self, filepath: str) -> bool:
        """加载模型权重"""
        try:
            # 加载权重
            super().load_weights(filepath)

            # 加载额外的参数
            params_path = filepath + '_params.npy'
            if os.path.exists(params_path):
                extra_params = np.load(params_path, allow_pickle=True).item()
                self.temperature.assign(extra_params.get('temperature', 1.0))
                self.noise_scale.assign(extra_params.get('noise_scale', 0.1))
                if 'dropout_rate' in extra_params:
                    self.dropout_rate = extra_params['dropout_rate']
                if 'gradient_clip_norm' in extra_params:
                    self.gradient_clip_norm.assign(extra_params['gradient_clip_norm'])
                if 'ppo_clip_ratio' in extra_params:
                    self.ppo_clip_ratio = extra_params['ppo_clip_ratio']
                if 'gradient_history' in extra_params:
                    self.gradient_history = deque(extra_params['gradient_history'], maxlen=100)
                if 'logits_clip_min' in extra_params:
                    self.logits_clip_min = extra_params['logits_clip_min']
                if 'logits_clip_max' in extra_params:
                    self.logits_clip_max = extra_params['logits_clip_max']

            logger.info(f"模型权重已从 {filepath} 加载")
            return True

        except Exception as e:
            logger.error(f"加载权重时出错: {e}")
            return False

    def summary(self):
        """打印模型摘要"""
        print(f"\n{'=' * 50}")
        print(f"Model: {self.scope}")
        print(f"{'=' * 50}")

        # 策略网络
        print("\nPolicy Network:")
        print(f"  Input dimension: {self.x_dim_p}")
        print(f"  Hidden layers: {self.num_layers} x {self.layer_size}")
        print(f"  Output dimension: {self.n_actions}")

        # 价值网络
        print("\nValue Network:")
        print(f"  Input dimension: {self.x_dim_v}")
        print(f"  Hidden layers: {self.num_layers} x {self.layer_size}")
        print(f"  Output dimension: 1")

        # 参数统计
        total_params = sum([tf.size(v).numpy() for v in self.trainable_variables])
        policy_params = sum([tf.size(v).numpy() for v in self.get_policy_variables()])
        value_params = sum([tf.size(v).numpy() for v in self.get_value_variables()])

        print(f"\nParameter Count:")
        print(f"  Policy network: {policy_params:,}")
        print(f"  Value network: {value_params:,}")
        print(f"  Total: {total_params:,}")

        print(f"\nRegularization: L2 with scalar {self.regu_scalar}")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Temperature: {self.temperature.numpy():.3f}")
        print(f"Noise scale: {self.noise_scale.numpy():.3f}")
        print(f"Gradient clip norm: {self.gradient_clip_norm.numpy():.2f} (自适应)")
        print(f"PPO clip ratio: {self.ppo_clip_ratio}")
        print(f"PPO epochs: {self.ppo_epochs}")
        print(f"PPO minibatch size: {self.ppo_minibatch_size}")
        print(f"PPO buffer size: {self.ppo_buffer_size}")
        print(f"Logits clip range: [{self.logits_clip_min:.1f}, {self.logits_clip_max:.1f}]")
        print(f"{'=' * 50}\n")

    def get_config(self) -> Dict:
        """获取配置信息"""
        config = {
            'scope': self.scope,
            'num_layers': self.num_layers,
            'layer_size': self.layer_size,
            'x_dim_p': self.x_dim_p,
            'x_dim_v': self.x_dim_v,
            'n_actions': self.n_actions,
            'regu_scalar': self.regu_scalar,
            'dropout_rate': self.dropout_rate,
            'temperature': float(self.temperature.numpy()),
            'noise_scale': float(self.noise_scale.numpy()),
            'gradient_clip_norm': float(self.gradient_clip_norm.numpy()),
            'ppo_clip_ratio': self.ppo_clip_ratio,
            'ppo_epochs': self.ppo_epochs,
            'ppo_minibatch_size': self.ppo_minibatch_size,
            'ppo_buffer_size': self.ppo_buffer_size,
            'gradient_anomaly_count': self.gradient_anomaly_count,
            'recent_grad_norms': list(self.gradient_history)[-10:] if self.gradient_history else [],
            'logits_clip_min': self.logits_clip_min,
            'logits_clip_max': self.logits_clip_max
        }
        return config

    def update_from_config(self, config: Dict):
        """从配置更新参数"""
        if 'temperature' in config:
            self.set_temperature(config['temperature'])
        if 'noise_scale' in config:
            self.set_noise_scale(config['noise_scale'])
        if 'gradient_clip_norm' in config:
            self.gradient_clip_norm.assign(config['gradient_clip_norm'])
        if 'dropout_rate' in config:
            self.dropout_rate = config['dropout_rate']
        if 'ppo_clip_ratio' in config:
            self.ppo_clip_ratio = config['ppo_clip_ratio']
        if 'ppo_epochs' in config:
            self.ppo_epochs = config['ppo_epochs']
        if 'ppo_minibatch_size' in config:
            self.ppo_minibatch_size = config['ppo_minibatch_size']
        if 'ppo_buffer_size' in config:
            self.ppo_buffer_size = config['ppo_buffer_size']
        if 'logits_clip_min' in config:
            self.logits_clip_min = config['logits_clip_min']
        if 'logits_clip_max' in config:
            self.logits_clip_max = config['logits_clip_max']

    def reset_exploration(self):
        """重置探索参数到初始值"""
        self.temperature.assign(1.0)
        self.noise_scale.assign(0.1)
        self.logits_clip_min = -50.0
        self.logits_clip_max = 50.0
        logger.info(f"探索参数已重置")

    def get_exploration_stats(self) -> Dict[str, float]:
        """获取探索参数统计"""
        return {
            'temperature': float(self.temperature.numpy()),
            'noise_scale': float(self.noise_scale.numpy()),
            'gradient_clip_norm': float(self.gradient_clip_norm.numpy()),
            'gradient_anomaly_count': self.gradient_anomaly_count,
            'avg_gradient_norm': float(np.mean(self.gradient_history)) if self.gradient_history else 0.0,
            'max_gradient_norm': float(np.max(self.gradient_history)) if self.gradient_history else 0.0,
            'logits_clip_range': [self.logits_clip_min, self.logits_clip_max],
            'ppo_buffer_fill': len(
                self.ppo_buffer['states_p']) / self.ppo_buffer_size if self.ppo_buffer_size > 0 else 0.0
        }