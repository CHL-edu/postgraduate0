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

        # 梯度裁剪值
        self.gradient_clip_norm = 40.0

        # 初始化网络：执行一次前向传播以创建变量
        self._build_network()

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
                # 添加噪声以增加探索（可选）
                if self.noise_scale > 0:
                    noise = tf.random.normal(tf.shape(x), stddev=self.noise_scale)
                    x = x + noise

        # 输出层
        logits = self.policy_output(x)

        # 使用温度参数来控制探索
        temperature = tf.maximum(self.temperature, 0.1)
        scaled_logits = logits / temperature

        # 使用稳定的softmax计算
        max_logits = tf.reduce_max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = tf.exp(scaled_logits - max_logits)
        sum_exp_logits = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
        policy = exp_logits / (sum_exp_logits + 1e-8)

        # 防止数值不稳定
        policy = tf.clip_by_value(policy, 1e-8, 1.0 - 1e-8)

        # 重新归一化确保和为1
        policy = policy / (tf.reduce_sum(policy, axis=-1, keepdims=True) + 1e-8)

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
        log_policy = tf.math.log(policy + 1e-8)
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
        """计算策略损失 - 改进版本带PPO裁剪"""
        # 获取策略分布
        policy = self.call_policy(input_p, training=training)

        # 获取旧策略（用于PPO）
        old_policy = tf.stop_gradient(policy)

        # One-hot编码动作
        actions_onehot = tf.one_hot(actions, self.n_actions, dtype=tf.float32)

        # 计算选择的动作的概率
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, axis=1)
        old_responsible_outputs = tf.reduce_sum(old_policy * actions_onehot, axis=1)

        # 防止log(0)
        responsible_outputs = tf.clip_by_value(responsible_outputs, 1e-8, 1.0)
        old_responsible_outputs = tf.clip_by_value(old_responsible_outputs, 1e-8, 1.0)

        # PPO裁剪
        ratio = responsible_outputs / (old_responsible_outputs + 1e-8)
        clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2)

        # 计算两种损失
        surrogate1 = ratio * tf.stop_gradient(advantages)
        surrogate2 = clipped_ratio * tf.stop_gradient(advantages)

        # 使用较小的损失（更保守）
        policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        # 计算熵
        log_policy = tf.math.log(policy + 1e-8)
        entropy = -tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=1))

        # 总损失 = 策略损失 - 熵奖励
        entropy_weight = 0.01
        total_loss = policy_loss - entropy * entropy_weight

        return total_loss, policy_loss, -entropy

    @tf.function
    def get_value_loss(self, input_v: tf.Tensor, target_v: tf.Tensor,
                       training: bool = True) -> tf.Tensor:
        """计算价值损失 - 使用Huber损失"""
        # 获取价值预测
        value = self.call_value(input_v, training=training)
        value = tf.reshape(value, [-1])

        # 确保target_v也是一维的
        target_v = tf.reshape(target_v, [-1])

        # Huber损失（比MSE更稳定）
        delta = 1.0
        error = target_v - value
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
        """计算梯度并进行裁剪"""
        with tf.GradientTape() as tape:
            policy_total_loss, policy_loss, entropy = self.get_policy_loss(
                inputs_p, actions, advantages, training=True
            )
            value_loss = self.get_value_loss(inputs_v, target_v, training=True)

            # 合并损失（价值损失权重调整为0.5）
            total_loss = policy_total_loss + value_loss * 0.5

            # 添加正则化损失
            if self.losses:
                regularization_loss = tf.add_n(self.losses)
                total_loss += regularization_loss

        # 计算梯度
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # 过滤None梯度
        valid_gradients = []
        for grad in gradients:
            if grad is not None:
                valid_gradients.append(grad)
            else:
                # 使用零梯度替代None
                valid_gradients.append(tf.zeros_like(self.trainable_variables[gradients.index(grad)]))

        # 梯度裁剪
        clipped_gradients, grad_norm = tf.clip_by_global_norm(
            valid_gradients, self.gradient_clip_norm
        )

        # 返回梯度和损失信息
        losses = {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'grad_norm': grad_norm
        }

        return clipped_gradients, losses

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
                'gradient_clip_norm': self.gradient_clip_norm
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
                    self.gradient_clip_norm = extra_params['gradient_clip_norm']

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
        print(f"Gradient clip norm: {self.gradient_clip_norm}")
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
            'gradient_clip_norm': self.gradient_clip_norm
        }
        return config

    def update_from_config(self, config: Dict):
        """从配置更新参数"""
        if 'temperature' in config:
            self.set_temperature(config['temperature'])
        if 'noise_scale' in config:
            self.set_noise_scale(config['noise_scale'])
        if 'gradient_clip_norm' in config:
            self.gradient_clip_norm = config['gradient_clip_norm']
        if 'dropout_rate' in config:
            self.dropout_rate = config['dropout_rate']

    def reset_exploration(self):
        """重置探索参数到初始值"""
        self.temperature.assign(1.0)
        self.noise_scale.assign(0.1)
        logger.info(f"探索参数已重置: temperature=1.0, noise_scale=0.1")

    def get_exploration_stats(self) -> Dict[str, float]:
        """获取探索参数统计"""
        return {
            'temperature': float(self.temperature.numpy()),
            'noise_scale': float(self.noise_scale.numpy())
        }