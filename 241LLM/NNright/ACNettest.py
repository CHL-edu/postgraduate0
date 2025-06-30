from __future__ import division
import numpy as np
import tensorflow as tf
from collections import deque
import random
import string

np.random.seed(1)
tf.random.set_seed(1)


class AC_Net(tf.keras.Model):
    """Actor-Critic Network for Deep RMSA"""

    def __init__(self, scope, trainer, x_dim_p, x_dim_v, n_actions, num_layers, layer_size, regu_scalar):
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
                    activation=None,  # 激活函数在BN后应用
                    kernel_regularizer=tf.keras.regularizers.l2(regu_scalar),
                    kernel_initializer=tf.keras.initializers.glorot_uniform(),
                    bias_initializer=tf.keras.initializers.zeros(),
                    name=f'{scope}_policy_hidden_{i}'
                )
            )
            self.policy_batch_norms.append(
                tf.keras.layers.BatchNormalization(name=f'{scope}_policy_bn_{i}')
            )

        # Policy output layer
        self.policy_output = tf.keras.layers.Dense(
            n_actions,
            activation=None,  # softmax在forward中应用
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
                    kernel_initializer=tf.keras.initializers.glorot_uniform(),
                    bias_initializer=tf.keras.initializers.zeros(),
                    name=f'{scope}_value_hidden_{i}'
                )
            )
            self.value_batch_norms.append(
                tf.keras.layers.BatchNormalization(name=f'{scope}_value_bn_{i}')
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

    @tf.function
    def call_policy(self, inputs, training=False):
        """前向传播策略网络"""
        x = inputs

        # 隐藏层
        for i in range(self.num_layers):
            x = self.policy_layers[i](x)
            x = self.policy_batch_norms[i](x, training=training)
            x = self.activation(x)
            if training:
                x = self.dropout(x, training=training)

        # 输出层
        logits = self.policy_output(x)

        # 使用温度参数来控制探索（防止除零）
        temperature = tf.maximum(self.temperature, 0.1)
        policy = tf.nn.softmax(logits / temperature)

        # 防止数值不稳定
        policy = tf.clip_by_value(policy, 1e-8, 1.0 - 1e-8)
        # 重新归一化确保和为1
        policy = policy / tf.reduce_sum(policy, axis=-1, keepdims=True)

        return policy

    @tf.function
    def call_value(self, inputs, training=False):
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

    def get_policy_value(self, input_p, input_v, training=False):
        """获取策略分布、价值和熵"""
        policy = self.call_policy(input_p, training=training)
        value = self.call_value(input_v, training=training)

        # 计算熵（用于鼓励探索）- 添加小epsilon防止log(0)
        entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-8), axis=1)
        entropy = tf.reduce_mean(entropy)

        return policy, value, entropy

    def normalized_columns_initializer(self, std=1.0):
        """归一化列初始化器"""

        def _initializer(shape, dtype=None):
            out = np.random.randn(*shape).astype(np.float32)
            # 添加epsilon防止除零
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True) + 1e-8)
            return tf.constant(out)

        return _initializer

    def get_policy_loss(self, input_p, actions, advantages, training=True):
        """计算策略损失"""
        # 获取策略分布
        policy = self.call_policy(input_p, training=training)

        # One-hot编码动作
        actions_onehot = tf.one_hot(actions, self.n_actions, dtype=tf.float32)

        # 计算选择的动作的概率
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, axis=1)

        # 防止log(0)
        responsible_outputs = tf.clip_by_value(responsible_outputs, 1e-8, 1.0)

        # 策略梯度损失
        policy_loss = -tf.reduce_mean(tf.math.log(responsible_outputs) * tf.stop_gradient(advantages))

        # 计算熵
        entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.math.log(policy + 1e-8), axis=1))

        # 总损失 = 策略损失 - 熵奖励
        entropy_weight = 0.01
        total_loss = policy_loss - entropy * entropy_weight

        # 添加正则化损失
        if self.losses:
            regularization_loss = tf.add_n(self.losses)
            total_loss += regularization_loss

        return total_loss, policy_loss, -entropy

    def get_value_loss(self, input_v, target_v, training=True):
        """计算价值损失"""
        # 获取价值预测
        value = self.call_value(input_v, training=training)
        value = tf.reshape(value, [-1])

        # 确保target_v也是一维的
        target_v = tf.reshape(target_v, [-1])

        # 均方误差损失
        value_loss = tf.reduce_mean(tf.square(target_v - value))

        # 添加正则化损失
        if self.losses:
            regularization_loss = tf.add_n(self.losses)
            value_loss += regularization_loss

        return value_loss

    def get_policy_variables(self):
        """获取策略网络的可训练变量"""
        variables = []
        for i in range(self.num_layers):
            variables.extend(self.policy_layers[i].trainable_variables)
            variables.extend(self.policy_batch_norms[i].trainable_variables)
        variables.extend(self.policy_output.trainable_variables)
        return variables

    def get_value_variables(self):
        """获取价值网络的可训练变量"""
        variables = []
        for i in range(self.num_layers):
            variables.extend(self.value_layers[i].trainable_variables)
            variables.extend(self.value_batch_norms[i].trainable_variables)
        variables.extend(self.value_output.trainable_variables)
        return variables

    def set_temperature(self, temperature):
        """设置温度参数"""
        self.temperature.assign(tf.maximum(temperature, 0.1))

    def save_weights(self, filepath):
        """保存模型权重"""
        try:
            super().save_weights(filepath)
            return True
        except Exception as e:
            print(f"保存权重时出错: {e}")
            return False

    def load_weights(self, filepath):
        """加载模型权重"""
        try:
            super().load_weights(filepath)
            return True
        except Exception as e:
            print(f"加载权重时出错: {e}")
            return False

    def summary(self):
        """打印模型摘要"""
        # 创建一个简单的输入来构建模型
        dummy_input_p = tf.keras.Input(shape=(self.x_dim_p,))
        dummy_input_v = tf.keras.Input(shape=(self.x_dim_v,))

        # 策略网络
        print("Policy Network:")
        x = dummy_input_p
        for i in range(self.num_layers):
            x = self.policy_layers[i](x)
            x = self.policy_batch_norms[i](x)
            x = self.activation(x)
        policy_out = self.policy_output(x)
        policy_model = tf.keras.Model(inputs=dummy_input_p, outputs=policy_out)
        policy_model.summary()

        # 价值网络
        print("\nValue Network:")
        x = dummy_input_v
        for i in range(self.num_layers):
            x = self.value_layers[i](x)
            x = self.value_batch_norms[i](x)
            x = self.activation(x)
        value_out = self.value_output(x)
        value_model = tf.keras.Model(inputs=dummy_input_v, outputs=value_out)
        value_model.summary()

    def get_config(self):
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
            'temperature': self.temperature.numpy()
        }
        return config