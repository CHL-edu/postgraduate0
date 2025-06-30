from __future__ import division
import numpy as np
import tensorflow as tf
from collections import deque
import random
import string

np.random.seed(1)
tf.random.set_seed(1)


class AC_Net(tf.keras.Model):

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

        # Build policy network layers
        self.policy_layers = []
        for i in range(num_layers):
            self.policy_layers.append(
                tf.keras.layers.Dense(layer_size, activation='elu',
                                      kernel_regularizer=tf.keras.regularizers.l2(regu_scalar),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.3),
                                      bias_initializer=tf.keras.initializers.Constant(0.1),
                                      name=f'{scope}_policy_hidden_{i}')
            )

        # Policy output layer
        self.policy_output = tf.keras.layers.Dense(
            n_actions, activation='softmax',
            kernel_initializer=self.normalized_columns_initializer(0.01),
            use_bias=False,
            name=f'{scope}_policy_output'
        )

        # Build value network layers
        self.value_layers = []
        for i in range(num_layers):
            self.value_layers.append(
                tf.keras.layers.Dense(layer_size, activation='elu',
                                      kernel_regularizer=tf.keras.regularizers.l2(regu_scalar),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(0., 0.3),
                                      bias_initializer=tf.keras.initializers.Constant(0.1),
                                      name=f'{scope}_value_hidden_{i}')
            )

        # Value output layer
        self.value_output = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=self.normalized_columns_initializer(1.0),
            use_bias=False,
            name=f'{scope}_value_output'
        )

        # 初始化网络：执行一次前向传播以创建变量
        self._build_network()

    def _build_network(self):
        """通过执行一次前向传播来初始化所有层的变量"""
        # 创建虚拟输入
        dummy_input_p = tf.zeros((1, self.x_dim_p))
        dummy_input_v = tf.zeros((1, self.x_dim_v))

        # 执行前向传播以初始化变量
        _ = self.call_policy(dummy_input_p)
        _ = self.call_value(dummy_input_v)

    @tf.function
    def call_policy(self, inputs):
        x = inputs
        for layer in self.policy_layers:
            x = layer(x)
        return self.policy_output(x)

    @tf.function
    def call_value(self, inputs):
        x = inputs
        for layer in self.value_layers:
            x = layer(x)
        return self.value_output(x)

    def get_policy_value(self, input_p, input_v):
        policy = self.call_policy(input_p)
        value = self.call_value(input_v)
        entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-6))
        return policy, value, entropy

    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    def get_policy_loss(self, input_p, actions, advantages):
        policy = self.call_policy(input_p)
        actions_onehot = tf.one_hot(actions, self.n_actions, dtype=tf.float32)
        responsible_outputs = tf.reduce_sum(policy * actions_onehot, axis=1)

        entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-6))
        policy_loss = -tf.reduce_sum(tf.math.log(responsible_outputs + 1e-6) * advantages)

        # Weight of entropy: 0.01
        total_loss = policy_loss - entropy * 0.01

        return total_loss, policy_loss, entropy

    def get_value_loss(self, input_v, target_v):
        value = self.call_value(input_v)
        value_loss = tf.reduce_sum(tf.square(target_v - tf.reshape(value, [-1])))
        return value_loss

    def get_policy_variables(self):
        variables = []
        for layer in self.policy_layers:
            variables.extend(layer.trainable_variables)
        variables.extend(self.policy_output.trainable_variables)
        return variables

    def get_value_variables(self):
        variables = []
        for layer in self.value_layers:
            variables.extend(layer.trainable_variables)
        variables.extend(self.value_output.trainable_variables)
        return variables