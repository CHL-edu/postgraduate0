# GeACNet.py (修复后)

from __future__ import division
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)


# 继承自 tf.keras.Model，这是 TF2.x 中定义自定义模型的标准方式。
class AC_Net(tf.keras.Model):

    def __init__(self, scope, x_dim_p, x_dim_v, n_actions, num_layers, layer_size, regu_scalar):
        super(AC_Net, self).__init__(name=scope)  # 调用父类构造函数，并可以传递 name。
        self.scope = scope
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.x_dim_p = x_dim_p
        self.n_actions = n_actions
        self.x_dim_v = x_dim_v
        self.regu_scalar = regu_scalar

        # L2 正则化器实例。
        self.regularizer = tf.keras.regularizers.L2(self.regu_scalar)

        # --- Policy Network Layers ---
        # 将策略网络和价值网络的层定义为独立的 Keras 模型或层，结构更清晰。
        policy_layers = [tf.keras.layers.Dense(self.layer_size, activation='elu', kernel_regularizer=self.regularizer,
                                               name='policy_hidden_1')]
        for i in range(self.num_layers - 1):
            policy_layers.append(
                tf.keras.layers.Dense(self.layer_size, activation='elu', kernel_regularizer=self.regularizer,
                                      name=f'policy_hidden_{i + 2}'))
        self.policy_dnn_base = tf.keras.Sequential(policy_layers, name='policy_dnn_base')

        self.policy_output_layer = tf.keras.layers.Dense(
            self.n_actions, activation=tf.nn.softmax,
            kernel_initializer=self.normalized_columns_initializer(0.01),
            bias_initializer=None, name='policy_output')

        # --- Value Network Layers ---
        value_layers = [tf.keras.layers.Dense(self.layer_size, activation='elu', kernel_regularizer=self.regularizer,
                                              name='value_hidden_1')]
        for i in range(self.num_layers - 1):
            value_layers.append(
                tf.keras.layers.Dense(self.layer_size, activation='elu', kernel_regularizer=self.regularizer,
                                      name=f'value_hidden_{i + 2}'))
        self.value_dnn_base = tf.keras.Sequential(value_layers, name='value_dnn_base')

        self.value_output_layer = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=self.normalized_columns_initializer(1.0),
            bias_initializer=None, name='value_output')

    # 定义模型的前向传播逻辑。这是 TF2 Keras Model 的核心。
    @tf.function
    def call(self, inputs):
        inputs_policy, inputs_value = inputs
        # Policy Network
        policy_hidden = self.policy_dnn_base(inputs_policy)
        policy_output = self.policy_output_layer(policy_hidden)

        # Value Network
        value_hidden = self.value_dnn_base(inputs_value)
        value_output = self.value_output_layer(value_hidden)

        return policy_output, value_output

    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):  # 修正初始化器签名以匹配 Keras API
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)

        return _initializer

    # Agent 将直接调用模型实例 `local_net((input_p, input_v))` 来获取策略和价值。
    # Agent 将负责实现完整的训练逻辑（计算损失、梯度并应用到全局网络）。
    # 因此，此模型类中不再包含 train 或 get_policy_and_value 等方法。