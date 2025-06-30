# DeepRMSA - 深度强化学习光网络资源分配系统

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目概述

DeepRMSA是一个基于深度强化学习的弹性光网络（EON）路由、调制和频谱分配（RMSA）系统。该系统使用异步优势Actor-Critic（A3C）算法结合近端策略优化（PPO），实现了高效的动态频谱资源管理。

### 主要特性

- 🚀 **并行训练**：支持多线程并行训练，充分利用多核CPU
- 🧠 **先进算法**：结合A3C和PPO，提供稳定高效的策略学习
- 🔒 **线程安全**：实现了线程安全的频谱图管理和资源分配
- 📊 **实时监控**：提供训练过程的实时监控和可视化
- 🎯 **自适应优化**：动态调整学习参数，自动特征归一化
- 💾 **内存管理**：智能内存管理，防止内存泄漏
- 🛡️ **容错设计**：具备错误恢复和优雅停止机制

## 🏗️ 系统架构

```
DeepRMSA/
├── 主控制器 (A3C627.py)
│   ├── 配置管理器
│   ├── 资源监控器
│   └── 训练协调器
├── 神经网络 (ACNet627.py)
│   ├── Actor网络 (策略)
│   ├── Critic网络 (价值)
│   └── PPO优化器
└── RMSA智能体 (Agent627.py)
    ├── 频谱管理器
    ├── 请求处理器
    └── 经验缓冲区
```

## 🔧 环境要求

### 系统要求
- **操作系统**: Linux (推荐Ubuntu 18.04+)
- **内存**: 至少16GB RAM (推荐32GB+)
- **CPU**: 多核处理器 (推荐8核+)
- **存储**: 至少10GB可用空间



## 📦 安装步骤

1. **克隆仓库**

2. **创建虚拟环境**

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **准备数据文件**


## 🚀 快速开始

### 基本使用

```bash
# 使用默认配置运行
python A3C627.py

# 使用自定义配置文件
python A3C627.py --config my_config.ini
```

### 配置文件示例

创建 `config.ini` 文件：

```ini
[network]
link_num = 44
node_num = 14
slot_total = 100

[path]
n_paths = 10
m_positions = 1
k_path = 5

[model]
num_layers = 5
layer_size = 128
regu_scalar = 1e-4

[training]
max_cpu = 4
lambda_req = 12
gamma = 0.95
episode_size = 1000
batch_size = 200
max_episodes = 50000
save_interval = 100
early_stop_bp_threshold = 0.05
early_stop_reward_threshold = 900
memory_limit_gb = 90

[paths]
output_dir = ./output/
candidate_paths_file = ./Src_Dst_Paths.dat
```

## 📊 训练监控

### 实时输出
训练过程中会显示：
```
[Episode 1000] BP: 0.1234 | Reward: 567.89 | Value: 12.34 | 
V_Loss: 0.0012 | P_Loss: 0.0034 | Entropy: 1.234 | 
Epsilon: 0.456 | LR: 0.0001 | Time: 2.34s
```

### TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir=output/model/

# 在浏览器中打开
http://localhost:6006
```

### 日志文件
- **训练日志**: `output/logs/training_*.log`
- **JSON日志**: `output/model/training_log.jsonl`
- **模型检查点**: `output/checkpoints/`

## 🎯 性能指标

系统优化目标：
- **阻塞率 (BP)**: < 0.05 (5%)
- **平均奖励**: > 900
- **收敛时间**: 通常在1,000-15,000回合
- **最终效果**: 通常在BP=0.1-，reward=800+早停
- ![training_log_PPO](https://github.com/user-attachments/assets/43920f10-60ee-4666-bc04-bdeff45ba14c)

## 🧪 算法详解

### Actor-Critic架构
- **Actor (策略网络)**: 输出动作概率分布
- **Critic (价值网络)**: 估计状态价值函数
- **共享特征提取**: 提高学习效率

### PPO优化
- **概率比率裁剪**: 防止策略更新过大
- **多轮更新**: 提高样本利用率
- **自适应裁剪范围**: 动态调整探索程度

### 特征工程
- **状态表示**: 节点编码 + 路径特征 + 频谱特征
- **自适应归一化**: 动态调整特征范围
- **增量学习**: 在线更新统计信息

## 🛠️ 高级配置

### 网络拓扑
系统默认使用NSFNet拓扑（14节点，44链路）。自定义拓扑需要修改：
1. `create_linkmap()` 函数
2. 候选路径文件

### 动作空间
- 动作数 = k_path × M
- k_path: 候选路径数
- M: 频谱位置选择数

### 探索策略
- **ε-贪婪**: 初始ε=1.0，指数衰减至0.05
- **温度参数**: 控制softmax随机性
- **噪声注入**: 训练时添加高斯噪声

## 🐛 故障排除

### 常见问题

1. **内存溢出**
   - 减小 `batch_size` 或 `episode_size`
   - 降低 `memory_limit_gb` 参数
   - 检查内存泄漏

2. **训练不收敛**
   - 调整学习率
   - 增加探索参数
   - 检查特征归一化

3. **路径文件错误**
   - 确保文件格式正确
   - 验证路径完整性
   - 检查文件权限

### 调试模式
```bash
# 启用详细日志
export TF_CPP_MIN_LOG_LEVEL=0
python A3C627.py --debug
```

## 🙏 致谢

- author Xiaoliang Chen, xlichen@ucdavis.edu
- copyright NGNS lab @ucdavis
