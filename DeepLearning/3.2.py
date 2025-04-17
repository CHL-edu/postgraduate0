import torch
import numpy as np
import matplotlib.pyplot as plt


def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    # 1. 生成特征数据
    X = torch.normal(0, 1, (num_examples, len(w)))
    print((num_examples, len(w)))
    # 2. 计算真实y值（无噪声）
    y_clean = X@w + b

    # 3. 添加噪声
    y_noisy = y_clean + torch.normal(0, 0.01, y_clean.shape)

    # 4. 可视化
    plt.figure(figsize=(15, 5))

    # 4.1 特征分布可视化
    plt.subplot(1, 3, 1)
    for i in range(len(w)):
        plt.hist(X[:, i].numpy(), bins=50, alpha=0.5, label=f'Feature {i + 1}')
    plt.title("Feature Distributions")
    plt.legend()

    # 4.2 单特征时的X-y关系
    if len(w) == 1:
        plt.subplot(1, 3, 2)
        plt.scatter(X.numpy(), y_noisy.numpy(), s=1, alpha=0.3)
        plt.plot(X.numpy(), y_clean.numpy(), 'b-', linewidth=2)
        plt.title("X vs y (Red line = true relationship)")
    else:
        # 4.3 多特征时展示每个特征与y的关系
        for i in range(len(w)):
            plt.subplot(1, 3, i + 2)
            plt.scatter(X[:, i].numpy(), y_noisy.numpy(), s=5, alpha=0.3)
            plt.title(f"Feature {i + 1} vs y")

    plt.tight_layout()
    plt.show()

    return X, y_noisy.reshape((-1, 1))


# 单特征示例
print("单特征示例:")
true_w = torch.tensor([2.0])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 双特征示例
print("\n双特征示例:")
true_w = torch.tensor([2.0,7.0])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)