import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
torch.float32) for x in [true_w, features, poly_features, labels]]


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            total_loss += l.sum().item()
            total_samples += l.numel()
    return total_loss / total_samples


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)),
                            batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)),
                           batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    # 设置绘图
    plt.figure(figsize=(10, 6))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.xlim([1, num_epochs])
    plt.ylim([1e-3, 1e2])

    train_losses = []
    test_losses = []
    epochs = []

    for epoch in range(num_epochs):
        # 训练一个epoch
        net.train()
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y.reshape(-1, 1))
            l.mean().backward()
            trainer.step()

        if epoch == 0 or (epoch + 1) % 20 == 0:
            net.eval()
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            epochs.append(epoch + 1)
            plt.plot(epochs, train_losses, 'b-', label='train')
            plt.plot(epochs, test_losses, 'r-', label='test')
            if epoch == 0:
                plt.legend()
            plt.draw()
            plt.pause(0.1)

    plt.show()
    print('weight:', net[0].weight.data.numpy())


# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])

# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])

# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)