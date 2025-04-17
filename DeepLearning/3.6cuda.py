import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import time
import numpy as np

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载FashionMNIST数据集
def load_data_fashion_mnist(batch_size, num_workers):
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers))


# 定义模型参数
def init_params():
    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True, device=device)
    b = torch.zeros(num_outputs, requires_grad=True, device=device)
    return W, b


# 定义softmax函数
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


# 定义网络
def net(X, W, b):
    X = X.to(device)
    return softmax(torch.matmul(X.reshape((-1, 784)), W) + b)


# 定义损失函数
def cross_entropy(y_hat, y):
    y = y.to(device)
    return -torch.log(y_hat[range(len(y_hat)), y])


# 定义准确率计算函数
def accuracy(y_hat, y):
    y = y.to(device)
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 定义累加器
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 定义评估函数
def evaluate_accuracy(net, data_iter, W, b):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X, W, b), y), y.numel())
    return metric[0] / metric[1]


# 定义训练周期
def train_epoch_ch3(net, train_iter, loss, updater, W, b):
    metric = Accumulator(3)
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X, W, b)
        l = loss(y_hat, y)
        l.sum().backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


# 定义动画绘制类
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: self._set_axes(
            xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def _set_axes(self, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        ax = self.axes[0]
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if legend: ax.legend(legend)
        ax.grid()

    def add(self, x, y):
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw()
        plt.pause(0.001)


# 定义训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, W, b):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater, W, b)
        test_acc = evaluate_accuracy(net, test_iter, W, b)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")


# 定义优化器
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param.sub_(lr * param.grad / batch_size)
                param.grad.zero_()


# 定义预测函数
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def predict_ch3(net, test_iter, W, b, n=6):
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        break
    trues = get_fashion_mnist_labels(y.cpu())
    preds = get_fashion_mnist_labels(net(X, W, b).argmax(axis=1).cpu())
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].cpu().reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    # 计算耗时
    elapsed_time = time.time() - start_time
    print(f"完整耗时: {elapsed_time:.4f} 秒")
    plt.show()


# 主函数
if __name__ == '__main__':
    # 测量时间
    start_time = time.time()

    print(f"使用设备: {device}")

    # 数据加载
    batch_size = 4096
    train_iter, test_iter = load_data_fashion_mnist(batch_size, num_workers=4)  # Windows下设为0

    # 初始化参数
    W, b = init_params()

    # 设置超参数
    lr = 0.1


    def updater(batch_size):
        return sgd([W, b], lr, batch_size)


    # 训练模型
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater, W, b)

    # 预测
    predict_ch3(net, test_iter, W, b)

