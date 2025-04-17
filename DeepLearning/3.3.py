import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def synthetic_data(w, b, num_examples):  # @save
    """生成y=X@wT+b+噪声"""
    # 1. 生成特征数据
    X = torch.normal(0, 1, (num_examples, len(w)))
    print((num_examples, len(w)))
    # 2. 计算真实y值（无噪声）
    y_clean = X@w + b

    # 3. 添加噪声
    y_noisy = y_clean + torch.normal(0, 0.01, y_clean.shape)
    print(X.size(),y_noisy.size())
    return X, y_noisy.reshape((-1, 1))

batch_size  =10
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

true_w = torch.tensor([2, -3.4, 3.0])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


#input data
w = torch.normal(0, 0.01, size=true_w.size(), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}','*****\n',true_w,'\n',w)
print(f'b的估计误差: {true_b - b}','*****',b)