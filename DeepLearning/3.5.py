import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import time
d2l.use_svg_display()

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
print(len(mnist_train), len(mnist_test))

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 测量数据加载时间
start_time = time.time()
for X, y in data.DataLoader(mnist_train, batch_size=18):
    continue
elapsed_time = time.time() - start_time
print(f"完整遍历训练集耗时: {elapsed_time:.4f} 秒")
print(f"最后一个batch的形状: {X.size()}")  # 输出最后一个batch的形状


# 获取一张图像
img, label = mnist_train[20]  # img形状: [1, 28, 28]

# 调整维度并显示
img_permuted = img.permute(1, 2, 0)  # 转为 [28, 28, 1]
plt.imshow(img_permuted.squeeze(), cmap='gray')  # squeeze()移除单通道维度
plt.title(f"Label: {get_fashion_mnist_labels([label])[0]}")
plt.axis('off')
plt.show()