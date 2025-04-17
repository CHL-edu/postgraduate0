import math
import numpy as np
import matplotlib.pyplot as plt

def normal(x, mu, sigma):
    """正态分布概率密度函数"""
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 生成x轴数据
x = np.arange(-7, 7, 0.01)

# 定义不同参数组合
params = [(0, 1), (0, 2), (3, 1)]

# 创建图形
plt.figure(figsize=(4.5, 2.5))

# 绘制每条曲线
for mu, sigma in params:
    plt.plot(x, normal(x, mu, sigma), label=f'mean {mu}, std {sigma}')

# 添加标签和图例
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()

# 显示图形
plt.show()