#backward
#https://blog.csdn.net/baidu_38797690/article/details/122180655


import torch

x = torch.arange(4.0)
y = torch.arange(4.0)

x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)
y.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)

z = x ** 2 + y ** 3
z.backward(torch.ones_like(z))  # 反向传播计算梯度
print(y.grad)