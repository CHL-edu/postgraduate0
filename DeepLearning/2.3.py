import torch
X = torch.tensor([0,1,2,3])
Y = torch.tensor([1,2,3,4])

Z = torch.zeros_like(Y)
print(Z)
print('id(Z):', id(Z))
Z[:] = X + Y
print(Z)
print('id(Z):', id(Z))