import torch
x = torch.arange(12,dtype=torch.float32).reshape(-1,4)
y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
X = torch.cat((x,y),dim=0)#行链接
Y = torch.cat((x,y),dim=1)#列连接
print(X,'\n',X[0:2],'\n',X[2,0:2])
