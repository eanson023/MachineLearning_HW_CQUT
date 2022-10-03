import torch

X=torch.arange(10).reshape(10,1)
print((X>5).type(torch.float64))