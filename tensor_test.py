import torch
import numpy as np

print(torch.__version__)

t = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]

t = torch.tensor(t)
x = torch.Tensor()
print(t)
print(x)

print(t.shape)

t = t.reshape(1,9)
print(t)
print(t.shape)