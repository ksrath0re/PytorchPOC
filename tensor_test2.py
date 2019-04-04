import torch
import numpy as np

t = torch.Tensor()
print(type(t))

print(t.dtype)
print(t.device)
print(t.layout)

#device = torch.device('cuda:0')
#print(device)
x = t.cuda()
print(x.device)

print(t + x)
#Error : RuntimeError: expected type torch.FloatTensor but got torch.cuda.FloatTensor
#Because one tensor is of cpu type and another is of gpu type.