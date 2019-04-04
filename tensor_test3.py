#Tensor creation options
import torch
import numpy as np

data = np.array([1,2,3])
print(data, type(data))

data1 = torch.Tensor(data)
print(data1)

data2 = torch.tensor(data)
print(data2)

data3 = torch.as_tensor(data)
print(data3)

data4 = torch.from_numpy(data)
print(data4)

data5 = np.array([1,2.4,3])

data51 = torch.Tensor(data5)
print(data51)

data52 = torch.tensor(data5)
print(data52)

data53 = torch.as_tensor(data5)
print(data53)

data54 = torch.from_numpy(data5)
print(data54)

print(torch.get_default_dtype())