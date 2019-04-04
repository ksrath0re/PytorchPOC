import torch
import numpy as np

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=torch.float32)

print(t.size())
#or
print(t.shape)

print(torch.tensor(t.shape).prod())
#or
print(t.numel())

t = t.reshape([2,6])
print(t)

t = t.reshape(1,4,3)
print(t)

print(t.reshape(1,12))
print(t.reshape(1,12).shape)
print(t.reshape(1,12).squeeze())
print(t.reshape(1,12).squeeze().shape)
print(t.reshape(1,12).squeeze().unsqueeze(dim=0))

x = torch.ones(3,4,5)
#print(x)
print(x.reshape(1,20,3))
print(x.reshape(1,20,3).squeeze()) #Reduces one dimension only when the first axes is 1 otherwise squeeze won't work
#This is further used to flatten the data
print(x.reshape(1,20,3).squeeze().shape)
print(x.reshape(1,20,3).squeeze().unsqueeze(dim=0).shape)
print(x.reshape(1,20,3).squeeze().unsqueeze(dim=1).shape)
print(x.reshape(1,20,3).squeeze().unsqueeze(dim=2).shape) #dim value decides where to put the newly added dimension
print(x.reshape(2,10,3).squeeze()) # squeeze Does not work here
print(x.reshape(2,10,3).squeeze().shape)

#Flattetening the data
t = t.reshape(1,-1)
t = t.squeeze()
print(t, t.shape)
