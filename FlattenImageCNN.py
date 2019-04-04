import torch

t1 = torch.tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])

t2 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])

t3 = torch.tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])

t = torch.stack((t1,t2,t3))
print(t.shape)
print(t)
#Now let's convert to grayscale channel

t = t.reshape(3,1,4,4)
print(t.shape)

#Flattening the tensor batch
print(t.reshape(1,-1)[0])
#or
print(t.reshape(-1))
#or
print(t.flatten())

#But here we have flattened all the 3 images into one single tensor. Instead we wanted 3 different images flattened separately.
#So we will choose an axis from where we'd flatten the images
#Two ways to do it.

print(t.reshape(3,16))
#or
print(t.flatten(start_dim=1))
#the start_dim parameter. This tells the flatten() method which axis it should start the flatten operation.
#The one here is an index, so it’s the second axis which is the color channel axis. We skip over the batch axis so
#to speak, leaving it intact.
