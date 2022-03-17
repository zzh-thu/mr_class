import torch
import numpy as np

# x = torch.tensor(2.,requires_grad = True)
# a = torch.add(x,1)
# b = torch.add(x,2)
# y = torch.mul(a,b)
# y.backward()

# print(x.grad)

# with torch.no_grad():
#     x = torch.tensor(2.,requires_grad=True)
#     y=x**2
#     dy =torch.autograd.grad(x,x)
#     print(dy)

x = np.array(1)
x = x.astype(np.float32) 