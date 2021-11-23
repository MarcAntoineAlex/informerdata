from matplotlib import pyplot as plt
import numpy as np
import torch
from models.model import Normal
data = torch.from_numpy(np.load('/Users/marc-antoine/Desktop/0/arch0.npy'))
# model = Normal(num=30, length=300)
# means = torch.linspace(0, 300, 30, requires_grad=True)
print(data.shape, data)
# plt.hist(data, bins=20)
# plt.show()
plt.scatter(np.arange(len(data)), data, s=0.5)
plt.show()

a = torch.linspace(0, 10, 11)
print(a)
print(a[:200])

