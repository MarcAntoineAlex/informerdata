from matplotlib import pyplot as plt
import numpy as np
import torch
from models.model import Normal
# data1 = torch.from_numpy(np.load('/Users/marc-antoine/Desktop/374575/0/arch0.npy'))[:, None, None]
# data2 = torch.from_numpy(np.load('/Users/marc-antoine/Desktop/374575/0/arch1.npy'))
# data3 = torch.from_numpy(np.load('/Users/marc-antoine/Desktop/374575/0/arch2.npy'))
model = Normal(num=30, length=300)
means = torch.linspace(0, 300, 30, requires_grad=True)
print(means)
res = model(means)
for n, p in model.named_parameters():
    print(n, p)
numres = res.detach().numpy()
plt.plot(numres)
plt.ylim(0, 2)
plt.show()
loss = res.sum()
loss.backward()
print(means.grad)
# plt.hist(data, bins=20)
# plt.show()
# plt.scatter(np.arange(len(data)), data, s=0.5)
# plt.show()
# plt.hist(2/(1+np.exp(-data)), bins=20)
# plt.show()
