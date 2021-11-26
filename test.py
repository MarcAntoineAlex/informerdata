from matplotlib import pyplot as plt
import numpy as np
import torch
from models.model import Normal
model = Normal(num=30, length=3000)
means = torch.linspace(0, 3000, 30, requires_grad=True)
after = model(means).squeeze().detach().numpy()
plt.plot(after)
plt.show()
data = []

# for i in range(9):
#     data.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/377695/0/arch{}.npy'.format(i))))
#
# normal = Normal(num=852, length=8521)
#
# for i in range(1, 9):
#     afteri = normal(data[i]).squeeze().detach().numpy()
#     afteri_1 = normal(data[i-1]).squeeze().detach().numpy()
#     diff = afteri - afteri_1
#     plt.scatter(np.arange(len(diff)), diff, s=0.5)
#     plt.show()





