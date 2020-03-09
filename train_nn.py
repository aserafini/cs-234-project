import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from MNIST_Net import MNIST_Net
from pol_net import pol_net
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time



# np.random.seed(69)
torch.manual_seed(2)

# 

# n_epochs = 15
# lr_list = np.logspace(-2, -3, n_epochs)

# train_acc = []
# test_acc = []


# for index in range(len(lr_list)):

# 	train_loss = net.train_epoch(index, lr_list[index])

# 	train_accuracy = net.train_accuracy()
# 	test_accuracy = net.test_accuracy()

# 	train_acc.append(train_accuracy)
# 	test_acc.append(test_accuracy)

# 	print('train accuracy:', train_accuracy)
# 	print('test accuracy:', test_accuracy)

# print('Finished Training')

# PATH = './mnist_net.pth'
# torch.save(net.state_dict(), PATH)

# plt.plot(range(1, n_epochs + 1), train_acc, label='train')
# plt.plot(range(1, n_epochs + 1), test_acc, label='test')
# plt.legend()
# plt.savefig('mnist.png')
net = MNIST_Net()
# start = time.time()
# net.train_accuracy()
# end = time.time()
# num_batches = len(net.trainloader)
# print(((end-start)*num_batches)/60.0)

model = pol_net(net, logger=None)
print("Let's train that model!!!!")
model.train()

