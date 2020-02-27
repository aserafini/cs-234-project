import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from neural_net import MNIST_Net
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# np.random.seed(69)
torch.manual_seed(69)

net = MNIST_Net()

n_epochs = 15
lr_list = np.logspace(-2, -3, n_epochs)

train_acc = []
test_acc = []


for index in range(len(lr_list)):

	train_loss = net.train_epoch(index, lr_list[index])

	train_accuracy = net.train_accuracy()
	test_accuracy = net.test_accuracy()

	train_acc.append(train_accuracy)
	test_acc.append(test_accuracy)

	print('train accuracy:', train_accuracy)
	print('test accuracy:', test_accuracy)

print('Finished Training')

PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)

plt.plot(range(1, n_epochs + 1), train_acc, label='train')
plt.plot(range(1, n_epochs + 1), test_acc, label='test')
plt.legend()
plt.savefig('mnist.png')






