import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from neural_net import MNIST_Net

# np.random.seed(69)
torch.manual_seed(69)

net = MNIST_Net()

n_epochs = 4
lr_list = np.logspace(-2, -3, n_epochs)
for index in range(len(lr_list)):
	train_loss = net.train_epoch(index, lr_list[index])
	test_loss = net.test_loss()
	print('train loss:', train_loss)
	print('test loss:', test_loss)

print('Finished Training')

PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)


# criterion = nn.CrossEntropyLoss() #softmax lives in here
# optimizer = optim.SGD(net.parameters(), lr=None, momentum=0) 

# # lr scheduler of some sort goes here i guess??? confused

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(net.trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0




