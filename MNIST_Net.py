import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class MNIST_Net(nn.Module):

	def __init__(self):
		super(MNIST_Net, self).__init__()

		transform = transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(
				(0.1307,), (0.3081,))])

		trainset = torchvision.datasets.MNIST(root='./data', train=True,
			download=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
			shuffle=True, num_workers=0)

		testset = torchvision.datasets.MNIST(root='./data', train=False,
			download=True, transform=transform)
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
			shuffle=False, num_workers=0)

		self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

		self.fc1 = nn.Linear(784, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

		self.criterion = nn.CrossEntropyLoss() #softmax lives in here
		self.optimizer = optim.SGD(self.parameters(), lr=6969696969, momentum=0) 

	def train_epoch(self, epoch, lr):
		print('potentially less fucking stupid learning rate', lr)
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

		running_loss = 0.0
		train_loss = 0.0
		for i, data in enumerate(self.trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data

			# zero the parameter gradients
			self.optimizer.zero_grad()

			# forward + backward + optimize
			outputs = self.forward(inputs)
			loss = self.criterion(outputs, labels)
			loss.backward()
			self.optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 2000))
				train_loss += running_loss
				running_loss = 0.0

		return train_loss / 15000

	def predict(self, images):
		with torch.no_grad():
			outputs = self.forward(images)
			_, predicted = torch.max(outputs.data, 1)

		return predicted

	def test_accuracy(self):
		correct = 0
		total = 0
		for data in self.testloader:
			images, labels = data
			predicted = self.predict(images)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		return correct / total

	def train_accuracy(self):
		correct = 0
		total = 0
		for data in self.trainloader:
			images, labels = data
			predicted = self.predict(images)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		return correct / total


	def unlearn(self):
		
		def weight_reset(m):
			if isinstance(m, nn.Linear):
				m.reset_parameters()
		
		self.apply(weight_reset)


	def forward(self, x):
		# print(x.shape)
		x = x.view(-1, 784)
		# print( "again!",x.shape)
		# print('here')
		x = F.relu(self.fc1(x))
		# print('here2')
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


