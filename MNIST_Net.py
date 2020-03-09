import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import numpy as np

class MNIST_Net(nn.Module):

	def __init__(self):
		super(MNIST_Net, self).__init__()

		transform = transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(
				(0.1307,), (0.3081,))])

		trainset = torchvision.datasets.MNIST(root='./data', train=True,
			download=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024,
			shuffle=True, num_workers=0)

		testset = torchvision.datasets.MNIST(root='./data', train=False,
			download=True, transform=transform)
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=1024,
			shuffle=False, num_workers=0)

		self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

		self.fc1 = nn.Linear(784, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

		self.criterion = nn.CrossEntropyLoss() #softmax lives in here
		self.optimizer = optim.SGD(self.parameters(), lr=1, momentum=0)

	# def get_weights(self):
	# 	weights = np.zeros(len(self.parameters()))

	# 	# HI, JC here:
	# 	# faster is weights = np.array([p.data for p in self.parameters()])
	# 	for idx, p in enumerate(self.parameters()):
	# 		print("parameter", p)
	# 		weights[idx] = p.data
	# 	return weights

	def take_grad_step(self, grads):
		for idx, p in enumerate(self.parameters()):
			p.grad = grads[idx].clone()

		self.optimizer.step()

	def train_batch(self):
		for i, data in enumerate(self.trainloader, 0):
			images, labels = data

			# zero the parameter gradients
			self.optimizer.zero_grad()

			# forward + backward + optimize
			logits = self.forward(images)
			loss = self.criterion(logits, labels)
			loss.backward()

			grads = []
			for p in self.parameters():
				grads.append(p.grad)

			yield grads, loss, images, labels

	def predict(self, images):
		with torch.no_grad():
			logits = self.forward(images)
			_, predicted = torch.max(logits.data, 1)

		return predicted
	
	def test_loss(self):
		with torch.no_grad():
			loss = []
			for data in self.testloader:
				images, labels = data
				logits = self.forward(images)
				loss.append(self.criterion(logits, labels))
			mean_loss = np.mean(loss)

		return mean_loss

	def train_loss(self):
		with torch.no_grad():
			loss = []
			for data in self.trainloader:
				images, labels = data
				logits = self.forward(images)
				loss.append(self.criterion(logits, labels))
			mean_loss = np.mean(loss)

		return mean_loss
	
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


