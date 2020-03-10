# import torch
# import torchvision
# import torchvision.transforms as transforms

import torch.nn as nn
# import torch.nn.functional as F

import torch.optim as optim

import numpy as np
from scipy.stats import ortho_group
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_pos_normal(mean, std):
	x = -1.
	while x <= 0:
		x = np.random.normal(loc = mean, scale = std)

	return x

class Ellipse():

	def __init__(self, dim):
		
		self.dim = dim 

		self.m = np.zeros((dim,)) 
		self.A = np.identity(dim)
		self.height = 0.

	def evaluate(self, x):
		return (x - self.m).T @ self.A @ (x - self.m) + self.height

	def grad(self, x):
		return 2 * self.A @ (x - self.m)	

	def sample_ellipse(self, mean_boundary, eigen_mean, eigen_std, height_boundary,):
		self.m = np.random.uniform(low = -mean_boundary, high = mean_boundary, size = (self.dim,))
		
		eigens = []
		for m in range(self.dim):
			eigens.append(get_pos_normal(mean = eigen_mean, std = eigen_std))
		diag = np.diag(eigens)
		basis_matrix = ortho_group.rvs(dim = self.dim)
		self.A = basis_matrix.T @ diag @ basis_matrix

		self.height = np.random.uniform(low = 0., high = height_boundary)

class Ellipse_Net(nn.Module):

	def __init__(self):
		super(Ellipse_Net, self).__init__()

		self.num_means = 1

		self.landscape_dim = 2
		self.mean_boundary = 10
		self.eigen_mean = 1
		self.eigen_std = 0.2
		self.height_boundary = 5 

		# maybe modify this init later
		self.weights = np.random.uniform(low = -self.mean_boundary, high = self.mean_boundary, size = (self.landscape_dim,))

		if self.num_means == 1:
			self.ell = Ellipse(self.landscape_dim)
			self.ell.sample_ellipse(self.mean_boundary, self.eigen_mean, self.eigen_std, self.height_boundary)
		else:
			self.ellipse_list = [Ellipse(self.landscape_dim) for i in range(self.num_means)]
			for ellipse in self.ellipse_list:
				ellipse.sample_ellipse(self.mean_boundary, self.eigen_mean, self.eigen_std, self.height_boundary)

	def evaluate(self, x):
		if self.num_means == 1:
			return self.ell.evaluate(x)
		else:	
			return np.sum([ellipse.evaluate(x) for ellipse in self.ellipse_list])

	def grad(self, x):
		if self.num_means == 1:
			return self.ell.grad(x)
		else:	
			return np.sum([ellipse.grad(x) for ellipse in self.ellipse_list])
		
	def plot(self):
		if self.landscape_dim == 2:
			x = np.linspace(-1 * self.mean_boundary, 1 * self.mean_boundary, 100)
			y = np.linspace(-1 * self.mean_boundary, 1 * self.mean_boundary, 100)

			# filling the heatmap, value by value
			fun_map = np.empty((x.size, y.size))
			for i in range(x.size):
				for j in range(y.size):
					fun_map[i,j] = self.evaluate(np.array([x[i], y[j]]))


			fig, ax = plt.subplots(figsize=(6,6))

			im = ax.imshow(fun_map)
			# ax.set_xticks(range(40))
			# ax.set_xticklabels(np.arange(-2 * self.mean_boundary, 2 * self.mean_boundary, 5))
			# ax.set_yticklabels(np.arange(-2 * self.mean_boundary, 2 * self.mean_boundary, 5))
			# ax.set_xlim(left = -2 * self.mean_boundary, right = 2 * self.mean_boundary)
			# ax.set_ylim(bottom = -2 * self.mean_boundary, top = 2 * self.mean_boundary)
			fig.colorbar(im)
			fig.savefig('landscape.png')

		return	

	def take_grad_step(self, grads, alpha = 1.):
		self.weights -= alpha * grads

	# def train_batch(self):
	# 	for i, data in enumerate(self.trainloader, 0):
	# 		images, labels = data

	# 		# zero the parameter gradients
	# 		self.optimizer.zero_grad()

	# 		# forward + backward + optimize
	# 		logits = self.forward(images)
	# 		loss = self.criterion(logits, labels)
	# 		loss.backward()

	# 		grads = []
	# 		for p in self.parameters():
	# 			grads.append(p.grad)

	# 		yield grads, loss, images, labels

	# def predict(self, images):
	# 	with torch.no_grad():
	# 		logits = self.forward(images)
	# 		_, predicted = torch.max(logits.data, 1)

	# 	return predicted
	
	# def test_loss(self):
	# 	with torch.no_grad():
	# 		loss = []
	# 		for data in self.testloader:
	# 			images, labels = data
	# 			logits = self.forward(images)
	# 			loss.append(self.criterion(logits, labels))
	# 		mean_loss = np.mean(loss)

	# 	return mean_loss

	# def train_loss(self):
	# 	with torch.no_grad():
	# 		loss = []
	# 		for data in self.trainloader:
	# 			images, labels = data
	# 			logits = self.forward(images)
	# 			loss.append(self.criterion(logits, labels))
	# 		mean_loss = np.mean(loss)

	# 	return mean_loss
	
	# def test_accuracy(self):
	# 	correct = 0
	# 	total = 0
	# 	for data in self.testloader:
	# 		images, labels = data
	# 		predicted = self.predict(images)
	# 		total += labels.size(0)
	# 		correct += (predicted == labels).sum().item()
	# 	return correct / total

	# def train_accuracy(self):
	# 	correct = 0
	# 	total = 0
	# 	for data in self.trainloader:
	# 		images, labels = data
	# 		predicted = self.predict(images)
	# 		total += labels.size(0)
	# 		correct += (predicted == labels).sum().item()
	# 	return correct / total

	# def unlearn(self):
		
	# 	def weight_reset(m):
	# 		if isinstance(m, nn.Linear):
	# 			m.reset_parameters()
		
	# 	self.apply(weight_reset)


	# def forward(self, x):
	# 	# print(x.shape)
	# 	x = x.view(-1, 784)
	# 	# print( "again!",x.shape)
	# 	# print('here')
	# 	x = F.relu(self.fc1(x))
	# 	# print('here2')
	# 	x = F.relu(self.fc2(x))
	# 	x = self.fc3(x)
	# 	return x


landscape = Ellipse_Net()

landscape.ellipse_list[0].m = np.array([-10., -10])
landscape.ellipse_list[1].m = np.array([-10., 10])

landscape.plot()
	





