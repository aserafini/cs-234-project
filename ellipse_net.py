# import torch
# import torchvision
# import torchvision.transforms as transforms
import torch
import torch.nn as nn
# import torch.nn.functional as F

import torch.optim as optim

import numpy as np
from scipy.stats import ortho_group
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pol_net import pol_net
from matplotlib.pyplot import cm

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

		self.descent_paths = []
		# self.current_descent = [[-15], [-15]]
		self.resets = 0

		# maybe modify this init later
		self.weights = np.random.uniform(low = -self.mean_boundary, high = self.mean_boundary, size = (self.landscape_dim,))
		# self.start = self.weights
		self.current_descent = [[self.weights[0]], [self.weights[1]]]
		# self.weights = np.array([-15., -15.],)

		if self.num_means == 1:
			self.ell = Ellipse(self.landscape_dim)
			self.ell.sample_ellipse(self.mean_boundary, self.eigen_mean, self.eigen_std, self.height_boundary)
		else:
			self.ellipse_list = [Ellipse(self.landscape_dim) for i in range(self.num_means)]
			for ellipse in self.ellipse_list:
				ellipse.sample_ellipse(self.mean_boundary, self.eigen_mean, self.eigen_std, self.height_boundary)

	def parameters(self):
		parameters = [torch.Tensor([0]*self.landscape_dim)]
		return parameters

	def evaluate(self, x):
		if self.num_means == 1:
			return self.ell.evaluate(x)
		else:	
			return np.sum([ellipse.evaluate(x) for ellipse in self.ellipse_list])

	def grad(self, x):
		if self.num_means == 1:
			return torch.Tensor(self.ell.grad(x))
		else:	
			return torch.Tensor(np.sum([ellipse.grad(x) for ellipse in self.ellipse_list]))
		
	def plot(self, save=True):
		if self.landscape_dim == 2:
			x = np.linspace(-2 * self.mean_boundary, 2 * self.mean_boundary, 100)
			y = np.linspace(-2 * self.mean_boundary, 2 * self.mean_boundary, 100)

			# filling the heatmap, value by value
			fun_map = np.empty((x.size, y.size))
			for i in range(x.size):
				for j in range(y.size):
					fun_map[i,j] = self.evaluate(np.array([x[i], y[j]]))


			# fig, ax = plt.subplots(figsize=(6,6))

			# im = ax.imshow(fun_map)
			plt.imshow(fun_map)
			plt.colorbar()
			# plt.scatter(-15, -15)
			# ax.set_xticks(range(40))
			# ax.set_xticklabels(np.arange(-2 * self.mean_boundary, 2 * self.mean_boundary, 5))
			# ax.set_yticklabels(np.arange(-2 * self.mean_boundary, 2 * self.mean_boundary, 5))
			# ax.set_xlim(left = -2 * self.mean_boundary, right = 2 * self.mean_boundary)
			# ax.set_ylim(bottom = -2 * self.mean_boundary, top = 2 * self.mean_boundary)
			# fig.colorbar(im)

			plt.savefig('plots/landscape.png')
			plt.close()

		return	

	def take_grad_step(self, grads, alpha = 1.):
		self.weights -= alpha * (grads[0]).data.numpy()

		# self.weights -= np.array([0., 0.])

		self.current_descent[0].append(self.weights[0])
		self.current_descent[1].append(self.weights[1])

	def train_batch(self):
		# for i, data in enumerate(self.trainloader, 0):

		images, labels = None, None

		# zero the parameter gradients
		# self.optimizer.zero_grad()

		# forward + backward + optimize
		# logits = self.forward(images)
		# loss = self.criterion(logits, labels)
		# loss.backward()
		loss = self.evaluate(self.weights)
		grads = [self.grad(self.weights)]
		# grads = []
		# for p in self.parameters():
		# 	grads.append(p.grad)

		yield grads, loss, images, labels

	def criterion(self, logits, labels):
		loss = self.evaluate(self.weights)
		# print(loss)
		# loss = np.linalg.norm(self.weights - np.array([-15.,-15.]))
		# loss = np.linalg.norm(self.weights - np.array([-30.,-30.]))
		return loss

	# don't eed
	# def predict(self, images):
	# 	with torch.no_grad():
	# 		logits = self.forward(images)
	# 		_, predicted = torch.max(logits.data, 1)

	# 	return predicted
	
	#d dot eed
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
	
	def test_accuracy(self):
		return "no accuracies with ellipses"
		# correct = 0
		# total = 0
		# for data in self.testloader:
		# 	images, labels = data
		# 	predicted = self.predict(images)
		# 	total += labels.size(0)
		# 	correct += (predicted == labels).sum().item()
		# return correct / total

	def train_accuracy(self):
		return "no accuracies with ellipses"
	# 	correct = 0
	# 	total = 0
	# 	for data in self.trainloader:
	# 		images, labels = data
	# 		predicted = self.predict(images)
	# 		total += labels.size(0)
	# 		correct += (predicted == labels).sum().item()
	# 	return correct / total

	def unlearn(self):
		# self.weights = np.random.uniform(low = -self.mean_boundary, high = self.mean_boundary, size = (self.landscape_dim,))
		# self.descent_paths.append(self.current_descent)
		if self.resets % 100 == 1:
			self.plot_descent_paths()
		# self.current_descent = [[-15], [-15]]
		# self.weights = np.array([-15., -15.],)

		self.weights = np.random.uniform(low = -self.mean_boundary, high = self.mean_boundary, size = (self.landscape_dim,))
		self.current_descent = [[self.weights[0]], [self.weights[1]]]
		self.resets += 1



	def forward(self, x):
		# # print(x.shape)
		# x = x.view(-1, 784)
		# # print( "again!",x.shape)
		# # print('here')
		# x = F.relu(self.fc1(x))
		# # print('here2')
		# x = F.relu(self.fc2(x))
		# x = self.fc3(x)
		return None

	def plot_descent_paths(self):
		print("current descent", self.current_descent[0])
		c=next(color)

		# plt.scatter(-15, -15, s= 200, c = "black")
		plt.scatter(self.current_descent[0][0], self.current_descent[1][0], s=100, c='black')
		plt.text(self.current_descent[0][0]+0.1, self.current_descent[1][0]+0.1, 'start', fontsize=9)

		plt.scatter(self.ell.m[0], self.ell.m[1], s= 100, c = "black")
		plt.text(self.ell.m[0]+0.1, self.ell.m[1]+0.1, 'min', fontsize=9)

		plt.scatter(self.current_descent[0], self.current_descent[1], c=c)

		plt.savefig('plots/descent' + str(self.resets) + '.png')
		plt.close()


# landscape.ellipse_list[0].m = np.array([-10., -10])
# landscape.ellipse_list[1].m = np.array([-10., 10])
torch.manual_seed(0)
np.random.seed(0)
landscape = Ellipse_Net()
landscape.plot(save=True)
plt.close()
color=iter(cm.cool(np.linspace(0,1,12)))
model = pol_net(landscape, logger=None)
print(model.net.ell.height)
print("Let's train that model!!!!")
model.train()

# print(model.avg_rewards)
# print(model.sigma_rewards)
# landscape.plot_descent_paths()
plt.savefig('plots/descent_paths')

plt.close()
plt.plot(model.avg_rewards)
plt.savefig('plots/avg_rewards')

plt.close()
plt.plot(model.lin_weights)
plt.savefig('plots/lin_weights')





