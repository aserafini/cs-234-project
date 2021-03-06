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

import logging

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

		# self.descent_paths = []
		self.resets = 0
		self.plot_freq = 1000

		# maybe modify this init later
		self.weights = np.random.uniform(low = -self.mean_boundary, high = self.mean_boundary, size = (self.landscape_dim,))

		# self.weights = np.array([-15., -15.],)
		self.current_descent = [[self.weights[0]], [self.weights[1]]]

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
			plt.scatter([40], [40])
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

		self.current_descent[0].append(self.weights[0])
		self.current_descent[1].append(self.weights[1])

	def train_batch(self):

		images, labels = None, None

		loss = self.evaluate(self.weights)
		grads = [self.grad(self.weights)]

		yield grads, loss, images, labels

	def criterion(self, logits, labels):
		loss = self.evaluate(self.weights)
		return loss
	
	def test_accuracy(self):
		return "no accuracies with ellipses"

	def train_accuracy(self):
		return "no accuracies with ellipses"

	def unlearn(self, color_iter):
		# self.weights = np.random.uniform(low = -self.mean_boundary, high = self.mean_boundary, size = (self.landscape_dim,))
		# self.descent_paths.append(self.current_descent)
		if self.resets % self.plot_freq == 1:
			self.plot_descent_landscape(color_iter[0], True)
			
		# self.current_descent = [[-15], [-15]]
		# self.weights = np.array([-15., -15.],)
		self.weights = np.random.uniform(low = -self.mean_boundary, high = self.mean_boundary, size = (self.landscape_dim,))
		self.current_descent = [[self.weights[0]], [self.weights[1]]]

		self.ell.sample_ellipse(self.mean_boundary, self.eigen_mean, self.eigen_std, self.height_boundary)

		self.resets += 1

	def forward(self, x):
		return None

	def plot_descent_paths(self, color, separate_graphs = True):
		# print("current descent", self.current_descent[0])
		
		c = color

		plt.scatter(self.current_descent[0][0], self.current_descent[1][0], s=100, c='black')
		# plt.text(self.current_descent[0][0]+0.1, self.current_descent[1][0]+0.1, 'start', fontsize=9)

		plt.scatter(self.ell.m[0], self.ell.m[1], s= 100, c = "black")
		# plt.text(self.ell.m[0]+0.1, self.ell.m[1]+0.1, 'min', fontsize=9)

		if separate_graphs == True:
			plt.scatter(self.current_descent[0], self.current_descent[1], c=c)
			plt.savefig('plots/descent' + str(self.resets) + '.png')
			plt.close()
		else:
			plt.scatter(self.current_descent[0], self.current_descent[1], c=c)

	def plot_descent_landscape(self, color, separate_graphs = True):
		# print("current descent", self.current_descent[0])
		max_x = np.max([np.max(self.current_descent[0]), self.ell.m[0]])
		min_x = np.min([np.min(self.current_descent[0]), self.ell.m[0]])
		max_y = np.max([np.max(self.current_descent[1]), self.ell.m[1]])
		min_y = np.min([np.min(self.current_descent[1]), self.ell.m[1]])
		x_buff = (max_x - min_x)*.3
		y_buff = (max_x - min_x)*.3
		max_x += x_buff
		min_x -= x_buff
		max_y += y_buff
		min_y -= y_buff

		x = np.linspace(min_x, max_x, 100)
		y = np.linspace(min_y, max_y, 100)

		# filling the heatmap, value by value
		fun_map = np.empty((x.size, y.size))
		for i in range(x.size):
			for j in range(y.size):
				fun_map[99 - j, i] = self.evaluate(np.array([x[i], y[j]]))

		plt.imshow(fun_map, cmap = 'BuPu_r')
		plt.colorbar()

		x_descent = 100 * (self.current_descent[0] - min_x) / (max_x - min_x)
		y_descent = 100 - 100 * (self.current_descent[1] - min_y) / (max_y - min_y)

		c = color
		start_dot_x = 100 * (self.current_descent[0][0] - min_x) / (max_x - min_x)
		start_dot_y = 100 - 100 * (self.current_descent[1][0] - min_y) / (max_y - min_y)

		end_dot_x = 100 * (self.ell.m[0] - min_x) / (max_x - min_x)
		end_dot_y = 100 - 100 * (self.ell.m[1] - min_y) / (max_y - min_y)

		plt.scatter([start_dot_x], [start_dot_y], s=100, c='black')
		plt.scatter([end_dot_x], [end_dot_y], s= 100, c = "black")

		if separate_graphs == True:
			color_list = cm.summer(np.linspace(0,1, len(x_descent)))

			plt.scatter(x_descent, y_descent, c=color_list, marker='x')
			plt.savefig('plots/descent_landscape' + str(self.resets) + '.png')
			plt.close()
		else:
			plt.scatter(self.current_descent[0], self.current_descent[1], c=c)	


mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)

torch.manual_seed(0)
np.random.seed(0)

landscape = Ellipse_Net()
landscape.plot(save = True)
plt.close()

model = pol_net(landscape, logger=None)
print(model.net.ell.height)
print("Let's train that model!!!!")
model.train()

plt.savefig('plots/descent_paths')

plt.close()
plt.plot(model.avg_rewards)
plt.savefig('plots/avg_rewards')

plt.close()
plt.plot(model.lin_weights)
plt.savefig('plots/lin_weights')





