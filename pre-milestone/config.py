# holds important stuff like numbers and dims

class MNIST_config:
	def __init__(self):
		self.transform = transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])

		self.trainset = torchvision.datasets.MNIST(root='./data', train=True,
		                                        download=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
		                                          shuffle=True, num_workers=2)

		self.testset = torchvision.datasets.MNIST(root='./data', train=False,
		                                       download=True, transform=transform)
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
		                                         shuffle=False, num_workers=2)

		self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

		self.dim_in = 784
		self.dim_out = 10

		self.fc_layers = [[self.dim_in, 100],
		 [100, 100],
		 [100, self.dim_out]]


		 self.layers = [['flatten'],
		 ['conv',23,234,332],
		 ['relu']]

