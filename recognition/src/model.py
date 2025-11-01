import torch
import torch.nn.functional as tnf
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
	"""
    A graph convolutional network layer.
	"""
	def __init__(self, dataset, hidden_channels):
		"""
		Parameters:
			dataset: the dataset being trained on (used for setting up
					 dimensions of the model)
			hidden_channels: the number of hidden channels to use for the
							 inner layer
		"""
		super().__init__()
		self.conv1 = GCNConv(dataset.num_features, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, dataset.y.size()[1])
		

	def forward(self, x, edge_index):
		"""
		Training step of the model.

		Parameters:
			x: node features
			edge_index: edge data
		"""
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = tnf.dropout(x, p = 0.5, training = self.training)
		x = self.conv2(x, edge_index)
		return x

def train_epoch(model, criterion, optimizer, data):
	"""
	Completes an epoch of training.

	Parameters:
		model: the model to train
		criterion: the algorithm to use to determinne loss
		optimizer: the optimizer algorithm to use
		data: the dataset to train over (with an appropriate train mask)

	Returns the loss for this epoc
	"""
	model.train()
	optimizer.zero_grad()
	out = model(data.x, data.edge_index)
	loss = criterion(out[data.train_mask], data.y[data.train_mask])
	loss.backward()
	optimizer.step()
	return loss

def test_model(model, data):
	"""
	Tests a trained model against a dataset.

	Parameters:
		model: the trained model to test
		data: the dataset to use to test

	Returns the accuracy of the test.
	"""
	model.eval()
	out = model(data.x, data.edge_index).argmax(dim = -1).reshape((-1, 1)).eq(torch.tensor([[0, 1, 2, 3]]))
	test_correct = torch.mul(out[data.test_mask], data.y[data.test_mask])
	test_acc = test_correct.sum() / data.test_mask.sum()
	print(test_correct.sum(), data.test_mask.sum(), test_acc, sep = "|")
	return test_acc
