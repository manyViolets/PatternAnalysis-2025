import torch
from modules import GCN
from dataset import load_csv_data, DATA_PATH

NUM_HIDDEN_CHANNELS = 150
NUM_EPOCHS = 100
SAVE_PATH = "../out/gcn_model"

def train_epoch(model, criterion, optimizer, data):
	"""
	Completes an epoch of training.

	Parameters:
		model: the model to train
		criterion: the algorithm to use to determinne loss
		optimizer: the optimizer algorithm to use
		data: the dataset to train over (with an appropriate train mask)

	Returns the loss for this epoch
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
	return test_acc

def save_model(model, path):
	"""
	Saves the model for further use.

	Parameters:
		model: the model to save
		path: the path to a folder to save the model in
	"""
	torch.save(model.state_dict(), path)

if __name__ == "__main__":
	# Setup device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Extract data
	data = load_csv_data(DATA_PATH, num_test = 0.3)

	# Set up model, optimizer and criterion
	model = GCN(data.num_features, data.y.size()[1], NUM_HIDDEN_CHANNELS)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
	criterion = torch.nn.CrossEntropyLoss()

	# Train model
	for epoch in range(0, NUM_EPOCHS):
		loss = train_epoch(model, criterion, optimizer, data)
		print(f"Epoch {epoch}: loss = {loss:.4f}")

	# Test model
	test_acc = test_model(model, data)
	print(f"Test Accuracy: {test_acc:.4f}")

	# Save model
	save_model(model, SAVE_PATH)
