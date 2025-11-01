import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import pandas as pd
import src.csv_loading as csv_loading
import src.encoders as encoders
import src.model as gcn
import src.visualise as vis

def extract_csv_data(root, device = None, num_val = 0.1, num_test = 0.2):
	"""
	Extracts and preprocesses the musae graph data from csv files inside the provided root folder.

	Parameters:
		root: a file path to the location where the musae folder was downloaded to
			  (see https://snap.stanford.edu/data/facebook-large-page-page-network.html).
			  Assumes file was unzipped in root/
		device: the device for torch to use for preprocessing, if necessary
		num_val: proportion of dataset nodes to use for validation
		num_test: proportion of dataset nodes to use for testing

	Returns the preprocessed dataset for feeding to PyTorch
	"""
	
	# Create encoders
	name_encoder = encoders.NameEncoder(
		"sentence-transformers/distiluse-base-multilingual-cased-v2",
		device = device
	)

	type_encoder = encoders.PageEncoder(["government", "company", "tvshow", "politician"])

	# Extract the node data from csv
	node_xs, node_ys, row_mapping = csv_loading.load_node_csv(
		root + "/facebook_large/facebook_large/musae_facebook_target.csv",
		"id",
		x_encoders = {"page_name" : name_encoder},
		y_encoders = {"page_type" : type_encoder}
	)

	# Extract the edge data from csv
	edges, _ = csv_loading.load_edge_csv(
		root + "/facebook_large/facebook_large/musae_facebook_edges.csv",
		None,
		"id_1",
		"id_2",
		row_mapping,
		directed = False
	)

	# Create dataset
	data = Data(x = node_xs, edge_index = edges, y = node_ys)

	# Split into training, validation and testing nodes
	transform = RandomNodeSplit(num_val = num_val, num_test = num_test)
	
	return transform(data)


if __name__ == '__main__':
	# Setup device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	data = extract_csv_data(".", num_test = 0.5)
	
	print(data.num_nodes,
          data.num_edges,
          data.num_features,
          data.y.size(),
          sum(data.y),
          sep = "|"
	)

	model = gcn.GCN(data, 150)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
	criterion = torch.nn.CrossEntropyLoss()

	print(data.edge_index)
	
	for epoch in range(0, 100):
		loss = gcn.train_epoch(model, criterion, optimizer, data)
		print(f"Epoch {epoch}: loss = {loss:.4f}")

	test_acc = gcn.test_model(model, data)
	print(f"Test Accuracy: {test_acc:.4f}")

	vis.tsne_plot(model(data.x, data.edge_index), true_colour = data.y.argmax(dim = -1))
