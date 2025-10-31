import torch
from sentence_transformers import SentenceTransformer

class NameEncoder():
	"""
	Encodes a string of a page name into a torch-friendly tensor in a lower dimensional space.
	Uses a specified pre-trained model from the sentence-transformers library.
	"""
	def __init__(self, model_name, device = None):
		"""
		Parameters:
			model_name: Name of the model to use to encode
			device: The device for PyTorch
		"""
		self._model = SentenceTransformer(model_name)
		self._device = device
	
	# Tells torch not to use an automatic gradient, since the model has its own
	@torch.no_grad()
	def __call__(self, df):
		"""
		Parameters:
			df: the dataframe to encode

		Returns the encoding of the dataframe.
		"""
		x = self._model.encode(df.to_numpy(), show_progress_bar = True, convert_to_tensor = True,
							   device = self._device)
		# Return the encoded data to the cpu
		return x.cpu()
		

class PageEncoder():
	"""
	A one-hot encoding of the page type of a node.
	"""
	def __init__(self, categories):
		"""
		Parameters:
			categories: A list of the possible categories for a page
		"""
		self._categories = categories
		
	def __call__(self, df):
		"""
		Parameters:
			df: the dataframe to encode

		Returns the encoding of the dataframe.
		"""
		matches = df.transform(
			[lambda x: x == category for category in self._categories],
			axis = 0
		).to_numpy()
		
		return torch.tensor(matches, dtype = torch.float64)
