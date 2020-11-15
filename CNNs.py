##########################################
'''
- Uses a Densenet161 as base feature extractor
- Cascades two AvgPool2d and Flatten to obtain a 2208 sized feature vector

- an autoencoder made of - a sequential auto_encoder and a sequential auto_decoder : obtains a 128 length though vector
- first the video data should be used to train the auto-encoder
- during autoencoder training, use model(x, autoencode=True)
- then the autoencoder params are freezed using freeze_autoencoder()

- then use model(x) to obtain features as auto_encoder(base_cnn(x))
'''
##########################################
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T

class get_CNN(nn.Module):

	def __init__(self):
		super(get_CNN, self).__init__()
		
		base_model = models.densenet161(pretrained=True)

		for param in base_model.parameters():
			param.requires_grad = False

		children_list = list(base_model.children())[:-1]
		
		children_list.append(nn.AvgPool2d(3, stride=2))
		children_list.append(nn.AvgPool2d(3, stride=3))
		children_list.append(nn.Flatten())

		self.cnn_model = nn.Sequential(*children_list)

		self.auto_encoder = nn.Sequential(
							nn.Linear(2208, 1024),
							nn.ReLU(),
							nn.Linear(1024, 512),
							nn.ReLU(),
							#nn.Linear(512, 128),
							#nn.ReLU()
							)

		self.auto_decoder = nn.Sequential(
							#nn.Linear(128, 512),
							#nn.ReLU(),
							nn.Linear(512, 1024),
							nn.ReLU(),
							nn.Linear(1024, 2208),
							nn.ReLU()
							)


	def forward(self, x, cnn_output=False, autoencode=False):

		if cnn_output:
			return self.cnn_model(x)
		
		elif autoencode:
			return self.auto_decoder(self.auto_encoder(self.cnn_model(x)))
		
		else:
			return self.auto_encoder(self.cnn_model(x))	

	def freeze_autoencoder(self):

		for param in self.auto_encoder.parameters():
			param.requires_grad = False

		for param in self.auto_decoder.parameters():
			param.requires_grad = False

	def save_model(self, filename, encoder=False, autoencoder=False):
		
		assert (encoder and not autoencoder) or (not encoder and autoencoder) 	#encoder XOR autoencoder

		if encoder:		#save only encoder state dict
			torch.save(self.auto_encoder.state_dict(), filename)

		elif autoencoder:	#save both encoder and decoder state_dict
			torch.save([self.auto_encoder.state_dict(), self.auto_decoder.state_dict()], filename)

	def load_model(self, path):

		data = torch.load(path)

		if len(data) == 1:	#only encoder state dict
			self.auto_encoder.load_state_dict(data)

		elif len(data) == 2:	#encoder decoder separate 
			self.auto_encoder.load_state_dict(data[0])
			self.auto_decoder.load_state_dict(data[1])


if __name__=='__main__':

	INPUT = torch.randn(1,3,224,224)

	model = get_CNN()

	print(model(INPUT).size())
