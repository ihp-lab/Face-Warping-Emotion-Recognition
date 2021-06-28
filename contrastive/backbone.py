"""
Defines models
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict

from torchvision.models.video import r3d_18, r2plus1d_18

def set_requires_grad(model, requires_grad=True):
	for param in model.parameters():
		param.requires_grad = requires_grad

def init_weights(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		torch.nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)

class Flatten(nn.Module):
	def forward(self, input):
		
		return input.view(input.size(0), -1)

class Emotion_Recognizer(nn.Module):
	def __init__(self, opt):
		super(Emotion_Recognizer, self).__init__()

		self.encoder = Encoder(opt)
		self.decoder = Decoder(opt)

	def forward(self, images):
		features = self.encoder(images)
		labels = self.decoder(features)

		return labels

	def get_parameters(self) -> List[Dict]:
		params = [
			{"params": self.encoder.parameters(), "lr_mult": 1.},
			{"params": self.decoder.parameters(), "lr_mult": 1.}
		]
		return params

class Encoder(nn.Module):
	def __init__(self, opt):
		super(Encoder, self).__init__()

		if opt.backbone == 'r3d':
			r3d = r3d_18(pretrained=True)
			r3d_layers = list(r3d.children())[:-1]
			self.model = nn.Sequential(*r3d_layers)
		elif opt.backbone == 'r21d':
			r21d = r2plus1d_18(pretrained=True)
			r21d_layers = list(r21d.children())[:-1]
			self.model = nn.Sequential(*r21d_layers)
		elif opt.backbone == 'resnet18':
			resnet18 = torchvision.models.resnet18(pretrained=True)
			resnet18_layers = list(resnet18.children())[:-1]
			self.model = nn.Sequential(*resnet18_layers)

		self.dropout_rate = opt.dropout_rate
		self.dropout = nn.Dropout(self.dropout_rate)
		self.flatten = Flatten()
		self.fc1 = nn.Linear(512, 256)
		self.fc2 = nn.Linear(256, 128)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.model(x)
		x = self.flatten(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = F.normalize(x, p=2, dim=1)
		x = self.dropout(x)

		return x

class Decoder(nn.Module):
	def __init__(self, opt):
		super(Decoder, self).__init__()

		self.fc = nn.Linear(128, 1)
		self.dropout_rate = opt.dropout_rate
		self.dropout = nn.Dropout(self.dropout_rate)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.fc(x)
		x = self.dropout(x)
		x = self.tanh(x)

		return x
