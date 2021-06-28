"""
Loads data
"""

import os
import PIL
import torch
import pandas
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class NonTemporalDataset(Dataset):
	def __init__(self, dataset_file_path, loader_type, opt, domain=None):
		self.opt = opt
		if domain is None:
			self.domain = self.opt.source_domain
		else:
			self.domain = domain

		dataset_file = pandas.read_csv(dataset_file_path)
		self.file_path = dataset_file['file_path'].tolist()
		self.v_labels = dataset_file['valence'].tolist()
		self.a_labels = dataset_file['arousal'].tolist()

		if loader_type == 'train':
			transform_list = [	transforms.Resize(112),
								transforms.RandomHorizontalFlip(),
								transforms.ToTensor(),
								transforms.Normalize(mean=[0.485, 0.456, 0.406],
													std=[0.229, 0.224, 0.225]),
								transforms.RandomErasing(scale=(0.02,0.25))]
		else:
			transform_list = [	transforms.Resize(112),
								transforms.ToTensor(),
								transforms.Normalize(mean=[0.485, 0.456, 0.406],
													std=[0.229, 0.224, 0.225])]

		self.transform = transforms.Compose(transform_list)

	def __getitem__(self, idx):
		image_path = self.file_path[idx]
		if self.domain == 'AffectNet':
			image = PIL.Image.open('../data/AffectNet/' + image_path)
		elif self.domain == 'Aff-Wild2':
			image = PIL.Image.open('../data/Aff-Wild2/cropped_aligned/' + image_path)
		elif self.domain == 'SEWA':
			image = PIL.Image.open('../data/SEWA/prep_SEWA/' + image_path)

		image = self.transform(image).float()
		if not self.opt.backbone == 'resnet18':
			image = torch.unsqueeze(image, 1)
			image = torch.cat(self.opt.frames*[image], dim=1)

		v_label = self.v_labels[idx]
		a_label = self.a_labels[idx]

		return image, v_label, a_label, idx

	def __len__(self):

		return len(self.file_path)

	def __getlabel__(self, idx):
		v_label = self.v_labels[idx]
		a_label = self.a_labels[idx]

		return v_label, a_label, idx

def collate_non_fn_temporal_dataset(data):
	images, v_labels, a_labels, idx = zip(*data)

	images = torch.stack(images)
	v_labels = torch.FloatTensor(v_labels)
	a_labels = torch.FloatTensor(a_labels)
	idx = torch.LongTensor(idx)

	return images, v_labels, a_labels, idx

def get_non_temporal_dataset(	dataset_file_path, batch_size, balance,
								shuffle, workers_num, collate_fn, loader_type, opt, domain=None):
	dataset = NonTemporalDataset(	dataset_file_path=dataset_file_path,
									loader_type=loader_type,
									opt=opt,
									domain=domain)

	if balance:
		dataloader = DataLoader(dataset=dataset,
								batch_size=batch_size,
								shuffle=shuffle,
								num_workers=workers_num,
								collate_fn=collate_fn,
								pin_memory=False)
	else:
		dataloader = DataLoader(dataset=dataset,
								batch_size=batch_size,
								shuffle=shuffle,
								num_workers=workers_num,
								collate_fn=collate_fn,
								pin_memory=False)

	return dataloader

def get_loaders_non_temporal_dataset(dataset_file_path, loader_type, opt, domain=None):
	if loader_type == 'train':
		dataloader = get_non_temporal_dataset(	dataset_file_path=dataset_file_path,
												batch_size=opt.batch_size,
												balance=True,
												shuffle=True,
												workers_num=opt.workers_num,
												collate_fn=collate_non_fn_temporal_dataset, 
												loader_type=loader_type, 
												opt=opt,
												domain=domain)
	else:
		dataloader = get_non_temporal_dataset(	dataset_file_path=dataset_file_path,
												batch_size=opt.batch_size,
												balance=False,
												shuffle=False,
												workers_num=opt.workers_num,
												collate_fn=collate_non_fn_temporal_dataset,
												loader_type=loader_type, 
												opt=opt,
												domain=domain)

	return dataloader

def get_dataloader(dataset_file_path, loader_type, opt, domain=None):
	dataloader = get_loaders_non_temporal_dataset(	dataset_file_path,
													loader_type, opt, domain=domain)

	return dataloader
