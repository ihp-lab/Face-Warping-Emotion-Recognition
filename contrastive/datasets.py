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
	def __init__(self, dataset_file_path, opt):
		self.opt = opt
		self.frames = self.opt.frames

		dataset_file = pandas.read_csv(dataset_file_path)
		self.real_path = dataset_file['real_path'].tolist()
		self.fake_path = dataset_file['fake_path'].tolist()

		transform_list = [	transforms.Resize(112),
							transforms.ToTensor(),
							transforms.Normalize(mean=[0.485, 0.456, 0.406],
												std=[0.229, 0.224, 0.225])]

		self.transform = transforms.Compose(transform_list)

	def __getitem__(self, idx):
		real_image_path = self.real_path[idx]
		fake_image_path = self.fake_path[idx]

		real_image_list = []
		fake_image_list = []

		for i in range(self.frames):
			_idx = int(real_image_path[-9:-4])+i
			real_image_path_i = real_image_path[:-9] + str(format(_idx, '05d')) + real_image_path[-4:]
			real_image = PIL.Image.open(real_image_path_i)
			real_image = self.transform(real_image).float()
			real_image = torch.unsqueeze(real_image, 1)
			real_image_list.append(real_image)

			fake_image_path_i = fake_image_path[:-9] + str(format(_idx, '05d')) + fake_image_path[-4:]
			fake_image = PIL.Image.open(fake_image_path_i)
			fake_image = self.transform(fake_image).float()
			fake_image = torch.unsqueeze(fake_image, 1)
			fake_image_list.append(fake_image)

		real_images = torch.cat(real_image_list, dim=1)
		fake_images = torch.cat(fake_image_list, dim=1)

		return real_images, fake_images

	def __len__(self):

		return len(self.real_path)

def collate_non_fn_temporal_dataset(data):
	real_images, fake_images = zip(*data)

	real_images = torch.stack(real_images)
	fake_images = torch.stack(fake_images)

	return real_images, fake_images

def get_non_temporal_dataset(	dataset_file_path, batch_size, shuffle,
								workers_num, collate_fn, opt):
	dataset = NonTemporalDataset(	dataset_file_path=dataset_file_path, opt=opt)

	dataloader = DataLoader(	dataset=dataset,
								batch_size=batch_size,
								shuffle=shuffle,
								num_workers=workers_num,
								collate_fn=collate_fn,
								pin_memory=False)

	return dataloader

def get_loaders_non_temporal_dataset(dataset_file_path, opt):
	dataloader = get_non_temporal_dataset(	dataset_file_path=dataset_file_path,
											batch_size=opt.batch_size,
											shuffle=True,
											workers_num=opt.workers_num,
											collate_fn=collate_non_fn_temporal_dataset,
											opt=opt)

	return dataloader

def get_dataloader(dataset_file_path, opt):
	dataloader = get_loaders_non_temporal_dataset(	dataset_file_path, opt)

	return dataloader
