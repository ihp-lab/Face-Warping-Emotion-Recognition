"""
Trains and validates models
"""

import os
import csv
import PIL
import copy
import torch
import random
import pandas
import backbone
import warnings
import argparse
import numpy as np

from PIL import Image
from torchvision import transforms

from loss import NCELoss
from datasets import get_dataloader

from tools.utils import set_seed

warnings.filterwarnings('always')
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4'

def main():
	parser = argparse.ArgumentParser()

	# Seed
	parser.add_argument('--seed', type=int, default=0)

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints_v3', help='relative path to log')
	parser.add_argument('--source_domain', default='Aff-Wild2_v3')
	parser.add_argument('--verbose', type=bool, default=True, help='True or False')
	parser.add_argument('--save_checkpoint', type=bool, default=True, help='True or False')
	parser.add_argument('--save_model', type=bool, default=True, help='True or False')

	# Data parameters
	parser.add_argument('--workers_num', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('--period', type=int, default=50)
	parser.add_argument('--frames', type=int, default=16)

	# Training and optimization
	parser.add_argument('--epochs_num', type=int, default=25, help='number of training epochs')
	#parser.add_argument('--batch_size', type=int, default=120, help='size of a mini-batch')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, help='initial momentum')

	# Model parameters
	parser.add_argument('--backbone', type=str, default='r21d', choices=['r3d', 'r21d'])
	parser.add_argument('--dropout_rate', type=float, default=0.1, help='0.1')
	parser.add_argument('--nce_T', type=float, default=0.1, help='0.1')

	# GPU
	parser.add_argument('--gpu_num', default='cuda:0', help='GPU device')

	opt = parser.parse_args()

	if opt.verbose:
		print('Training and validating models')
		for arg in vars(opt):
			print(arg + ' = ' + str(getattr(opt, arg)))

	set_seed(opt.seed)

	solver = Solver(opt)
	solver.train_model()

class Solver():
	def __init__(self, opt):
		self.opt = opt
		
		# Use specific GPU
		#self.device = torch.device(self.opt.gpu_num)

		# Dataloaders
		self.root_path = '../datasets/Contrastive/'+self.opt.source_domain
		self.train_video_list = sorted(os.listdir(self.root_path+'/train'))
		self.val_video_list = sorted(os.listdir(self.root_path+'/val'))
		self.csv_dict = {}

		transform_list = [	transforms.Resize(112),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225])]
		self.transform = transforms.Compose(transform_list)

		# Model, optimizer and loss function
		self.model = backbone.Encoder(self.opt)
		#self.model = backbone.Encoder(self.opt).to(self.device)
		self.model = torch.nn.DataParallel(self.model, device_ids=[0,1,2,3]).cuda()
		for param in self.model.parameters():
			param.requires_grad = True

		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.learning_rate, momentum=opt.momentum)
		self.criterion = NCELoss(nce_T=self.opt.nce_T)
		#self.criterion = NCELoss(nce_T=self.opt.nce_T, device=self.device)

	def train_model(self):
		best_acc = 0.
		best_loss = 10.

		# Train and validate
		for epoch in range(self.opt.epochs_num):
			if self.opt.verbose:
				print('epoch: {}/{}'.format(epoch+1, self.opt.epochs_num))

			train_loss = self.train()
			val_loss, val_acc = self.val()

			if self.opt.verbose:
				print(	'train_loss: {0:.3f}'.format(train_loss),
						'val_loss: {0:.3f}'.format(val_loss),
						'val_acc: {0:.3f}'.format(val_acc))

			state = {	'epoch': epoch+1,
						'model': self.model.state_dict(),
						'opt': self.opt}

			os.makedirs(os.path.join(self.opt.logger_path, self.opt.backbone, self.opt.source_domain[:-3]), exist_ok=True)

			if self.opt.save_checkpoint:
				model_file_name = os.path.join(self.opt.logger_path, self.opt.backbone, self.opt.source_domain[:-3], 'checkpoint.pth.tar')
				torch.save(state, model_file_name)

			if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
				best_acc = val_acc
				best_loss = val_loss

				if self.opt.save_model:
					model_file_name = os.path.join(self.opt.logger_path, self.opt.backbone, self.opt.source_domain[:-3], 'model.pth.tar')
					torch.save(state, model_file_name)

		return
	
	def get_item(self, real_path, fake_path):
		length = len(real_path)
		idx = np.random.randint(length)

		real_image_path = real_path[idx]
		fake_image_path = fake_path[idx]

		real_image_list = []
		fake_image_list = []

		for i in range(self.opt.frames):
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

	def get_samples(self, load_type):
		if load_type == 'train':
			video_list = self.train_video_list
		else:
			video_list = self.val_video_list
	
		real_images_batch_list = []
		fake_images_batch_list = []
		random.shuffle(video_list)
		for video in video_list:
			if not video in self.csv_dict:
				dataset_file_path = self.root_path+'/'+load_type+'/'+video
				dataset_file = pandas.read_csv(dataset_file_path)

				real_path = dataset_file['real_path'].tolist()
				fake_path = dataset_file['fake_path'].tolist()

				self.csv_dict[video] = [real_path, fake_path]

			real_images, fake_images = self.get_item(self.csv_dict[video][0], self.csv_dict[video][1])
			
			real_images = torch.unsqueeze(real_images, 0)
			real_images_batch_list.append(real_images)
			fake_images = torch.unsqueeze(fake_images, 0)
			fake_images_batch_list.append(fake_images)

		real_images_batch = torch.cat(real_images_batch_list, dim=0)
		fake_images_batch = torch.cat(fake_images_batch_list, dim=0)

		return real_images_batch, fake_images_batch

	def train(self):
		self.model.train()

		running_loss = 0.

		for i in range(self.opt.period):
			real_images, fake_images = self.get_samples('train')

			real_images = real_images.cuda()
			fake_images = fake_images.cuda()
			#real_images = real_images.to(self.device)
			#fake_images = fake_images.to(self.device)

			real_embeddings = self.model(real_images)
			fake_embeddings = self.model(fake_images)

			loss = self.criterion(real_embeddings, fake_embeddings)
			loss = loss.mean()

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			running_loss += loss.item()

			if self.opt.verbose and i > 0 and int(self.opt.period / 10) > 0 and i % (int(self.opt.period / 10)) == 0:
				print('.', flush=True, end='')

		train_loss = running_loss / self.opt.period

		return train_loss
	
	def val(self):
		self.model.eval()
		running_loss = 0.
		running_acc = 0.

		with torch.no_grad():
			for i in range(self.opt.period):
				real_images, fake_images = self.get_samples('val')

				real_images = real_images.cuda()
				fake_images = fake_images.cuda()
				#real_images = real_images.to(self.device)
				#fake_images = fake_images.to(self.device)

				real_embeddings = self.model(real_images)
				fake_embeddings = self.model(fake_images)

				loss = self.criterion(real_embeddings, fake_embeddings)
				loss = loss.mean()

				running_loss += loss.item()
				acc = self.calculate_acc(real_embeddings, fake_embeddings)
				running_acc += acc

			running_loss = running_loss / self.opt.period
			running_acc = running_acc / self.opt.period

		return running_loss, running_acc

	def calculate_acc(self, c1_emb, c2_emb):
		batch_size = c1_emb.shape[0]

		pos_emb = c1_emb
		neg_emb = torch.cat([c2_emb, c1_emb], dim=0)

		similarity = torch.matmul(pos_emb, neg_emb.transpose(1,0))

		pos_mask = torch.cat([torch.eye(batch_size).cuda(), torch.zeros((batch_size, batch_size)).cuda()], dim=1)
		identity_mask = torch.cat([torch.zeros((batch_size, batch_size)).cuda(), torch.eye(batch_size).cuda()], dim=1)
		neg_mask = torch.ones(similarity.shape).cuda() - pos_mask - identity_mask

		#pos_mask = torch.cat([torch.eye(batch_size).to(self.device), torch.zeros((batch_size, batch_size)).to(self.device)], dim=1)
		#identity_mask = torch.cat([torch.zeros((batch_size, batch_size)).to(self.device), torch.eye(batch_size).to(self.device)], dim=1)
		#neg_mask = torch.ones(similarity.shape).to(self.device) - pos_mask - identity_mask

		pos_sim = (pos_mask * similarity).sum(dim=1)
		neg_sim = neg_mask * similarity

		count = 0
		for i in range(batch_size):
			pos_cur = pos_sim[i].item()
			neg_cur = neg_sim[i,:].cpu().detach().numpy()
			wrong_count = np.count_nonzero(neg_cur > pos_cur)
			if wrong_count == 0:
				count += 1

		return 1.0 * count / batch_size

if __name__ == '__main__':
	main()
