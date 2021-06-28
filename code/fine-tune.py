"""
Trains and validates models
"""

import os
import csv
import copy
import torch
import backbone
import warnings
import argparse
import numpy as np

from datasets import get_dataloader

from tools.utils import set_seed
from tools.eval import get_eval_metrics
from tools.lr_scheduler import StepwiseLR

warnings.filterwarnings('always')

def main():
	parser = argparse.ArgumentParser()

	# Seed
	parser.add_argument('--seed', type=int, default=0)

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints/model', help='relative path to log')
	parser.add_argument('--source_domain', default='Aff-Wild2', choices=['Aff-Wild2', 'SEWA'])
	parser.add_argument('--target_domain', default='SEWA', choices=['Aff-Wild2', 'SEWA'])
	parser.add_argument('--label', default='valence', choices=['valence', 'arousal'])
	parser.add_argument('--verbose', type=bool, default=True, help='True or False')
	parser.add_argument('--save_checkpoint', type=bool, default=True, help='True or False')
	parser.add_argument('--save_model', type=bool, default=True, help='True or False')

	# Data parameters
	parser.add_argument('--workers_num', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('--frames', type=int, default=1)

	# Training and optimization
	parser.add_argument('--epochs_num', type=int, default=25, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=64, help='size of a mini-batch')
	parser.add_argument('--lr_mult', type=float, default=0.1, help='lr_mult for encoder')
	parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='initial learning rate')

	# Model parameters
	parser.add_argument('--backbone', type=str, default='r21d', choices=['r3d', 'r21d'])
	parser.add_argument('--dropout_rate', type=float, default=0.1, help='0.1')

	# GPU
	parser.add_argument('--gpu_num', default='cuda:0', help='GPU device')

	opt = parser.parse_args()

	if opt.verbose:
		print('Training and validating models')
		for arg in vars(opt):
			print(arg + ' = ' + str(getattr(opt, arg)))

	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_num[-1]

	set_seed(opt.seed)

	solver = Solver(opt)

	val_ccc, test_ccc = solver.train_model()

	print('val ccc: {0:.3f}'.format(val_ccc))
	print('test ccc: {0:.3f}'.format(test_ccc))

class Solver():
	def __init__(self, opt):
		self.opt = opt

		# Dataloaders
		train_dataset_file_path = os.path.join('../mini_datasets', opt.source_domain, 'train.csv')
		self.train_loader = get_dataloader(train_dataset_file_path, 'train', opt)

		val_dataset_file_path = os.path.join('../mini_datasets', self.opt.target_domain, 'val.csv')
		self.val_loader = get_dataloader(val_dataset_file_path, 'val', self.opt, self.opt.target_domain)

		test_dataset_file_path = os.path.join('../mini_datasets', self.opt.target_domain, 'test.csv')
		self.test_loader = get_dataloader(test_dataset_file_path, 'test', self.opt, self.opt.target_domain)

		# Model, optimizer and loss function
		self.model = backbone.Emotion_Recognizer_Fine_Tune(self.opt).cuda()
		self.model.encoder = torch.nn.DataParallel(self.model.encoder, device_ids=[0]).cuda()

		checkpoint = torch.load(os.path.join('../contrastive/checkpoints_v3', self.opt.backbone, self.opt.source_domain, 'model.pth.tar'), map_location='cuda')
		self.model.encoder.load_state_dict(checkpoint['model'])

		for param in self.model.encoder.parameters():
			param.requires_grad = True
		for param in self.model.decoder.parameters():
			param.requires_grad = True

		self.optimizer = torch.optim.Adam(self.model.get_parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
		self.lr_sheduler = StepwiseLR(self.optimizer, init_lr=self.opt.learning_rate, gamma=0.0003, decay_rate=0.75)
		self.criterion = torch.nn.MSELoss()

	def train_model(self):
		best_ccc = 0.
		best_model = copy.deepcopy(self.model)

		# Train and validate
		for epoch in range(self.opt.epochs_num):
			if self.opt.verbose:
				print('epoch: {}/{}'.format(epoch + 1, self.opt.epochs_num))

			train_loss, train_ccc = self.train()
			val_loss, val_ccc = self.test(self.val_loader, self.model)
			if self.opt.verbose:
				print(	'train_loss: {0:.3f}'.format(train_loss),
						'train_ccc: {0:.3f}'.format(train_ccc),
						'val_loss: {0:.3f}'.format(val_loss),
						'val_ccc: {0:.3f}'.format(val_ccc))

			state = {	'epoch': epoch+1,
						'model': self.model.state_dict(),
						'opt': self.opt}
			os.makedirs(os.path.join(self.opt.logger_path, self.opt.backbone, self.opt.source_domain, self.opt.target_domain, self.opt.label), exist_ok=True)

			if self.opt.save_checkpoint:
				model_file_name = os.path.join(self.opt.logger_path, self.opt.backbone, self.opt.source_domain, self.opt.target_domain, self.opt.label, 'checkpoint.pth.tar')
				torch.save(state, model_file_name)

			if val_ccc > best_ccc:
				best_ccc = val_ccc
				best_model = copy.deepcopy(self.model)

				if self.opt.save_model:
					model_file_name = os.path.join(self.opt.logger_path, self.opt.backbone, self.opt.source_domain, self.opt.target_domain, self.opt.label, 'model.pth.tar')
					torch.save(state, model_file_name)

		test_loss, test_ccc = self.test(self.test_loader, best_model)

		return best_ccc, test_ccc

	def train(self):
		self.model.train()
		self.lr_sheduler.step()

		running_loss = 0.

		groundtruth = []
		prediction = []

		for i, train_data in enumerate(self.train_loader):
			images, v_labels, a_labels, _ = train_data

			images = images.cuda()
			v_labels = v_labels.cuda()
			a_labels = a_labels.cuda()

			if self.opt.label == 'valence':
				labels = v_labels
			else:
				labels = a_labels

			predictions = self.model(images).view_as(labels)

			loss = self.criterion(labels, predictions)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			running_loss += loss.item()

			groundtruth.append(labels.tolist())
			prediction.append(predictions.tolist())

			if self.opt.verbose and i > 0 and int(len(self.train_loader) / 10) > 0 and i % (int(len(self.train_loader) / 10)) == 0:
				print('.', flush=True, end='')

		train_loss = running_loss / len(self.train_loader)
		train_ccc = get_eval_metrics(groundtruth, prediction)

		return train_loss, train_ccc

	def test(self, test_loader, model):
		model.eval()

		running_loss = 0.

		with torch.no_grad():
			groundtruth = []
			prediction = []

			for i, test_data in enumerate(test_loader):
				images, v_labels, a_labels, _ = test_data

				images = images.cuda()
				v_labels = v_labels.cuda()
				a_labels = a_labels.cuda()

				if self.opt.label == 'valence':
					labels = v_labels
				else:
					labels = a_labels

				predictions = model(images).view_as(labels)

				loss = self.criterion(predictions, labels)

				running_loss += loss.item()

				groundtruth.append(labels.tolist())
				prediction.append(predictions.tolist())

			test_loss = running_loss / len(test_loader)
			test_ccc = get_eval_metrics(groundtruth, prediction)

			return test_loss, test_ccc

if __name__ == '__main__':
	main()
