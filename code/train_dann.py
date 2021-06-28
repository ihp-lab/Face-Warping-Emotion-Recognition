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

from test_models import test
from datasets import get_dataloader

from tools.utils import set_seed
from tools.eval import get_eval_metrics
from tools.lr_scheduler import StepwiseLR

from adaptation.dann import DomainAdversarialLoss
from modules.domain_discriminator import DomainDiscriminator

warnings.filterwarnings('always')

def main():
	parser = argparse.ArgumentParser()

	# Seed
	parser.add_argument('--seed', type=int, default=0)

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints/dann', help='relative path to log')
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
	parser.add_argument('--epochs_num', type=int, default=10, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=128, help='size of a mini-batch')
	parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='initial learning rate')

	# Model parameters
	parser.add_argument('--backbone', type=str, default='r21d', choices=['r3d', 'r21d', 'resnet18'])
	parser.add_argument('--dropout_rate', type=float, default=0.1, help='0.1')

	parser.add_argument('--features_dim', type=int, default=128)
	parser.add_argument('--domain_weight', type=float, default=1, choices=[0.3, 1, 3])

	# GPU
	parser.add_argument('--gpu_num', default='cuda:0', help='GPU device')

	opt = parser.parse_args()

	if opt.verbose:
		print('Training and validating models')
		for arg in vars(opt):
			print(arg + ' = ' + str(getattr(opt, arg)))

	set_seed(opt.seed)
	
	half_batch = opt.batch_size // 2
	opt.batch_size = half_batch

	solver = Solver(opt)

	val_ccc, test_ccc = solver.train_model()

	print('val ccc: {0:.3f}'.format(val_ccc))
	print('test ccc: {0:.3f}'.format(test_ccc))

class Solver():
	def __init__(self, opt):
		self.opt = opt
		
		# Use specific GPU
		self.device = torch.device(self.opt.gpu_num)

		# Dataloaders
		val_dataset_file_path = os.path.join('../mini_datasets', self.opt.target_domain, 'val.csv')
		self.val_loader = get_dataloader(val_dataset_file_path, 'val', self.opt, self.opt.target_domain)

		test_dataset_file_path = os.path.join('../mini_datasets', self.opt.target_domain, 'test.csv')
		self.test_loader = get_dataloader(test_dataset_file_path, 'test', self.opt, self.opt.target_domain)

		# Model, optimizer and loss function
		checkpoint = torch.load(os.path.join(	'checkpoints/bl', self.opt.backbone, self.opt.source_domain, self.opt.label, 'model.pth.tar'),
												map_location=self.device)

		self.model = backbone.Emotion_Recognizer(self.opt).to(self.device)
		self.model.load_state_dict(checkpoint['model'])
		for param in self.model.parameters():
			param.requires_grad = True
		
		self.discriminator = DomainDiscriminator(in_feature=self.opt.features_dim, hidden_size=64).to(self.device)
		for param in self.discriminator.parameters():
			param.requires_grad = True

		self.optimizer = torch.optim.Adam(list(	self.model.parameters())+list(self.discriminator.parameters()),
												lr=self.opt.learning_rate,
												weight_decay=self.opt.weight_decay)
		self.lr_sheduler = StepwiseLR(self.optimizer, init_lr=self.opt.learning_rate, gamma=0.0003, decay_rate=0.75)
		self.main_criterion = torch.nn.MSELoss()
		self.domain_criterion = DomainAdversarialLoss(self.discriminator).to(self.device)

	def get_batch_iterator(self):
		source_dataset_file_path = os.path.join('../mini_datasets', self.opt.source_domain, 'train.csv')
		source_loader = get_dataloader(source_dataset_file_path, 'train', self.opt, self.opt.source_domain)

		target_dataset_file_path = os.path.join('../mini_datasets', self.opt.target_domain, 'train.csv')
		target_loader = get_dataloader(target_dataset_file_path, 'train', self.opt, self.opt.target_domain)

		batches = zip(source_loader, target_loader)
		n_batches = min(len(source_loader), len(target_loader))

		return batches, n_batches - 1

	def train_model(self):
		best_ccc = 0.
		best_model = copy.deepcopy(self.model)

		# Train and validate
		for epoch in range(self.opt.epochs_num):
			if self.opt.verbose:
				print('epoch: {}/{}'.format(epoch + 1, self.opt.epochs_num))

			self.batch_iterator, self.n_batches = self.get_batch_iterator()

			label_loss, domain_loss, domain_acc, train_ccc = self.train()
			val_loss, val_ccc = test(	self.val_loader, self.model,
										self.main_criterion, self.device, self.opt)
			if self.opt.verbose:
				print(	'\n',
						'label_loss: {0:.3f}'.format(label_loss),
						'domain_loss: {0:.3f}'.format(domain_loss),
						'domain_acc: {0:.3f}'.format(domain_acc),
						'train_ccc: {0:.3f}'.format(train_ccc),
						'\n',
						'val_loss: {0:.3f}'.format(val_loss),
						'val_ccc: {0:.3f}'.format(val_ccc))

			state = {	'epoch': epoch+1,
						'model': self.model.state_dict(),
						'discriminator': self.discriminator.state_dict(),
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

		test_ccc, test_ccc = test(	self.test_loader, best_model,
									self.main_criterion, self.device, self.opt)

		return best_ccc, test_ccc

	def train(self):
		self.model.train()
		self.domain_criterion.train()
		self.lr_sheduler.step()

		total_label_loss = 0.
		total_domain_loss = 0.
		domain_acc = 0.

		groundtruth = []
		prediction = []

		for i, train_data in enumerate(self.batch_iterator):
			(images_s, v_labels_s, a_labels_s, _), \
			(images_t, _, _, _) = train_data

			images_s = images_s.to(self.device)
			images_t = images_t.to(self.device)

			if self.opt.label == 'valence':
				labels_s = v_labels_s.to(self.device)
			else:
				labels_s = a_labels_s.to(self.device)

			embedding_s = self.model.encoder(images_s)
			embedding_t = self.model.encoder(images_t)

			domain_loss = self.domain_criterion(embedding_s, embedding_t)
			one_batch_domain_acc = self.domain_criterion.domain_discriminator_accuracy
			predictions = self.model.decoder(embedding_s).view_as(labels_s)

			label_loss = self.main_criterion(labels_s, predictions)

			loss = self.opt.domain_weight * domain_loss + label_loss

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			total_label_loss += label_loss.item()
			total_domain_loss += domain_loss.item()
			domain_acc += one_batch_domain_acc

			groundtruth.append(labels_s.tolist())
			prediction.append(predictions.tolist())

			if self.opt.verbose and i > 0 and int(self.n_batches / 10) > 0 and i % (int(self.n_batches / 10)) == 0:
				print('.', flush=True, end='')
			
			if i >= self.n_batches - 1:
				break

		train_ccc = get_eval_metrics(groundtruth, prediction)

		label_loss = total_label_loss / self.n_batches
		domain_loss = total_domain_loss / self.n_batches
		domain_acc = domain_acc / self.n_batches

		return label_loss, domain_loss, domain_acc, train_ccc

if __name__ == '__main__':
	main()
