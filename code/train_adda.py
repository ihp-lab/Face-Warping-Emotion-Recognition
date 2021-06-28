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

from modules.domain_discriminator import DomainDiscriminator

warnings.filterwarnings('always')

def main():
	parser = argparse.ArgumentParser()

	# Seed
	parser.add_argument('--seed', type=int, default=0)

	# Names, paths, logs
	parser.add_argument('--logger_path', default='checkpoints/adda', help='relative path to log')
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

		self.src_enc = copy.deepcopy(self.model.encoder).to(self.device)
		for param in self.src_enc.parameters():
			param.requires_grad = False

		self.src_clf = copy.deepcopy(self.model.decoder).to(self.device)
		for param in self.src_clf.parameters():
			param.requires_grad = False

		self.tgt_enc = copy.deepcopy(self.model.encoder).to(self.device)
		for param in self.tgt_enc.parameters():
			param.requires_grad = True

		self.discriminator = DomainDiscriminator(in_feature=self.opt.features_dim, hidden_size=64).to(self.device)
		for param in self.discriminator.parameters():
			param.requires_grad = True

		self.opt_tgt = torch.optim.Adam(self.tgt_enc.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
		self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)

		self.main_criterion = torch.nn.MSELoss()
		self.critic_criterion = torch.nn.BCELoss()

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

			domain_acc = self.train()

			self.model.encoder = copy.deepcopy(self.tgt_enc).to(self.device)
			self.model.decoder = copy.deepcopy(self.src_clf).to(self.device)

			val_loss, val_ccc = test(	self.val_loader, self.model,
										self.main_criterion, self.device, self.opt)
			if self.opt.verbose:
				print(	'\n',
						'domain_acc: {0:.3f}'.format(domain_acc),
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
		self.tgt_enc.train()
		self.discriminator.train()

		total_acc = 0.

		for i, train_data in enumerate(self.batch_iterator):
			(images_s, v_labels_s, a_labels_s, _), \
			(images_t, _, _, _) = train_data

			images_s = images_s.to(self.device)
			images_t = images_t.to(self.device)

			if self.opt.label == 'valence':
				labels_s = v_labels_s.to(self.device)
			else:
				labels_s = a_labels_s.to(self.device)

			##########################
			# 1. train discriminator #
			##########################

			feat_src = self.src_enc(images_s)
			feat_tgt = self.tgt_enc(images_t)
			feat_concat = torch.cat((feat_src, feat_tgt), 0)
			pred_concat = self.discriminator(feat_concat).view(-1)

			label_src = (torch.ones(feat_src.size(0)).float()).to(self.device)
			label_tgt = (torch.zeros(feat_tgt.size(0)).float()).to(self.device)
			label_concat = torch.cat((label_src, label_tgt), 0)

			loss_critic = self.critic_criterion(pred_concat, label_concat)

			self.opt_disc.zero_grad()
			loss_critic.backward()
			self.opt_disc.step()

			pred_cls = np.array((pred_concat > 0.5).tolist()).astype(int)
			label_concat = np.array(label_concat.tolist())

			acc = (pred_cls == label_concat).astype(float).mean()
			total_acc += acc

			###########################
			# 2. train target encoder #
			###########################

			feat_tgt = self.tgt_enc(images_t)
			pred_tgt = self.discriminator(feat_tgt).view(-1)
			label_tgt = (torch.ones(feat_tgt.size(0)).float()).to(self.device)

			loss_tgt = self.critic_criterion(pred_tgt, label_tgt)

			self.opt_tgt.zero_grad()
			loss_tgt.backward()
			self.opt_tgt.step()

			if self.opt.verbose and i > 0 and int(self.n_batches / 10) > 0 and i % (int(self.n_batches / 10)) == 0:
				print('.', flush=True, end='')
			
			if i >= self.n_batches - 1:
				break

		domain_acc = total_acc / self.n_batches

		return domain_acc

if __name__ == '__main__':
	main()
