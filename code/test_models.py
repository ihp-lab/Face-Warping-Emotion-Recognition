"""
Tests models
"""

import os
import csv
import torch
import backbone
import warnings
import argparse
import numpy as np

from datasets import get_dataloader
from tools.eval import get_eval_metrics

warnings.filterwarnings('always')

def main():
	parser = argparse.ArgumentParser()

	# Names, paths, logs
	parser.add_argument('--model', default='bl', help='bl')
	parser.add_argument('--source_domain', default='Aff-Wild2')
	parser.add_argument('--target_domain', default='SEWA')
	parser.add_argument('--label', default='valence', choices=['valence', 'arousal'])
	parser.add_argument('--test_file', default='test', choices=['test', 'test_a_low', 'test_a_high', 'test_v_low', 'test_v_high'])

	# Data parameters
	parser.add_argument('--workers_num', type=int, default=8, help='number of workers for data loading')
	parser.add_argument('--frames', type=int, default=1)

	# Training and optimization
	parser.add_argument('--batch_size', type=int, default=256, help='size of a mini-batch')

	# Model parameters
	parser.add_argument('--backbone', type=str, default='r21d', choices=['r3d', 'r21d', 'resnet18'])
	parser.add_argument('--dropout_rate', type=float, default=0.1, help='0.1')

	# GPU
	parser.add_argument('--gpu_num', default='cuda:0', help='GPU device')

	opt = parser.parse_args()

	print('Testing models')
	for arg in vars(opt):
		print(arg + ' = ' + str(getattr(opt, arg)))

	test_ccc = test_model(opt)

	print('test ccc: {0:.3f}'.format(test_ccc))

def test_model(opt):
	# Use specific GPU
	device = torch.device(opt.gpu_num)

	# Dataloader
	test_dataset_file_path = os.path.join('../mini_datasets', opt.target_domain, opt.test_file+'.csv')
	test_loader = get_dataloader(test_dataset_file_path, 'test', opt, opt.target_domain)

	# Model and loss function
	if opt.model == 'bl':
		checkpoint = torch.load(os.path.join('checkpoints', opt.model, opt.backbone, opt.source_domain, opt.label, 'model.pth.tar'), map_location=device)
	else:
		checkpoint = torch.load(os.path.join('checkpoints', opt.model, opt.backbone, opt.source_domain, opt.target_domain, opt.label, 'model.pth.tar'), map_location=device)

	model = backbone.Emotion_Recognizer(opt).to(device)
	model.load_state_dict(checkpoint['model'])

	criterion = torch.nn.MSELoss()

	test_loss, test_ccc = test(	test_loader, model,
								criterion, device, opt)

	return test_ccc

def test(test_loader, model, criterion, device, opt):
	model.eval()

	running_loss = 0.

	with torch.no_grad():
		groundtruth = []
		prediction = []

		for i, test_data in enumerate(test_loader):
			images, v_labels, a_labels, _ = test_data

			images = images.to(device)
			v_labels = v_labels.to(device)
			a_labels = a_labels.to(device)

			if opt.label == 'valence':
				labels = v_labels
			else:
				labels = a_labels

			predictions = model(images).view_as(labels)

			loss = criterion(predictions, labels)

			running_loss += loss.item()

			groundtruth.append(labels.tolist())
			prediction.append(predictions.tolist())

		test_loss = running_loss / len(test_loader)
		test_ccc = get_eval_metrics(groundtruth, prediction)

		return test_loss, test_ccc

if __name__ == '__main__':
	main()
