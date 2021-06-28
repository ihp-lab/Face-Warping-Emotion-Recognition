import torch
from torch import nn
import numpy as np

class NCELoss(nn.Module):
	def __init__(self, nce_T):
	#def __init__(self, nce_T, device):
		super().__init__()

		self.nce_T = nce_T
		#self.device = device

	def forward(self, c1_emb, c2_emb):
		batch_size = c1_emb.shape[0]

		pos_emb = c1_emb
		neg_emb = torch.cat([c2_emb, c1_emb], dim=0)

		similarity = torch.matmul(pos_emb, neg_emb.transpose(1,0)) / self.nce_T
		similarity = torch.exp(similarity)

		pos_mask = torch.cat([torch.eye(batch_size).cuda(), torch.zeros((batch_size, batch_size)).cuda()], dim=1)
		identity_mask = torch.cat([torch.zeros((batch_size, batch_size)).cuda(), torch.eye(batch_size).cuda()], dim=1)
		neg_mask = torch.ones(similarity.shape).cuda() - pos_mask - identity_mask

		pos_sim = pos_mask * similarity
		neg_sim = neg_mask * similarity

		out = pos_sim.sum(dim=1) / neg_sim.sum(dim=1)

		return torch.log(out) * -1.0
