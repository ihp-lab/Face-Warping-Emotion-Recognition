import torch
import random
import numpy as np
import torch.nn.functional as F

def set_seed(seed):
	# Reproducibility
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False

	random.seed(seed)
	np.random.seed(seed)
