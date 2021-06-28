import itertools
import numpy as np

def concordance_correlation_coefficient(y_true, y_pred,
										sample_weight=None,
										multioutput='uniform_average'):
	"""Concordance correlation coefficient. 
	Parameters
	----------
	y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
		Ground truth (correct) target values.
	y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
		Estimated target values.
	Returns
	-------
	loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
	between the true and the predicted values.
	Examples
	--------
	>>> y_true = [3, -0.5, 2, 7]
	>>> y_pred = [2.5, 0.0, 2, 8]
	>>> concordance_correlation_coefficient(y_true, y_pred)
	0.97678916827853024
	"""

	cor=np.corrcoef(y_true,y_pred)[0][1]
	
	mean_true=np.mean(y_true)
	mean_pred=np.mean(y_pred)
	
	var_true=np.var(y_true)
	var_pred=np.var(y_pred)
	
	sd_true=np.std(y_true)
	sd_pred=np.std(y_pred)
	
	numerator=2*cor*sd_true*sd_pred
	
	denominator=var_true+var_pred+(mean_true-mean_pred)**2

	return numerator/denominator

def get_eval_metrics(groundtruth, prediction):
	groundtruth = list(itertools.chain.from_iterable(groundtruth))
	prediction = list(itertools.chain.from_iterable(prediction))

	groundtruth = np.array(groundtruth)
	prediction = np.array(prediction)

	ccc = concordance_correlation_coefficient(groundtruth, prediction)

	return ccc
