import torch
import numpy as np
from datasets import load_metric

class Metric:
	def __init__(self, args, model, label_list):
		self.args = args
		self.model = model
		self.label_list = label_list
		self.metric = load_metric("seqeval")

	def compute_metrics(self, eval_preds):
		predictions, labels = eval_preds
		if self.args.crf:
			masks = torch.tensor(labels != -100).cuda()
			masks[:, 0] = True
			predictions = torch.tensor(predictions).cuda()
			predictions = self.model.decode(predictions, masks)
		else:
			predictions = np.argmax(predictions, axis=2)

		# Remove ignored index (special tokens)
		true_predictions = [
			[self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]
		true_labels = [
			[self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]

		results = self.metric.compute(predictions=true_predictions, references=true_labels)

		return {
			"precision": results["overall_precision"],
			"recall": results["overall_recall"],
			"f1": results["overall_f1"],
			"accuracy": results["overall_accuracy"],
		}
