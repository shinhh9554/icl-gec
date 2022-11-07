import numpy as np
from datasets import load_metric

class Metric:
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer
		self.metric = load_metric("sacrebleu")

	def compute_metrics(self, eval_preds):
		preds, labels = eval_preds
		if isinstance(preds, tuple):
			preds = preds[0]

		decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

		labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
		decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

		decoded_preds = [pred.strip() for pred in decoded_preds]
		decoded_labels = [[label.strip()] for label in decoded_labels]

		result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
		result = {"bleu": result["score"]}

		return result