import os
import argparse

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
	ElectraConfig,
	TrainingArguments,
	DataCollatorForTokenClassification,
	set_seed,
	Trainer
)

from metric import Metric
from processor import Preprocessor
from utils import MODEL_FOR_TOKEN_CLASSIFICATION
from tokenization_kocharelectra import KoCharElectraTokenizer

def parse_args():
	parser = argparse.ArgumentParser()

	# Model arguments
	parser.add_argument("--model_type", default='base', type=str)
	parser.add_argument("--model_name_or_path", default='monologg/kocharelectra-base-discriminator', type=str)

	# Dataset arguments
	parser.add_argument("--train_file", default='data/ged_train.jsonl', type=str)
	parser.add_argument("--validation_file", default='data/ged_valid.jsonl', type=str)
	parser.add_argument("--predict_file", default='data/ged_test.jsonl', type=str)
	parser.add_argument("--max_seq_length", default=128, type=int)
	parser.add_argument("--preprocessing_num_workers", default=1, type=int)

	# Training arguments
	parser.add_argument("--crf", default=True, type=bool)
	parser.add_argument("--seed", default=42, type=int)
	parser.add_argument("--do_train", default=True, type=bool)
	parser.add_argument("--do_evaluate", default=True, type=bool)
	parser.add_argument("--do_predict", default=True, type=bool)
	parser.add_argument("--output_dir", default='output', type=str)
	parser.add_argument("--num_train_epochs", default=5.0, type=float)
	parser.add_argument("--per_device_train_batch_size", default=96, type=int)
	parser.add_argument("--per_device_eval_batch_size", default=96, type=int)
	parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
	parser.add_argument("--learning_rate", default=5e-5, type=float)
	parser.add_argument("--weight_decay", default=0.001, type=float)
	parser.add_argument("--warmup_ratio", default=0.05, type=float)
	parser.add_argument("--fp16", default=True, type=bool)
	parser.add_argument("--evaluation_strategy", default='epoch', type=str)
	parser.add_argument("--logging_steps", default=50, type=int)
	parser.add_argument("--save_strategy", default='epoch', type=str)
	parser.add_argument("--load_best_model_at_end", default=True, type=bool)
	parser.add_argument("--metric_for_best_model", default='f1', type=str)

	args = parser.parse_args()

	return args


def main():
	# Parse the arguments
	args = parse_args()

	# Seed
	set_seed(args.seed)

	# Loading Datasets
	data_files = {
		"train": args.train_file,
		"validation": args.validation_file,
		"test": args.predict_file,
	}
	raw_datasets = load_dataset('json', data_files=data_files)

	# Get unique labels
	def get_label_list(labels):
		unique_labels = set()
		for label in labels:
			unique_labels = unique_labels | set(label.split())
		label_list = list(unique_labels)
		label_list.sort()
		return label_list

	label_list = get_label_list(raw_datasets["train"]['tag'])
	label_to_id = {l: i for i, l in enumerate(label_list)}
	num_labels = len(label_list)

	# Load pretrained model and tokenizer
	config = ElectraConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
	tokenizer = KoCharElectraTokenizer.from_pretrained(args.model_name_or_path)
	model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(args.model_name_or_path, config=config)

	# Set the correspondences label/ID inside the model config
	model.config.label2id = {l: i for i, l in enumerate(label_list)}
	model.config.id2label = {i: l for i, l in enumerate(label_list)}

	# Preprocessing the datasets.
	preprocessor = Preprocessor(args, tokenizer, label_to_id)

	train_dataset = raw_datasets["train"]
	train_dataset = train_dataset.map(
		preprocessor.tokenize_and_align_labels,
		batched=True,
		num_proc=args.preprocessing_num_workers,
		desc="Running tokenizer on train dataset",
	)

	eval_dataset = raw_datasets["validation"]
	eval_dataset = eval_dataset.map(
		preprocessor.tokenize_and_align_labels,
		batched=True,
		num_proc=args.preprocessing_num_workers,
		desc="Running tokenizer on validation dataset",
	)

	test_dataset = raw_datasets["test"]
	test_dataset = test_dataset.map(
		preprocessor.tokenize_and_align_labels,
		batched=True,
		num_proc=args.preprocessing_num_workers,
		desc="Running tokenizer on validation dataset",
	)

	# Set warmup steps
	total_batch_size = args.per_device_train_batch_size * torch.cuda.device_count()
	n_total_iterations = int(len(train_dataset) / total_batch_size * args.num_train_epochs)
	warmup_steps = int(n_total_iterations * args.warmup_ratio)

	# Train & Eval configs
	training_args = TrainingArguments(
		output_dir=args.output_dir,
		num_train_epochs=args.num_train_epochs,
		per_device_train_batch_size=args.per_device_train_batch_size,
		per_device_eval_batch_size=args.per_device_eval_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		warmup_ratio=args.warmup_ratio,
		warmup_steps=warmup_steps,
		fp16=args.fp16,
		evaluation_strategy=args.evaluation_strategy,
		logging_steps=args.logging_steps,
		save_strategy=args.save_strategy,
		load_best_model_at_end=args.load_best_model_at_end,
		metric_for_best_model=args.metric_for_best_model,
		do_predict=args.do_predict
	)

	# Data collator
	data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

	# Metrics
	metrics = Metric(args, model, label_list)

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=data_collator,
		compute_metrics=metrics.compute_metrics,
	)

	# Training
	if training_args.do_train:
		trainer.train()
		trainer.save_model()  # Saves the tokenizer too for easy upload

	# Evaluation
	if training_args.do_eval:
		trainer.evaluate()

	# Predict
	if training_args.do_predict:
		predictions, labels, metrics = trainer.predict(test_dataset)
		if args.crf:
			masks = torch.tensor(labels != -100).cuda()
			masks[:, 0] = True
			predictions = torch.tensor(predictions).cuda()
			predictions = model.decode(predictions, masks)
		else:
			predictions = np.argmax(predictions, axis=2)

		# Remove ignored index (special tokens)
		true_predictions = [
			[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]

		# Save predictions
		output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
		if trainer.is_world_process_zero():
			with open(output_test_predictions_file, "w") as writer:
				for prediction in true_predictions:
					writer.write(" ".join(prediction) + "\n")


if __name__ == '__main__':
	main()








