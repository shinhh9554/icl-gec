import argparse

import torch
from datasets import load_dataset
from transformers import (
	ElectraConfig,
	ElectraForTokenClassification,
	TrainingArguments,
	DataCollatorForTokenClassification,
	set_seed,
	Trainer
)

from metric import Metric
from processor import Preprocessor
from tokenization_kocharelectra import KoCharElectraTokenizer

def parse_args():
	parser = argparse.ArgumentParser()

	# Model arguments
	parser.add_argument("--model_name_or_path", default='gogamza/kobart-base-v2', type=str)

	# Dataset arguments
	parser.add_argument("--train_file", default='data/gec_train.jsonl', type=str)
	parser.add_argument("--validation_file", default='data/gec_valid.jsonl', type=str)
	parser.add_argument("--max_seq_length", default=128, type=int)
	parser.add_argument("--preprocessing_num_workers", default=1, type=int)

	# Training arguments
	parser.add_argument("--seed", default=42, type=int)
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
	parser.add_argument("--metric_for_best_model", default='bleu', type=str)

	args = parser.parse_args()

	return args


def main():
	# Parse the arguments
	args = parse_args()

	# Seed
	set_seed(args.seed)

	# Loading Datasets
	data_files = {"train": args.train_file, "validation": args.validation_file}
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
	model = ElectraForTokenClassification.from_pretrained(args.model_name_or_path, config=config)

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

	predict_dataset = raw_datasets["test"]
	predict_dataset = predict_dataset.map(
		preprocessor.tokenize_and_align_labels,
		batched=True,
		num_proc=args.preprocessing_num_workers,
		desc="Running tokenizer on prediction dataset",
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
	)

	# Data collator
	data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

	# Metrics
	metrics = Metric(label_list)

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		data_collator=data_collator,
		compute_metrics=metrics.compute_metrics,
	)

	# Training
	trainer.train()
	trainer.save_model()

	# Evaluation
	trainer.evaluate()

	# Test
	trainer.evaluate(predict_dataset)

if __name__ == '__main__':
	main()








