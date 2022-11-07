import torch
import argparse

from datasets import load_dataset
from transformers import (
	PreTrainedTokenizerFast,
	BartForConditionalGeneration,
	DataCollatorForSeq2Seq,
	Seq2SeqTrainingArguments,
	set_seed,
	Seq2SeqTrainer
)

from gec.utils.metric import Metric
from gec.utils.processor import Preprocessor


def parse_args():
	parser = argparse.ArgumentParser()

	# Model arguments
	parser.add_argument("--model_name_or_path", default='gogamza/kobart-base-v2', type=str)

	# Dataset arguments
	parser.add_argument("--train_file", default='data/gec_train.jsonl', type=str)
	parser.add_argument("--validation_file", default='data/gec_valid.jsonl', type=str)
	parser.add_argument("--max_input_length", default=128, type=int)
	parser.add_argument("--max_target_length", default=128, type=int)
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

	# Load pretrained model and tokenizer
	tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name_or_path)
	model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)

	# Preprocessing Datasets
	preprocessor = Preprocessor(args, tokenizer)

	# Train dataset
	train_dataset = raw_datasets["train"]
	train_dataset = train_dataset.map(
		preprocessor.preprocess_function,
		batched=True,
		num_proc=args.preprocessing_num_workers,
		desc="Running tokenizer on train dataset",
	)

	# Valid dataset
	eval_dataset = raw_datasets["validation"]
	eval_dataset = eval_dataset.map(
		preprocessor.preprocess_function,
		batched=True,
		num_proc=args.preprocessing_num_workers,
		desc="Running tokenizer on validation dataset",
	)

	# Set warmup steps
	total_batch_size = args.per_device_train_batch_size * torch.cuda.device_count()
	n_total_iterations = int(len(train_dataset) / total_batch_size * args.num_train_epochs)
	warmup_steps = int(n_total_iterations * args.warmup_ratio)

	# Train & Eval configs
	training_args = Seq2SeqTrainingArguments(
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
		predict_with_generate=True,
	)

	# Data Collator
	data_collator = DataCollatorForSeq2Seq(tokenizer, model)

	# Metrics
	metrics = Metric(tokenizer)

	# Trainer
	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=metrics.compute_metrics
	)

	# Training
	trainer.train()
	trainer.save_model()

	# Evaluation
	max_length = training_args.generation_max_length
	num_beams = training_args.generation_num_beams
	trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

if __name__ == '__main__':
	main()