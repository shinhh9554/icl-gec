import os
import sys
import logging
from typing import Optional
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    ElectraConfig,
    ElectraForTokenClassification,
    TrainingArguments,
    IntervalStrategy,
    HfArgumentParser,
    DataCollatorForTokenClassification,
    set_seed
)

from tokenization_kocharelectra import KoCharElectraTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default='../output',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    with_inference: bool = field(
        default=True,
        metadata={"help": "do train then inference"},
    )
    project_name: str = field(
        # PR 하실때는 None 으로 바꿔서 올려주세요! 얘의 목적은 wandb project name 설정을 위함입니다.
        default=None,
        metadata={"help": "wandb project name"},
    )
    run_name: Optional[str] = field(
        default='exp',
        metadata={"help": "wandb run name"},
    )
    retrieval_run_name: Optional[str] = field(
        default='bert-base',  # 이게 제일 성능이 좋았음
        metadata={"help": "retrieval encoder model folder name"},
    )
    evaluation_strategy: IntervalStrategy = field(
        default='steps',
        metadata={"help": "The evaluation strategy to use."},
    )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    per_device_retrieval_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for retrieval_evaluation."}
    )
    per_device_retrieval_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for retrieval_evaluation."}
    )
    num_train_epochs: float = field(default=10.0, metadata={"help": "Total number of training epochs to perform."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    retrieval_learning_rate: float = field(default=1e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default='exact_match', metadata={"help": "The metric to use to compare two different models."}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit"},
    )
    save_total_limit: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    eval_steps: int = field(default=100, metadata={"help": "Run an evaluation every X steps."})
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."})
    early_stopping_patience: int = field(default=5, metadata={"help": "early_stopping_patience."})
    fold: bool = field(default=False, metadata={"help": "SET 5-Fold or not"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-small",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    finetuned_mrc_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    additional_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Attach additional layer to end of model"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default="data/ged_train.jsonl",
        metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default="data/ged_valid.jsonl",
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default="data/ged_test.jsonl",
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    dataset_name: Optional[str] = field(
        default="basic",
        metadata={"help": "The name of the dataset to use. ['basic', 'concat', 'preprocess']"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )


def main():
    # Load Argument
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Set logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Get the datasets
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    raw_datasets = load_dataset('json', data_files=data_files)

    # Set column names
    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    text_column_name = column_names[0]
    label_column_name = column_names[1]

    # Get unique labels
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label.split())
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = ElectraConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
    )

    tokenizer = KoCharElectraTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    model = ElectraForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Preprocessing the datasets.
    padding = "max_length"
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            truncation=True
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            label_ids = [label_to_id['O']] + [label_to_id[tag] for tag in label.split()] + [label_to_id['O']]
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on train dataset",
        )

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        predict_dataset = predict_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=padding,
        max_length=data_args.max_seq_length
    )

if __name__ == '__main__':
    main()








