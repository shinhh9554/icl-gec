import os
import argparse
from datetime import datetime

import torch
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from trainer import Trainer
from utils import create_logger, load_tokenizer
from dataset import get_examples, NERDataset, DataCollator


def main(args):
    if args.do_train:
        logger = create_logger(args.model_dir, 'train')
    else:
        logger = create_logger(args.load_model_dir, 'test')

    logger.info("============================")
    for arg, value in sorted(vars(args).items()):
        logger.info("%s: %r", arg, value)
    logger.info("============================")
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.model_type)

    logger.info("loading train dataset")
    examples = get_examples(args)
    random.shuffle(examples)

    dev_pos = (len(examples) // 10) * 7
    test_pos = (len(examples) // 10) * 9

    train_dataset = NERDataset(examples[:dev_pos])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=DataCollator(tokenizer, args.max_seq_length))

    logger.info("loading valid dataset")
    val_dataset = NERDataset(examples[dev_pos:test_pos])
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        collate_fn=DataCollator(tokenizer, args.max_seq_length))

    logger.info("loading test dataset")
    test_dataset = NERDataset(examples[test_pos:])
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size,
        collate_fn=DataCollator(tokenizer, args.max_seq_length))

    trainer = Trainer(args, logger, train_dataloader, val_dataloader, test_dataloader)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    # 모델 hyperparameter 와 저장 위치 등 설정 정보를 선언
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default="kcelectra", type=str)
    parser.add_argument('--data_path', default='data/new_noise_sample.txt', type=str)
    parser.add_argument("--model_dir", default="./models", type=str)
    parser.add_argument("--load_model_dir", default="")
    parser.add_argument("--num_train_epochs", default=10, type=int, required=False)
    parser.add_argument("--batch_size", default=180, type=int, required=False)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--logging_steps', default=50, type=int)
    parser.add_argument("--do_train", action="store_true", default=True)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--gpu_id", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    if args.do_train:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        artifacts_dir = os.path.join(args.model_dir, f"{args.model_type}_{timestamp}")
        os.makedirs(artifacts_dir, exist_ok=True)
        args.model_dir = artifacts_dir

    main(args)
