import os
from typing import Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm
from torch.utils.data import Dataset


@dataclass
class NERExample:
    error: str
    correct: str
    label: Optional[int] = None


def get_tag_scheme():
    return [0, 1, 2, 3, 4, 5]


def get_examples(args):
    examples = []
    for line in open(args.data_path, 'r', encoding='utf-8').readlines():
        error, correct, label = line.strip('\n').split('\t')
        examples.append(NERExample(error=error, correct=correct, label=int(label)))

    return examples


class NERDataset(Dataset):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples

    def __getitem__(self, idx):
        error = self.examples[idx].error
        correct = self.examples[idx].correct
        label = self.examples[idx].label

        return {
            'error': error,
            'correct': correct,
            'label': label
        }

    def __len__(self):
        return len(self.examples)


class DataCollator:

    def __init__(self, tokenizer, max_length):
        self.padding = 'max_length'
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_pad_token_id = -100

    def __call__(self, samples):
        errors = [s['error'] for s in samples]
        corrects = [s['correct'] for s in samples]
        labels = [s['label'] for s in samples]

        encoding = self.tokenizer(
            errors,
            corrects,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }

        return return_value
