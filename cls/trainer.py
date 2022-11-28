import os
import math

import torch
import numpy as np
import transformers
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils import load_model
from dataset import get_tag_scheme
from metrics import compute_metrics

transformers.logging.set_verbosity_error()


class Trainer(object):
    def __init__(self, args, logger, train_dataloader=None, val_dataloader=None, test_dataloader=None):
        self.args = args
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.label_lst = get_tag_scheme()
        self.num_labels = len(self.label_lst)
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        # 모델 초기화
        self.model_class, self.model_name = load_model(args.model_type)
        self.model = self.model_class.from_pretrained(self.model_name, num_labels=6)

        # GPU or CPU
        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Early stop
        self.best_loss = None
        self.patience = 2
        self.counter = 0
        self.early_stopping = False

    def train(self):
        t_total = len(self.train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total
        )

        # Train!
        self.logger.info("===== Running training =====")
        self.logger.info("  Num examples = %d", len(self.train_dataloader.dataset))
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Total train batch size = %d", self.args.batch_size)
        self.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        loss_list_between_log_interval = []
        for epoch_id in range(self.args.num_train_epochs):
            self.model.train()
            epoch_iterator = tqdm(self.train_dataloader, f"[TRAIN] EP:{epoch_id}", total=len(self.train_dataloader))
            for step, batch in enumerate(epoch_iterator):
                global_step = len(self.train_dataloader) * epoch_id + step + 1
                optimizer.zero_grad()
                batch = tuple(t.to(self.device) for t in batch.values())  # GPU or CPU
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                loss_list_between_log_interval.append(loss.item())

                if global_step % self.args.logging_steps == 0:
                    mean_loss = np.mean(loss_list_between_log_interval)
                    self.logger.info(
                        f"EP:{epoch_id} global_step:{global_step} "
                        f"loss:{mean_loss:.4f}"
                    )
                    loss_list_between_log_interval.clear()

            # run evaluate
            val_loss = self.evaluate('val')
            self.early_stop(val_loss)

            # save Transformer
            if self.early_stopping is True:
                break

    def evaluate(self, mode):
        if mode == 'test':
            dataloader = self.test_dataloader
        else:
            dataloader = self.val_dataloader

        # Eval!
        self.logger.info("***** Running evaluating *****")
        self.logger.info("  Num examples = %d", len(dataloader.dataset))
        self.logger.info("  Batch size = %d", self.args.batch_size)

        self.model.eval()
        loss_list = []
        preds = None
        out_label_ids = None
        epoch_iterator = tqdm(dataloader, f"[EVAL]", total=len(dataloader))
        for batch in epoch_iterator:
            with torch.no_grad():
                batch = tuple(t.to(self.device) for t in batch.values())  # GPU or CPU
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }
                outputs = self.model(**inputs)
                eval_loss, logits = outputs[:2]
                loss_list.append(eval_loss.item())

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)

        result = compute_metrics(out_label_ids, preds)
        result['val_loss'] = np.mean(loss_list)

        self.logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            self.logger.info("  %s = %s", key, str(result[key]))

        return result['val_loss']

    def early_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_model()
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stopping = True
        else:
            self.best_loss = val_loss
            self.save_model()
            self.counter = 0

    def save_model(self):
        # Save Transformer checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained Transformer
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        self.logger.info("Saving Transformer checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether Transformer exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.load_model_dir)
            self.model.to(self.device)
            self.logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some Transformer files might be missing...")
