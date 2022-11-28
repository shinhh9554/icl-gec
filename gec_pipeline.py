import os
import re
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from konlpy.tag import Mecab

from ged.tokenization_kocharelectra import KoCharElectraTokenizer
from ged.utils import MODEL_FOR_TOKEN_CLASSIFICATION

from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

mecab = Mecab()
random.seed(13)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--ged_model_type", default='base', type=str)
    parser.add_argument("--ged_save_path", default='ged/output/checkpoint-3808', type=str)
    parser.add_argument("--gec_save_path", default='gec/output/checkpoint-4188', type=str)
    parser.add_argument("--cls_save_path", default='output/checkpoint-3808', type=str)

    # Dataset arguments
    parser.add_argument("--error", default='너가 굴도 좋아하나', type=str)
    parser.add_argument("--correct", default='네가 굴도 좋아하니?', type=str)

    args = parser.parse_args()

    return args


def gec_inference(model, tokenizer, device, error):
    input_ids = tokenizer.encode(error, return_tensors="pt")
    input_ids = torch.tensor(input_ids).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            max_length=128,
        )

        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return output


def ged_inference(model, tokenizer, device, error):
    label_lst = ["B-E", "E-E", "I-E", "O"]
    id_to_label = {i: label for i, label in enumerate(label_lst)}

    inputs = tokenizer(
        [error],
        max_length=128,
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v).to(device) for k, v in inputs.items()})
        probs = outputs.logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)

        predicted_tags = []
        for pred in preds[1:-1]:
            label = id_to_label[pred.item()]
            if 'E' in label:
                predicted_tags.append('E')
            else:
                predicted_tags.append('O')

    return predicted_tags


def inference_fn():
    args = parse_args()

    # cuda 설정
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model & tokenizer load
    # ged tokenizer 및 model load
    ged_tokenizer = KoCharElectraTokenizer.from_pretrained('monologg/kocharelectra-base-discriminator')
    ged_model = MODEL_FOR_TOKEN_CLASSIFICATION[args.ged_model_type].from_pretrained(args.ged_save_path)
    ged_model = ged_model.to(device)
    ged_model.eval()

    # gec tokenizer 및 model load
    # tokenizer 및 model load
    gec_tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    gec_model = BartForConditionalGeneration.from_pretrained(args.gec_save_path)
    gec_model = gec_model.to(device)
    gec_model.eval()

    # # noise classification tokenizer 및 model load
    # tokenizer = ElectraTokenizer.from_pretrained()
    # nc_model = ElectraForSequenceClassification.from_pretrained()
    # nc_model.to(device)
    # nc_model.eval()

    # 입력 데이터
    error = args.error
    correct = args.correct

    # GED
    e_pattern = re.compile(r'E+')
    ged_tags = ged_inference(ged_model, ged_tokenizer, device, error)
    if 'E' not in ged_tags:
        output = error
    else:
        e_flag = True
        e_words_lst = []
        c_words_lst = []
        annotation = [e for e in error]
        adjust_len = 0
        while e_flag:
            s_ei, e_ei = e_pattern.search(''.join(ged_tags)).span()
            e_word = error[s_ei:e_ei]
            annotation[s_ei + adjust_len : e_ei + adjust_len] = f'<unused0>{e_word}<unused1>'
            gec_input_sentence = ''.join(annotation)
            c_word = gec_inference(gec_model, gec_tokenizer, device, gec_input_sentence)
            next_sentence = ''.join(annotation).replace(f'<unused0>{e_word}<unused1>', f'{c_word}')
            annotation = [i for i in next_sentence]
            ged_tags[s_ei:e_ei] = ['O'] * (e_ei - s_ei)
            adjust_len += len(c_word) - len(e_word)
            e_words_lst.append(e_word)
            c_words_lst.append(c_word)
            if 'E' not in ged_tags:
                e_flag = False
        output = ''.join(annotation)
    print(f'{error}\t{output}\t{correct}')


if __name__ == '__main__':
    inference_fn()
