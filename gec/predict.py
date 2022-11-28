import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--save_path", default='output/checkpoint-3808', type=str)

    # Dataset arguments
    parser.add_argument("--error", default='너가 굴도 좋아하나', type=str)
    parser.add_argument("--correct", default='네가 굴도 좋아하니?', type=str)

    args = parser.parse_args()

    return args

def inference_fn():
    args = parse_args()

    # cuda 설정
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # tokenizer 및 model load
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    model = BartForConditionalGeneration.from_pretrained(args.save_path)
    model = model.to(device)
    model.eval()

    error = args.error
    correct_word = args.correct

    input_ids = tokenizer.encode(error, return_tensors="pt")
    input_ids = torch.tensor(input_ids).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=False,
            max_length=256,
        )

        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f'정답: {correct_word}')
    print(f'예측: {output}')

if __name__ == '__main__':
    inference_fn()