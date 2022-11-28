import torch
import argparse
from tokenization_kocharelectra import KoCharElectraTokenizer
from utils import MODEL_FOR_TOKEN_CLASSIFICATION

def parse_args():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--model_type", default='base', type=str)
    parser.add_argument("--save_path", default='output/checkpoint-3808', type=str)

    # Dataset arguments
    parser.add_argument("--error", default='너가 굴도 좋아하나', type=str)
    parser.add_argument("--correct", default='네가 굴도 좋아하니?', type=str)

    args = parser.parse_args()

    return args

def predict():
    args = parse_args()
    tokenizer = KoCharElectraTokenizer.from_pretrained('monologg/kocharelectra-base-discriminator')
    model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(args.save_path)
    label_lst = ["B-E", "E-E", "I-E", "O"]
    id_to_label = {i: label for i, label in enumerate(label_lst)}

    sentence = args.error
    label = args.correct
    inputs = tokenizer(
        [sentence],
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        probs = outputs.logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [id_to_label[pred.item()] for pred in preds]
        result = []
        for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
            if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                token_result = {
                    "token": token,
                    "predicted_tag": predicted_tag,
                    "top_prob": str(round(top_prob[0].item(), 4)),
                }
                result.append(token_result)

        print(f'오류 문장: {sentence}')
        print(f'교정 문장: {label}')
        for token in result:
            print(token)

if __name__ == '__main__':
    predict()
