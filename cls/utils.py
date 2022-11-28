import os
import tqdm
import logging

from transformers import ElectraForSequenceClassification, BertForSequenceClassification, ElectraTokenizer, BertTokenizer

MODEL_CLASSES = {
    'klue_bert': (BertForSequenceClassification, BertTokenizer, 'klue/bert-base'),
    'kcbert': (BertForSequenceClassification, BertTokenizer, 'beomi/kcbert-base'),
    'koelectra': (ElectraForSequenceClassification, ElectraTokenizer, 'monologg/koelectra-base-v3-discriminator'),
    'kcelectra': (ElectraForSequenceClassification, ElectraTokenizer, "beomi/KcELECTRA-base")
}


def load_model(model_type):
    model, _, model_name = MODEL_CLASSES[model_type]
    return model, model_name

def load_tokenizer(model_type):
    _, tokenizer, model_name = MODEL_CLASSES[model_type]
    return tokenizer.from_pretrained(model_name)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def create_logger(output_dir: str, mode: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    file_handler = logging.FileHandler(os.path.join(output_dir, f"{mode}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return logger
