import tqdm
import logging

from transformers import ElectraForTokenClassification

from ged.net import (
	ElectraCrfForTokenClassification,
	ElectraBiGRUFCRForTokenClassification,
	ElectraBiLSTMFCRForTokenClassification,
	ElectraBiLSTMForTokenClassification,
	ElectraBiGRUForTokenClassification
)

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "base": ElectraForTokenClassification,
	"base-crf": ElectraCrfForTokenClassification,
	"bilstm": ElectraBiLSTMForTokenClassification,
	"bilstm-crf": ElectraBiLSTMFCRForTokenClassification,
	"bigru": ElectraBiGRUForTokenClassification,
	"bigru-crf": ElectraBiGRUFCRForTokenClassification
}