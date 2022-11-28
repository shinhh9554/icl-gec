import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraPreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchcrf import CRF


class ElectraCrfForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraCrfForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # add hidden states and attention if they are here

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.byte())
            tags = self.crf.decode(logits, attention_mask.byte())
            outputs = (-1 * loss, tags,) + outputs[1:]

        else:
            if attention_mask is not None:
                tags = self.crf.decode(logits, attention_mask.byte())
            else:
                tags = self.crf.decode(logits)
            outputs = (tags,) + outputs[1:]

        return outputs


class ElectraBiGRUForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraBiGRUForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.bi_gru = nn.GRU(config.hidden_size,
                             config.hidden_size // 2,
                             num_layers=2,
                             dropout=classifier_dropout,
                             batch_first=True,
                             bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, hc = self.bi_gru(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class ElectraBiGRUFCRForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraBiGRUFCRForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.bi_gru = nn.GRU(config.hidden_size,
                             config.hidden_size // 2,
                             num_layers=2,
                             dropout=classifier_dropout,
                             batch_first=True,
                             bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, hc = self.bi_gru(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.byte())
            tags = self.crf.decode(logits, attention_mask.byte())
            outputs = (-1 * loss, tags,) + outputs[1:]

        else:
            if attention_mask is not None:
                tags = self.crf.decode(logits, attention_mask.byte())
            else:
                tags = self.crf.decode(logits)
            outputs = (tags,) + outputs[1:]

        return outputs


class ElectraBiLSTMForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraBiLSTMForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.bi_lstm = nn.LSTM(config.hidden_size,
                               config.hidden_size // 2,
                               num_layers=2,
                               dropout=classifier_dropout,
                               batch_first=True,
                               bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, hc = self.bi_lstm(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class ElectraBiLSTMFCRForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraBiLSTMFCRForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.bi_lstm = nn.LSTM(config.hidden_size,
                               config.hidden_size // 2,
                               num_layers=2,
                               dropout=classifier_dropout,
                               batch_first=True,
                               bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, hc = self.bi_lstm(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.byte())
            tags = self.crf.decode(logits, attention_mask.byte())
            outputs = (-1 * loss, tags,) + outputs[1:]

        else:
            if attention_mask is not None:
                tags = self.crf.decode(logits, attention_mask.byte())
            else:
                tags = self.crf.decode(logits)
            outputs = (tags,) + outputs[1:]

        return outputs

