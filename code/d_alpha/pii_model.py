import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model, DebertaV2PreTrainedModel)

class DebertaForPII(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classification_head = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        **kwargs,
    ):

        outputs = self.deberta(input_ids,attention_mask=attention_mask)

        sequence_output = outputs[0] # (batch_size, seq_len, hidden_size)

        sequence_output = self.dropout(sequence_output)
        logits = self.classification_head(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
