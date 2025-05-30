import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.mistral.modeling_mistral import (
    MistralModel, MistralPreTrainedModel)


class LSTMHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_labels):
        super().__init__()

        self.lstm = nn.LSTM(
            in_features,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.post_projection = nn.Linear(in_features, num_labels, bias=False)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, (_, _) = self.lstm(x)
        logits = self.post_projection(x)
        return logits


class MistralForFeatureExtraction(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = MistralModel(config)
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        **kwargs,
    ):

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]

        return sequence_output


class MistralForPII(nn.Module):
    def __init__(self, model, num_labels):
        super().__init__()

        self.num_labels = num_labels

        self.model = model

        self.classification_head = LSTMHead(
            in_features=model.config.hidden_size,
            hidden_dim=model.config.hidden_size//2,
            num_labels=self.num_labels
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        **kwargs,
    ):

        sequence_output = self.model(
            input_ids,
            attention_mask=attention_mask,
        )

        logits = self.classification_head(sequence_output.float())

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )
