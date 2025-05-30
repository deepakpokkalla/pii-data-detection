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
            bias=False,
        )

        self.tok_projection = nn.Linear(in_features, num_labels, bias=False)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, (_, _) = self.lstm(x)
        logits = self.tok_projection(x)
        return logits


class MistralForPII(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = MistralModel(config)

        self.classification_head = LSTMHead(
            in_features=config.hidden_size,
            hidden_dim=config.hidden_size//2,
            num_labels=self.num_labels
        )

        self.dropout = nn.Dropout(0.1)
        self.loss_fn = nn.CrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        labels=None,
        **kwargs,
    ):
        bs = input_ids.shape[0]

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]

        # ---
        mean_feature_vector = []

        for i in range(bs):
            span_vec_i = []

            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                span_vec_i.append(torch.mean(sequence_output[i, head:tail], dim=0))  # [h, h, ..num_spacy_tokens]
            span_vec_i = torch.stack(span_vec_i)  # (num_spacy_tokens,  h)
            mean_feature_vector.append(span_vec_i)

        mean_feature_vector = torch.stack(mean_feature_vector)

        logits = self.classification_head(mean_feature_vector)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )
