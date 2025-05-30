from dataclasses import dataclass, field

import torch
from transformers import DataCollatorForTokenClassification


@dataclass
class PIIDataCollator(DataCollatorForTokenClassification):
    """
    Data collator for the PII Data Detection task
    """
    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100
    return_tensors = "pt"

    def __call__(self, features):
        """
        prepare a batch data from features
        """
        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        documents = [feature["document"] for feature in features]
        word_ids = [feature["word_ids"] for feature in features]
        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        # padding for span_head_idxs, span_tail_idxs and
        seq_len = len(batch["input_ids"][0])
        span_len = max([len(l) for l in span_head_idxs])

        default_head_idx = seq_len - 2
        default_tail_idx = seq_len

        for idx, (ex_word_ids, ex_span_head_idxs, ex_span_tail_idxs) in enumerate(zip(word_ids, span_head_idxs, span_tail_idxs)):
            ex_word_ids = ex_word_ids + [-1] * (span_len - len(ex_word_ids))
            ex_span_head_idxs = ex_span_head_idxs + [default_head_idx] * (span_len - len(ex_span_head_idxs))
            ex_span_tail_idxs = ex_span_tail_idxs + [default_tail_idx] * (span_len - len(ex_span_tail_idxs))

            word_ids[idx] = ex_word_ids
            span_head_idxs[idx] = ex_span_head_idxs
            span_tail_idxs[idx] = ex_span_tail_idxs

        batch["document"] = documents
        batch["word_ids"] = word_ids
        batch["span_head_idxs"] = span_head_idxs
        batch["span_tail_idxs"] = span_tail_idxs

        if labels is not None:
            for idx, ex_labels in enumerate(labels):
                ex_labels = ex_labels + [self.label_pad_token_id] * (span_len - len(ex_labels))
                labels[idx] = ex_labels
            batch["labels"] = labels

        tensor_keys = [
            "input_ids",
            "attention_mask",
            "span_head_idxs",
            "span_tail_idxs",
            "word_ids",
            "document",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        return batch


def show_batch(
        batch,
        tokenizer,
        id2label,
        n_examples=16,
        task='training',
        print_fn=print
):
    bs = batch['input_ids'].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")

    n_examples = min(n_examples, bs)
    print_fn(f"Showing {n_examples} from a {task} batch...")

    for idx in range(n_examples):
        print_fn(f"--- Example {idx+1} ------")

        input_text = tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)

        # spans --
        spans = []
        for h, t in zip(batch['span_head_idxs'][idx], batch['span_tail_idxs'][idx]):
            if h >= 0:
                current_span = tokenizer.decode(batch['input_ids'][idx][h:t], skip_special_tokens=False)
                spans.append(current_span)

        print_fn(f"INPUT:\n{input_text}")

        if "infer" not in task.lower():
            print_fn("--"*20)
            labels = batch['labels'][idx].tolist()
            labels = [id2label.get(l, 'O') for l in labels]

            for t, l in zip(spans, labels):
                if l != 'O':
                    print_fn(f"{t:<16}: {l}")
        print_fn('~~'*40)
