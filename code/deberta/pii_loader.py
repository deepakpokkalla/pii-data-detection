from dataclasses import dataclass, field

import torch
from transformers import DataCollatorForTokenClassification


def apply_mask_augmentation(input_ids, tokenizer, mask_prob=0.1):
    # input_ids = deepcopy(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    indices_mask = torch.bernoulli(torch.full(input_ids.shape, mask_prob)).bool()

    do_not_mask_tokens = list(set(tokenizer.all_special_ids))

    pass_gate = [
        [0 if token_id in do_not_mask_tokens else 1 for token_id in token_id_seq] for token_id_seq in input_ids
    ]
    pass_gate = torch.tensor(pass_gate, dtype=torch.bool)

    indices_mask = torch.logical_and(indices_mask, pass_gate)
    input_ids[indices_mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return input_ids


@dataclass
class PIIDataCollator(DataCollatorForTokenClassification):
    """
    Data collator for the PII Data Detection task
    """
    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100 # labels to be ignored for loss computations
    return_tensors = "pt"

    # invoked to collate the features (examples in valid_ds)
    def __call__(self, features):
        """
        prepare a batch data from features
        """
        labels = [f["labels"] for f in features] if "labels" in features[0] else None
        documents = [f["document"] for f in features]
        word_ids = [f["word_ids"] for f in features]

        # Extract just input_ids and attention_mask for padding 
        inputs = [
            {'input_ids': f['input_ids'],
             'attention_mask': f['attention_mask']
            } 
            for f in features
        ]

        # returns a batch encoding
        batch = self.tokenizer.pad(
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None, # convert later manually
        )

        # padding for word_ids
        seq_len = len(batch["input_ids"][0])
        for idx, ex_word_ids in enumerate(word_ids):
            word_ids[idx] = ex_word_ids + [-1] * (seq_len - len(ex_word_ids))

        batch["word_ids"] = word_ids
        batch["document"] = documents

        # padding for labels: set "-100" to ignore
        if labels is not None:
            for idx, ex_labels in enumerate(labels):
                labels[idx] = ex_labels + [self.label_pad_token_id] * (seq_len - len(ex_labels))
            batch["labels"] = labels

        # convert to tensors
        tensor_keys = [
            "input_ids",
            "attention_mask",
            "word_ids",
            "document",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        return batch


@dataclass
class PIIDataCollatorTrain(DataCollatorForTokenClassification):
    """
    Data collator for the PII Data Detection task with optional mask augmentation
    """
    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100 # labels to be ignored for loss computations
    return_tensors = "pt"
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        [setattr(self, k, v) for k, v in self.kwargs.items()]

    # invoked to collate the features (examples in train_ds)
    def __call__(self, features):
        """
        prepare a batch data from features
        """
        labels = [f["labels"] for f in features] if "labels" in features[0] else None
        documents = [f["document"] for f in features]
        word_ids = [f["word_ids"] for f in features]

        # Extract just input_ids and attention_mask for padding 
        inputs = [
            {'input_ids': f['input_ids'],
             'attention_mask': f['attention_mask']
            } 
            for f in features
        ]

        # returns a batch encoding
        batch = self.tokenizer.pad(
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,  # convert later manually
        )

        # Random masking during training (similar to BERT pretraining), avoids SPL tokens. 
        if self.cfg.train_params.use_mask_aug:
            batch["input_ids"] = apply_mask_augmentation(
                batch["input_ids"], self.tokenizer, self.cfg.train_params.mask_aug_prob
            )

        # padding for word_ids
        seq_len = len(batch["input_ids"][0])
        for idx, ex_word_ids in enumerate(word_ids):
            word_ids[idx] = ex_word_ids + [-1] * (seq_len - len(ex_word_ids))

        batch["word_ids"] = word_ids
        batch["document"] = documents
        
        # padding for labels: set "-100" to ignore 
        if labels is not None:
            for idx, ex_labels in enumerate(labels):
                labels[idx] = ex_labels + [self.label_pad_token_id] * (seq_len - len(ex_labels))
            batch["labels"] = labels

        # convert to tensors
        tensor_keys = [
            "input_ids",
            "attention_mask",
            "word_ids",
            "document",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        return batch

# Print samples from train dataloader (tokens: labels) for visualization
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
        print_fn(f"--- Example {idx+1} ---")

        # skip_special_tokens = False by default
        input_text = tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][idx])
        # print_fn(f"INPUT:\n{input_text}")

        if "infer" not in task.lower(): # not inference
            print_fn("--"*10)
            labels = batch['labels'][idx].tolist()
            # batch["labels"] here - it's numeric ids here!?
            labels = [id2label.get(l, 'O') for l in labels]
            for t, l in zip(tokens, labels):
                if l != 'O':
                    print_fn(f"{t:<16}: {l}") # left-aligned with 16 characters width
        # print_fn('~~'*10)
        print_fn()
