from datasets import Dataset
from transformers import AutoTokenizer


class PIIDataset:
    """
    Dataset class for the PII Data Detection task
    """

    def __init__(self, cfg, label2id):
        self.cfg = cfg

        # tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.backbone_path,
            use_fast=True,
            padding_side="right",
            truncation_side="right"
        )

        # --- for "mistralai/Mistral-7B-v0.1"
        # if the eos token is an empty string, we assign it to a token
        if self.tokenizer.eos_token == "":
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            self.tokenizer.eos_token = "</s>"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # ---

        self.label2id = label2id
        self.ignore_idx = -100  # labels to be ignored for loss computations

    def process_inputs(self, examples):
        to_return = dict()

        input_tokens = []
        for ex_tokens, ex_trailing_whitespace in zip(examples['tokens'], examples['trailing_whitespace']):
            # ex_leading_whitespace = [False] + ex_trailing_whitespace[:-1]
            ex_leading_whitespace = [False]*len(ex_trailing_whitespace) # Why? 
            ex_input_tokens = [" " + t if ls else t for t, ls in zip(ex_tokens, ex_leading_whitespace)]
            input_tokens.append(ex_input_tokens)

        # Understood stride & overflowing tokens. why no padding but truncation - handled in collate_fn?? 
        re_tokenized_text = self.tokenizer(
            input_tokens,
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=True,
            is_split_into_words=True,
            return_overflowing_tokens=True,
            return_token_type_ids=False,
            stride=self.cfg.model.stride,
            return_length=True,
        )  # length more than original number of examples (UNDERSTAND THIS!!)

        # mapping tokens with corresponding documents; repeated if overflowing tokens
        # Link: https://huggingface.co/learn/nlp-course/chapter6/3b
        document = [
            examples['document'][sid] for sid in re_tokenized_text['overflow_to_sample_mapping']
        ]

        to_return['document'] = document
        to_return['input_ids'] = re_tokenized_text['input_ids']
        to_return['attention_mask'] = re_tokenized_text['attention_mask']
        to_return['input_length'] = re_tokenized_text['length'] # length of each tokenized sequence

        word_ids = []
        for idx in range(len(re_tokenized_text['input_ids'])):
            ex_word_ids = re_tokenized_text.word_ids(idx)
            ex_word_ids = [-1 if wid is None else wid for wid in ex_word_ids]
            word_ids.append(ex_word_ids)
        to_return['word_ids'] = word_ids

        # label alignment --- (UNDERSTOOD THIS! - Revise, if needed)
        # word_ids: Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.
        if 'provided_labels' in examples:
            new_labels = []
            for idx, eid in enumerate(re_tokenized_text['overflow_to_sample_mapping']):
                ex_word_ids = to_return['word_ids'][idx]
                ex_labels = examples['provided_labels'][eid] # eid for corresponding 'document'
                ex_new_labels = [
                    self.ignore_idx if wid == -1 else self.label2id.get(ex_labels[wid], self.ignore_idx) for wid in ex_word_ids
                ]
                new_labels.append(ex_new_labels)
            to_return['labels'] = new_labels

        return to_return

    def get_dataset(self, examples):
        # examples = deepcopy(examples)

        dataset_dict = {
            "document": [x["document"] for x in examples],
            "full_text": [x["full_text"] for x in examples],
            "tokens": [x["tokens"] for x in examples],
            "trailing_whitespace": [x["trailing_whitespace"] for x in examples],
        }

        if "labels" in examples[0]:  # test examples don't have labels ---
            dataset_dict["provided_labels"] = [x["labels"] for x in examples]

        task_dataset = Dataset.from_dict(dataset_dict)

        task_dataset = task_dataset.map(
            self.process_inputs,
            batched=True, # batch processing --> process_inputs
            batch_size=512,
            num_proc=self.cfg.model.num_proc,
            remove_columns=task_dataset.column_names, # remove all columns names
        )

        return task_dataset
