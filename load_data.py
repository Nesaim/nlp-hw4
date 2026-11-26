"""Data loading for the T5 text-to-SQL baseline.

Uses the original text and default T5 tokenizer without truncation.
Decoder inputs use T5's standard pad-token start; targets end in EOS.
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

PAD_IDX = 0
TOKENIZER_NAME = "google-t5/t5-small"


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_NAME)
        if self.tokenizer.pad_token_id != PAD_IDX:
            raise ValueError("The starter training loop expects pad_token_id=0")
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        questions = load_lines(os.path.join(data_folder, f"{split}.nl"))
        if not questions:
            raise ValueError(f"Empty split: {split}")
        encoder_ids = tokenizer(
            questions, add_special_tokens=True,
            padding=False, truncation=False
        )["input_ids"]
        start = torch.tensor([tokenizer.pad_token_id], dtype=torch.long)

        if split == "test":
            return [
                (torch.tensor(ids, dtype=torch.long), start.clone())
                for ids in encoder_ids
            ]

        queries = load_lines(os.path.join(data_folder, f"{split}.sql"))
        if len(questions) != len(queries):
            raise ValueError(f"Question/SQL length mismatch in {split}")
        target_ids = tokenizer(
            queries, add_special_tokens=True,
            padding=False, truncation=False
        )["input_ids"]

        examples = []
        for ids, targets in zip(encoder_ids, target_ids):
            if not targets or targets[-1] != tokenizer.eos_token_id:
                raise ValueError("Every SQL target must end with EOS")
            encoder = torch.tensor(ids, dtype=torch.long)
            target = torch.tensor(targets, dtype=torch.long)
            decoder = torch.cat([start, target[:-1]])
            examples.append((encoder, decoder, target, start.clone()))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def normal_collate_fn(batch):
    encoders, decoders, targets, starts = zip(*batch)
    encoder_ids = pad_sequence(encoders, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence(decoders, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(starts)
    return (encoder_ids, encoder_mask, decoder_inputs,
            decoder_targets, initial_decoder_inputs)


def test_collate_fn(batch):
    encoders, starts = zip(*batch)
    encoder_ids = pad_sequence(encoders, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    return encoder_ids, encoder_mask, torch.stack(starts)


def get_dataloader(batch_size, split):
    dset = T5Dataset("data", split)
    return DataLoader(
        dset, batch_size=batch_size, shuffle=(split == "train"),
        collate_fn=test_collate_fn if split == "test" else normal_collate_fn
    )


def load_t5_data(batch_size, test_batch_size):
    return (get_dataloader(batch_size, "train"),
            get_dataloader(test_batch_size, "dev"),
            get_dataloader(test_batch_size, "test"))


def load_prompting_data(data_folder):
    return (
        load_lines(os.path.join(data_folder, "train.nl")),
        load_lines(os.path.join(data_folder, "train.sql")),
        load_lines(os.path.join(data_folder, "dev.nl")),
        load_lines(os.path.join(data_folder, "dev.sql")),
        load_lines(os.path.join(data_folder, "test.nl")),
    )
