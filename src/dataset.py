from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# Load dataset
dataset = load_dataset("bentrevett/multi30k")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

MAX_LEN = 32


class TranslationDataset(Dataset):

    def __init__(self, split):
        self.data = dataset[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        src = tokenizer(
            sample["en"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        tgt = tokenizer(
            sample["de"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "src_ids": src["input_ids"].squeeze(0),
            "src_mask": src["attention_mask"].squeeze(0),
            "tgt_ids": tgt["input_ids"].squeeze(0),
            "tgt_mask": tgt["attention_mask"].squeeze(0)
        }


train_dataset = TranslationDataset("train")
val_dataset = TranslationDataset("validation")

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8
)
if __name__ == "__main__":
    print(next(iter(train_loader)))
