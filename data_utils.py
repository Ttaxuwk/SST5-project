import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import re


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


class Vocab:
    def __init__(self, texts, max_size=10000, min_freq=2):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        self.pad_idx = 0
        self.unk_idx = 1

        self.word2idx = {'<pad>': 0, '<unk>': 1}
        idx = 2
        for word, freq in counter.most_common(max_size - 2):
            if freq < min_freq:
                break
            self.word2idx[word] = idx
            idx += 1

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text, max_len=32):
        tokens = tokenize(text)
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]

        if len(ids) < max_len:
            ids = ids + [self.pad_idx] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids


class SST5Dataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def clean_dataset(texts, labels):
    """过滤掉 None 和空文本"""
    cleaned_texts, cleaned_labels = [], []
    for text, label in zip(texts, labels):
        if text is not None and isinstance(text, str) and len(text.strip()) > 0:
            cleaned_texts.append(text)
            cleaned_labels.append(label)
    return cleaned_texts, cleaned_labels


def load_sst5_data():
    print("加载 SST-5 数据集...")
    dataset = load_dataset("SetFit/sst5")

    train_texts, train_labels = clean_dataset(
        dataset["train"]["text"], dataset["train"]["label"])
    val_texts, val_labels = clean_dataset(
        dataset["validation"]["text"], dataset["validation"]["label"])
    test_texts, test_labels = clean_dataset(
        dataset["test"]["text"], dataset["test"]["label"])

    print(f"训练集: {len(train_texts)}")
    print(f"验证集: {len(val_texts)}")
    print(f"测试集: {len(test_texts)}")

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def get_dataloaders(batch_size=32, max_len=32, max_vocab=10000):
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_sst5_data()

    print("构建词表...")
    vocab = Vocab(train_texts, max_size=max_vocab, min_freq=2)
    print(f"词表大小: {vocab.vocab_size}")

    train_set = SST5Dataset(train_texts, train_labels, vocab, max_len)
    val_set = SST5Dataset(val_texts, val_labels, vocab, max_len)
    test_set = SST5Dataset(test_texts, test_labels, vocab, max_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, vocab