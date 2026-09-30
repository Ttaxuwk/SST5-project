import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# ========== Part 1: 基础设置 ==========
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# ========== Part 2: 加载 BERT Tokenizer ==========
print("加载 BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ========== Part 3: 数据加载 ==========
print("加载数据集...")
dataset = load_dataset("SetFit/sst5")


def clean_dataset(texts, labels):
    """过滤掉 None 和空文本"""
    cleaned_texts, cleaned_labels = [], []
    for text, label in zip(texts, labels):
        if text is not None and isinstance(text, str) and len(text.strip()) > 0:
            cleaned_texts.append(text)
            cleaned_labels.append(label)
    return cleaned_texts, cleaned_labels


train_texts, train_labels = clean_dataset(dataset["train"]["text"], dataset["train"]["label"])
val_texts, val_labels = clean_dataset(dataset["validation"]["text"], dataset["validation"]["label"])
test_texts, test_labels = clean_dataset(dataset["test"]["text"], dataset["test"]["label"])

print(f"训练集: {len(train_texts)}")
print(f"验证集: {len(val_texts)}")
print(f"测试集: {len(test_texts)}")


# ========== Part 4: 自定义 Dataset（预 tokenize）==========
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.labels = labels
        print(f"预 tokenize {len(texts)} 条数据...")
        encodings = tokenizer(
            texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        print("Tokenize 完成")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# SST-5 句子极短，max_len=32 足够
train_set = BERTDataset(train_texts, train_labels, tokenizer, max_len=32)
val_set = BERTDataset(val_texts, val_labels, tokenizer, max_len=32)
test_set = BERTDataset(test_texts, test_labels, tokenizer, max_len=32)

# 数据量小，batch_size 可以大一些
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)


# ========== Part 5: BERT 分类模型（冻结更多层）==========
class BERTClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3, freeze_layers=10):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            attn_implementation="eager"   # MPS 兼容
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        # 冻结 embeddings 和前 10 层（只训练最后 2 层 + 分类头）
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(freeze_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        return logits


# 数据量小，冻结更多层防止过拟合
model = BERTClassifier(num_classes=5, freeze_layers=10).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

dummy_ids = torch.randint(0, 1000, (2, 32)).to(device)
dummy_mask = torch.ones(2, 32).long().to(device)
out = model(dummy_ids, dummy_mask)
print("输出形状:", out.shape)


# ========== Part 6: 训练与评估函数 ==========
def train_one_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="训练", leave=False)
    for input_ids, attention_mask, y in pbar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({
            'loss': f'{running_loss / total:.4f}',
            'acc': f'{correct / total:.4f}'
        })

    return running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="验证", leave=False)
    for input_ids, attention_mask, y in pbar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, y)

        running_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({
            'loss': f'{running_loss / total:.4f}',
            'acc': f'{correct / total:.4f}'
        })

    return running_loss / total, correct / total


# ========== Part 7: 主训练循环 ==========
criterion = nn.CrossEntropyLoss()

epochs = 10
patience = 3
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5, weight_decay=0.01
)

total_steps = len(train_loader) * epochs
warmup_steps = int(total_steps * 0.1)


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))


scheduler = LambdaLR(optimizer, lr_lambda)
print(f"总步数: {total_steps}, warmup 步数: {warmup_steps}")

counter = 0
best_val_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n开始训练...\n")
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scheduler)
    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), 'models/best_bert.pth')
    else:
        counter += 1

    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f}")

    if counter >= patience:
        print(f"\n早停：连续 {patience} 个 epoch 验证准确率未改善")
        break

print(f"\n最佳验证准确率: {best_val_acc:.4f}")


# ========== Part 8: 测试集评估 ==========
print("\n在测试集上评估最佳 BERT...")
model.load_state_dict(torch.load('models/best_bert.pth'))
test_loss, test_acc = eval_one_epoch(model, test_loader, criterion)
print(f"测试集准确率: {test_acc:.4f}")
print(f"测试集损失: {test_loss:.4f}")


# ========== Part 9: 画训练曲线 ==========
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('BERT Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train')
plt.plot(history['val_acc'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('BERT Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/bert_training_curve.png', dpi=100)
plt.close()
print("已保存训练曲线到 results/bert_training_curve.png")