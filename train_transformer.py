import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_dataloaders

# ========== Part 1: 基础设置 ==========
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# ========== Part 2: 数据 ==========
train_loader, val_loader, test_loader, vocab = get_dataloaders(batch_size=32, max_len=32)


# ========== Part 3: 多头注意力 ==========
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_O(output)


# ========== Part 4: Transformer Encoder Block ==========
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


# ========== Part 5: Transformer 分类模型（小模型）==========
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=2, d_ff=128,
                 num_layers=1, num_classes=5, max_len=32, dropout=0.3, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.shape
        mask = (x != self.pad_idx)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        attn_mask = mask.unsqueeze(1).unsqueeze(2)

        for block in self.encoder_blocks:
            x = block(x, attn_mask)

        x = self.norm(x)

        mask_float = mask.unsqueeze(-1).float()
        summed = (x * mask_float).sum(dim=1)
        counts = mask_float.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits


# ========== Part 5.5: 实例化模型 ==========
model = TransformerClassifier(
    vocab_size=vocab.vocab_size,
    d_model=64,
    num_heads=2,
    d_ff=128,
    num_layers=1,
    num_classes=5,
    max_len=32,
    dropout=0.3,
    pad_idx=vocab.pad_idx
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total_params:,}")

dummy = torch.randint(0, vocab.vocab_size, (4, 32)).to(device)
out = model(dummy)
print("输出形状:", out.shape)

criterion = nn.CrossEntropyLoss()

epochs = 30
patience = 5
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

total_steps = len(train_loader) * epochs
warmup_steps = int(total_steps * 0.15)


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
print(f"总步数: {total_steps}, warmup 步数: {warmup_steps}")


# ========== Part 6: 训练与评估函数 ==========
def train_one_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


# ========== Part 7: 主训练循环 ==========
counter = 0
best_val_loss = float('inf')
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), 'models/best_transformer.pth')
    else:
        counter += 1

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:02d}/{epochs} | "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
          f"lr {current_lr:.6f}")

    if counter >= patience:
        print(f"\n早停：连续 {patience} 个 epoch 验证损失未改善")
        break

print(f"\n最佳验证准确率: {best_val_acc:.4f}")


# ========== Part 8: 测试集评估 ==========
print("\n在测试集上评估最佳模型...")
model.load_state_dict(torch.load('models/best_transformer.pth'))
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
plt.title('Transformer Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train')
plt.plot(history['val_acc'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Transformer Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/transformer_training_curve.png', dpi=100)
plt.close()
print("已保存训练曲线到 results/transformer_training_curve.png")