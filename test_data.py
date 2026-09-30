from data_utils import get_dataloaders, tokenize

train_loader, val_loader, test_loader, vocab = get_dataloaders()

# 类别名
classes = ['very negative', 'negative', 'neutral', 'positive', 'very positive']

# 取一个 batch
x, y = next(iter(train_loader))
print("一个 batch 的输入形状:", x.shape)
print("一个 batch 的标签形状:", y.shape)
print("前 10 个标签:", y[:10].tolist())
print("对应类别:", [classes[i] for i in y[:10].tolist()])

# 反查 token
sample_ids = x[0][:20].tolist()
sample_words = [vocab.idx2word[i] for i in sample_ids]
print("前 20 个 token:", sample_words)

# 测试分词
print("\n分词示例:")
print(tokenize("a stirring and beautiful film"))