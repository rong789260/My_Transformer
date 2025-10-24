import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
import os
import re
import time  # 导入时间模块

# 设置随机种子，保证可复现性
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# -------------------------- 1. 数据预处理 --------------------------
def split_english_punctuation(text):
    """Separate English punctuation from words"""
    punctuation = r"([,.?!;:\"'])"
    text = re.sub(punctuation, r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_parallel_corpus(file_path):
    """Read English-Chinese parallel corpus"""
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    en, zh = parts
                    en_split = split_english_punctuation(en)
                    if en_split and zh:
                        pairs.append((en_split, zh))
                    else:
                        print(f"Warning: Empty sentence after cleaning, skipped -> Original line: {line}")
                else:
                    print(f"Warning: Invalid format, skipped -> {line}")
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}, please check the path")
    return pairs


class TranslationDataset(Dataset):
    """English-Chinese parallel corpus dataset"""

    def __init__(self, data_pairs, src_vocab, tgt_vocab, max_len=20):
        self.data = data_pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.pad_idx = src_vocab["<PAD>"]
        self.sos_idx = tgt_vocab["<SOS>"]
        self.eos_idx = tgt_vocab["<EOS>"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.data[idx]

        # Encode English sentence
        src_tokens = src_sentence.split()
        src_ids = [self.src_vocab.get(token, self.src_vocab["<UNK>"]) for token in src_tokens]
        src_ids = src_ids[:self.max_len]
        src_len = len(src_ids)
        src_ids += [self.pad_idx] * (self.max_len - src_len)

        # Encode Chinese sentence
        tgt_tokens = list(tgt_sentence)
        tgt_ids = [self.sos_idx] + [self.tgt_vocab.get(token, self.tgt_vocab["<UNK>"]) for token in tgt_tokens]
        tgt_ids = tgt_ids[:self.max_len - 1]
        tgt_ids += [self.eos_idx]
        tgt_len = len(tgt_ids)
        tgt_ids += [self.pad_idx] * (self.max_len - tgt_len)

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "src_len": src_len,
            "tgt_len": tgt_len
        }


def build_vocab(sentences, min_freq=1):
    """Build vocabulary"""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    token_counts = {}
    for sent in sentences:
        if isinstance(sent, str):
            tokens = sent.split()
        else:
            tokens = sent
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


# -------------------------- 2. Transformer 模型实现 --------------------------
class PositionalEncoding(nn.Module):
    """Positional Encoding"""

    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    """Position-Wise Feed-Forward Network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):
    """Encoder Layer"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


class DecoderLayer(nn.Module):
    """Decoder Layer"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        cross_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(cross_output))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x


class Transformer(nn.Module):
    """Complete Transformer Model"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_heads=4,
                 num_encoder_layers=2, num_decoder_layers=2, d_ff=512, dropout=0.1,
                 use_pos_encoding=True, use_layer_norm=True):
        super().__init__()
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding
        self.use_layer_norm = use_layer_norm

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        if self.use_pos_encoding:
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_mask(self, src, tgt):
        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_self_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=src.device)).unsqueeze(0).unsqueeze(0)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_self_mask = tgt_self_mask.bool() & tgt_pad_mask.bool()
        tgt_cross_mask = src_mask

        return src_mask, tgt_self_mask, tgt_cross_mask

    def encode(self, src, src_mask):
        x = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        if self.use_pos_encoding:
            x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            if not self.use_layer_norm:
                attn_output, _ = layer.self_attn(x, x, x, src_mask)
                x = x + layer.dropout1(attn_output)
                ffn_output = layer.ffn(x)
                x = x + layer.dropout2(ffn_output)
            else:
                x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, tgt_self_mask, tgt_cross_mask):
        x = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        if self.use_pos_encoding:
            x = self.pos_encoding(x)

        for layer in self.decoder_layers:
            if not self.use_layer_norm:
                attn_output, _ = layer.self_attn(x, x, x, tgt_self_mask)
                x = x + layer.dropout1(attn_output)
                cross_output, _ = layer.cross_attn(x, enc_output, enc_output, tgt_cross_mask)
                x = x + layer.dropout2(cross_output)
                ffn_output = layer.ffn(x)
                x = x + layer.dropout3(ffn_output)
            else:
                x = layer(x, enc_output, tgt_self_mask, tgt_cross_mask)
        return x

    def forward(self, src, tgt):
        src_mask, tgt_self_mask, tgt_cross_mask = self.generate_mask(src, tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_self_mask, tgt_cross_mask)
        output = self.fc(dec_output)
        return output


# -------------------------- 3. 训练与评估函数 --------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        pred = model(src, tgt_input)

        loss = criterion(
            pred.reshape(-1, pred.size(-1)),
            tgt_output.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * src.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device, tgt_vocab):
    model.eval()
    total_loss = 0.0
    bleu_scores = []
    smoothing = SmoothingFunction().method4
    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            pred = model(src, tgt_input)
            loss = criterion(
                pred.reshape(-1, pred.size(-1)),
                tgt_output.reshape(-1)
            )
            total_loss += loss.item() * src.size(0)

            preds = []
            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0)
                pred_tokens = []
                current_tgt = torch.tensor([tgt_vocab["<SOS>"]], device=device).unsqueeze(0)

                for _ in range(len(tgt[i])):
                    pred_i = model(src_i, current_tgt)
                    next_token = pred_i[:, -1, :].argmax(dim=-1)
                    pred_tokens.append(next_token.item())
                    current_tgt = torch.cat([current_tgt, next_token.unsqueeze(0)], dim=1)

                    if next_token.item() == tgt_vocab["<EOS>"]:
                        break

                preds.append(pred_tokens)

            for pred_tokens, tgt_full in zip(preds, tgt.cpu().numpy()):
                tgt_tokens = [t for t in tgt_full[1:] if t != 0 and t != tgt_vocab["<EOS>"]]
                pred_tokens = [t for t in pred_tokens if t != 0 and t != tgt_vocab["<EOS>"]]
                tgt_chars = [tgt_vocab_inv[t] for t in tgt_tokens]
                pred_chars = [tgt_vocab_inv.get(t, "<UNK>") for t in pred_tokens]
                if len(tgt_chars) == 0 or len(pred_chars) == 0:
                    bleu = 0.0
                else:
                    bleu = sentence_bleu([tgt_chars], pred_chars, smoothing_function=smoothing)
                bleu_scores.append(bleu)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    return avg_loss, avg_bleu


# -------------------------- 4. 实验配置 --------------------------
def run_experiment(experiment_name, model_kwargs, train_loader, dev_loader, test_loader,
                   src_vocab, tgt_vocab, epochs=100, model_save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment: {experiment_name}, Device: {device}")

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab), **model_kwargs
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_losses = []
    dev_losses = []
    dev_bleus = []

    # 记录实验开始时间
    start_time = time.time()

    for epoch in range(epochs):
        # 记录每个epoch开始时间
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_bleu = evaluate(model, dev_loader, criterion, device, tgt_vocab)

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        dev_bleus.append(dev_bleu)

        # 计算epoch耗时
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {dev_loss:.4f} | Validation BLEU: {dev_bleu:.4f} | Time: {epoch_time:.2f}s")

    # 计算总实验耗时
    total_time = time.time() - start_time
    print(f"Total time for {experiment_name}: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    test_loss, test_bleu = evaluate(model, test_loader, criterion, device, tgt_vocab)
    print(f"Experiment {experiment_name} Test Results | Test Loss: {test_loss:.4f} | Test BLEU: {test_bleu:.4f}")

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    return {
        "name": experiment_name,
        "model": model,
        "train_losses": train_losses,
        "dev_losses": dev_losses,
        "dev_bleus": dev_bleus,
        "test_loss": test_loss,
        "test_bleu": test_bleu,
        "total_time": total_time  # 保存实验总时长
    }


# -------------------------- 5. 主函数 --------------------------
if __name__ == "__main__":
    # 记录整个程序开始时间
    overall_start = time.time()

    # 配置参数
    train_file = "../data/train.txt"
    dev_file = "../data/dev.txt"
    test_file = "../data/test.txt"

    max_seq_len = 20
    batch_size = 8
    epochs = 100

    # 创建模型和结果保存目录
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 加载数据并计时
    print("Loading dataset...")
    data_start = time.time()
    train_pairs = read_parallel_corpus(train_file)
    dev_pairs = read_parallel_corpus(dev_file)
    test_pairs = read_parallel_corpus(test_file)
    data_time = time.time() - data_start
    print(f"Dataset loaded in {data_time:.2f} seconds")

    print(f"Dataset size: Train {len(train_pairs)} | Dev {len(dev_pairs)} | Test {len(test_pairs)}")

    # 构建词汇表并计时
    vocab_start = time.time()
    all_en_sentences = [pair[0] for pair in train_pairs + dev_pairs + test_pairs]
    all_zh_sentences = [list(pair[1]) for pair in train_pairs + dev_pairs + test_pairs]

    src_vocab = build_vocab(all_en_sentences)
    tgt_vocab = build_vocab(all_zh_sentences)
    tgt_vocab["<SOS>"] = len(tgt_vocab)
    tgt_vocab["<EOS>"] = len(tgt_vocab)
    vocab_time = time.time() - vocab_start
    print(f"Vocabulary built in {vocab_time:.2f} seconds")

    print(f"Vocabulary size: English {len(src_vocab)} | Chinese {len(tgt_vocab)}")

    # 创建数据集和数据加载器
    dataset_start = time.time()
    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab, max_seq_len)
    dev_dataset = TranslationDataset(dev_pairs, src_vocab, tgt_vocab, max_seq_len)
    test_dataset = TranslationDataset(test_pairs, src_vocab, tgt_vocab, max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataset_time = time.time() - dataset_start
    print(f"Dataloaders created in {dataset_time:.2f} seconds")

    # 消融实验配置
    experiments = [
        {
            "name": "Baseline",
            "kwargs": {
                "d_model": 128, "n_heads": 4, "num_encoder_layers": 2,
                "num_decoder_layers": 2, "d_ff": 512, "dropout": 0.1,
                "use_pos_encoding": True, "use_layer_norm": True
            },
            "save_path": os.path.join(model_dir, "baseline_model.pt")
        },
        {
            "name": "Without Positional Encoding",
            "kwargs": {
                "d_model": 128, "n_heads": 4, "num_encoder_layers": 2,
                "num_decoder_layers": 2, "d_ff": 512, "dropout": 0.1,
                "use_pos_encoding": False, "use_layer_norm": True
            },
            "save_path": os.path.join(model_dir, "no_pos_encoding_model.pt")
        }
    ]

    # 运行所有实验
    print("\nStarting ablation experiments...")
    results = []
    for exp in experiments:
        print(f"\n{'=' * 50}\nStarting experiment: {exp['name']}")
        result = run_experiment(
            experiment_name=exp["name"],
            model_kwargs=exp["kwargs"],
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            epochs=epochs,
            model_save_path=exp["save_path"]
        )
        results.append(result)
        print(f"Experiment completed: {exp['name']}\n{'=' * 50}")

    # 结果可视化并计时
    vis_start = time.time()
    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(res["train_losses"], label=res["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curves for Different Experiments")
    plt.legend()
    plt.savefig("results/train_loss.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(res["dev_losses"], label=res["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Curves for Different Experiments")
    plt.legend()
    plt.savefig("results/dev_loss.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(res["dev_bleus"], label=res["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation BLEU Score")
    plt.title("BLEU Score Curves for Different Experiments")
    plt.legend()
    plt.savefig("results/dev_bleu.png")
    plt.close()
    vis_time = time.time() - vis_start
    print(f"Visualizations generated in {vis_time:.2f} seconds")

    # 测试集结果对比
    print("\nTest Set Final Results Comparison:")
    print("-" * 90)
    print(f"{'Experiment Name':<30} | Test Loss | Test BLEU | Time (s) | Time (min)")
    print("-" * 90)
    for res in results:
        print(
            f"{res['name']:<30} | {res['test_loss']:.4f} | {res['test_bleu']:.4f} | {res['total_time']:.2f} | {res['total_time'] / 60:.2f}")
    print("-" * 90)

    # 计算总耗时
    overall_time = time.time() - overall_start
    print(f"\nAll processes completed. Total time: {overall_time:.2f} seconds ({overall_time / 60:.2f} minutes)")