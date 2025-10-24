import torch
import re
import os
from PIL import Image

# -------------------------- 配置与初始化 --------------------------
# 模型路径
MODEL_PATH = "../models/baseline_model.pt"
SRC_VOCAB_PATH = None
TGT_VOCAB_PATH = None

# 模型参数
MODEL_KWARGS = {
    "d_model": 128,
    "n_heads": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "d_ff": 512,
    "dropout": 0.1,
    "use_pos_encoding": True,
    "use_layer_norm": True
}

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_english_punctuation(text):
    """分离英文标点与单词"""
    punctuation = r"([,.?!;:\"'])"
    text = re.sub(punctuation, r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(sentences, min_freq=1):
    """构建词汇表"""
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


# -------------------------- 加载词汇表 --------------------------
def load_vocab(train_file, dev_file, test_file):
    """加载或构建与训练时完全一致的词汇表"""

    # 读取训练/验证/测试集以构建相同的词汇表
    def read_corpus(file_path):
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    # 加载所有语料以保证词汇表一致性
    train_pairs = read_corpus(train_file)
    dev_pairs = read_corpus(dev_file)
    test_pairs = read_corpus(test_file)

    all_en_sentences = [split_english_punctuation(pair[0]) for pair in train_pairs + dev_pairs + test_pairs]
    all_zh_sentences = [list(pair[1]) for pair in train_pairs + dev_pairs + test_pairs]

    # 构建词汇表
    src_vocab = build_vocab(all_en_sentences)
    tgt_vocab = build_vocab(all_zh_sentences)
    tgt_vocab["<SOS>"] = len(tgt_vocab)
    tgt_vocab["<EOS>"] = len(tgt_vocab)

    return src_vocab, tgt_vocab


# -------------------------- Transformer模型定义--------------------------
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

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


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        cross_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(cross_output))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_heads=4,
                 num_encoder_layers=2, num_decoder_layers=2, d_ff=512, dropout=0.1,
                 use_pos_encoding=True, use_layer_norm=True):
        super().__init__()
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding
        self.use_layer_norm = use_layer_norm

        self.src_embedding = torch.nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = torch.nn.Embedding(tgt_vocab_size, d_model)

        if self.use_pos_encoding:
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.encoder_layers = torch.nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = torch.nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.fc = torch.nn.Linear(d_model, tgt_vocab_size)

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
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, tgt_self_mask, tgt_cross_mask):
        x = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        if self.use_pos_encoding:
            x = self.pos_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_self_mask, tgt_cross_mask)
        return x

    def forward(self, src, tgt):
        src_mask, tgt_self_mask, tgt_cross_mask = self.generate_mask(src, tgt)
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_self_mask, tgt_cross_mask)
        output = self.fc(dec_output)
        return output


# -------------------------- 翻译函数 --------------------------
def translate(sentence, model, src_vocab, tgt_vocab, max_len=20):
    """
    将英文句子翻译成中文
    :param sentence: 输入的英文句子（字符串）
    :param model: 加载好的Transformer模型
    :param src_vocab: 英文词汇表
    :param tgt_vocab: 中文词汇表
    :param max_len: 最大翻译长度
    :return: 翻译后的中文句子
    """
    # 预处理输入句子
    processed_sentence = split_english_punctuation(sentence)
    src_tokens = processed_sentence.split()

    # 转换为token ID
    src_ids = [src_vocab.get(token, src_vocab["<UNK>"]) for token in src_tokens]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)  # 增加batch维度

    # 初始化目标序列（从<SOS>开始）
    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}
    tgt_ids = [tgt_vocab["<SOS>"]]
    current_tgt = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        # 编码源句子
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
        enc_output = model.encode(src_tensor, src_mask)

        # 解码生成目标句子
        for _ in range(max_len):
            tgt_self_mask = torch.tril(torch.ones((current_tgt.size(1), current_tgt.size(1)), device=device)).unsqueeze(
                0).unsqueeze(0)
            tgt_pad_mask = (current_tgt != 0).unsqueeze(1).unsqueeze(2)
            tgt_self_mask = tgt_self_mask.bool() & tgt_pad_mask.bool()

            dec_output = model.decode(current_tgt, enc_output, tgt_self_mask, src_mask)
            next_token_logits = model.fc(dec_output[:, -1, :])
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            tgt_ids.append(next_token_id)
            current_tgt = torch.cat([current_tgt, torch.tensor([[next_token_id]], device=device)], dim=1)

            # 遇到<EOS>停止生成
            if next_token_id == tgt_vocab["<EOS>"]:
                break

    # 转换为中文句子（过滤<SOS>和<EOS>）
    translated_tokens = [tgt_vocab_inv[id_] for id_ in tgt_ids if
                         id_ not in [tgt_vocab["<SOS>"], tgt_vocab["<EOS>"], tgt_vocab["<PAD>"]]]
    return ''.join(translated_tokens)


# -------------------------- 主函数：加载模型并翻译 --------------------------
def main():
    # 1. 配置数据路径
    train_file = "../data/train.txt"
    dev_file = "../data/dev.txt"
    test_file = "../data/test.txt"

    # 2. 加载词汇表
    print("Loading vocabularies...")
    src_vocab, tgt_vocab = load_vocab(train_file, dev_file, test_file)
    print(f"Vocabularies loaded. English size: {len(src_vocab)}, Chinese size: {len(tgt_vocab)}")

    # 3. 初始化模型并加载权重
    print("Loading model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        **MODEL_KWARGS
    ).to(device)

    # 加载训练好的权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # 4. 测试翻译
    test_sentences = [
        "Have fun.",
        "I love learning.",
        "What is your name?",
        "She really does like animals.",
        "It's a nice day, isn't it?",
        "Today is a good day."
    ]

    print("\nStarting translation test...")
    for eng in test_sentences:
        chinese = translate(eng, model, src_vocab, tgt_vocab)
        print(f"English: {eng}")
        print(f"Chinese: {chinese}\n")

    # 5. 交互式翻译
    print("Enter 'q' to quit.")
    while True:
        user_input = input("English: ")
        if user_input.lower() == 'q':
            break
        chinese = translate(user_input, model, src_vocab, tgt_vocab)
        print(f"Chinese: {chinese}\n")


if __name__ == "__main__":
    main()