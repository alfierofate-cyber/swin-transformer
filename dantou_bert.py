import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) 配置
# -----------------------------
@dataclass#[可配置]

class TinyBertConfig:#配置类
    vocab_size: int = 30522#词表大小
    hidden_size: int = 256 # 隐藏层维度
    max_position_embeddings: int = 512#最大长度
    type_vocab_size: int = 2# token类型数量
    intermediate_size: int = 4 * 256#中间层维度
    hidden_dropout_prob: float = 0.1#隐藏层dropout概率
    attention_probs_dropout_prob: float = 0.1#注意力dropout概率
    layer_norm_eps: float = 1e-12#层归一化的epsilon
    num_labels: int = 2             # 分类任务用›


# -----------------------------
# 2) Embeddings: token + position + token_type
# -----------------------------
class BertEmbeddingsLite(nn.Module):
    def __init__(self, cfg: TinyBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)

        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        # input_ids: (B, L)
        B, L = input_ids.shape
        device = input_ids.device

        if token_type_ids is None:
            token_type_ids = torch.zeros((B, L), dtype=torch.long, device=device)

        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, L)

        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x  # (B, L, H)


# -----------------------------
# 3) 单头自注意力
# -----------------------------
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, cfg: TinyBertConfig):
        super().__init__()
        H = cfg.hidden_size
        # 单头：head_dim = H
        self.q = nn.Linear(H, H)
        self.k = nn.Linear(H, H)
        self.v = nn.Linear(H, H)
        self.out = nn.Linear(H, H)

        self.attn_dropout = nn.Dropout(cfg.attention_probs_dropout_prob)
        self.proj_dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x, attention_mask=None):
        """
        x: (B, L, H)
        attention_mask: (B, L) 其中 1 表示可见，0 表示 padding
        """
        B, L, H = x.shape
        q = self.q(x)  # (B, L, H)
        k = self.k(x)  # (B, L, H)
        v = self.v(x)  # (B, L, H)

        # 注意力分数: (B, L, L)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(H)

        if attention_mask is not None:
            # attention_mask: (B, L) -> (B, 1, L)
            mask = attention_mask.unsqueeze(1)
            # 把 padding 的位置加上 -inf，softmax 后趋近 0
            scores = scores.masked_fill(mask == 0, float("-inf"))

        probs = F.softmax(scores, dim=-1)  # (B, L, L)
        probs = self.attn_dropout(probs)

        ctx = torch.matmul(probs, v)  # (B, L, H)
        out = self.out(ctx)
        out = self.proj_dropout(out)
        return out  # (B, L, H)


# -----------------------------
# 4) 单层 Transformer block（BERT 风格）
# -----------------------------
class BertLayerLite(nn.Module):
    def __init__(self, cfg: TinyBertConfig):
        super().__init__()
        self.attn = SingleHeadSelfAttention(cfg)
        self.attn_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        self.ffn1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.ffn2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.ffn_dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.ffn_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x, attention_mask=None):
        # Self-attention + Residual + LN
        attn_out = self.attn(x, attention_mask=attention_mask)
        x = self.attn_ln(x + attn_out)

        # FFN + Residual + LN (GELU)
        y = self.ffn1(x)
        y = F.gelu(y)
        y = self.ffn2(y)
        y = self.ffn_dropout(y)
        x = self.ffn_ln(x + y)
        return x


# -----------------------------
# 5) 单层单头 BERT 主体
# -----------------------------
class TinyBertModel(nn.Module):
    def __init__(self, cfg: TinyBertConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = BertEmbeddingsLite(cfg)
        self.encoder_layer = BertLayerLite(cfg)  # 只有一层

        # 可选 pooler：取 CLS 向量做一个 tanh 投影
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        input_ids: (B, L)
        token_type_ids: (B, L)
        attention_mask: (B, L) 1=可见, 0=padding
        """
        x = self.embeddings(input_ids, token_type_ids=token_type_ids)
        x = self.encoder_layer(x, attention_mask=attention_mask)

        # sequence output: (B, L, H)
        sequence_output = x

        # pooled output: (B, H) 取 CLS (位置 0)
        cls = sequence_output[:, 0]
        pooled_output = torch.tanh(self.pooler(cls))
        return sequence_output, pooled_output


# -----------------------------
# 6) 示例：做句子分类
# -----------------------------
class TinyBertForSequenceClassification(nn.Module):
    def __init__(self, cfg: TinyBertConfig):
        super().__init__()
        self.bert = TinyBertModel(cfg)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_labels)

        if labels is None:
            return logits

        loss = F.cross_entropy(logits, labels)
        return loss, logits


# -----------------------------
# 7) 快速跑一下
# -----------------------------
if __name__ == "__main__":
    cfg = TinyBertConfig(hidden_size=256, intermediate_size=1024, num_labels=2)
    model = TinyBertForSequenceClassification(cfg)

    B, L = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, L))
    token_type_ids = torch.zeros((B, L), dtype=torch.long)
    attention_mask = torch.ones((B, L), dtype=torch.long)
    labels = torch.tensor([0, 1], dtype=torch.long)

    loss, logits = model(input_ids, token_type_ids, attention_mask, labels)
    print("loss:", float(loss), "logits:", logits.shape)
