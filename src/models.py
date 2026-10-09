"""Phase 4 neural models: TextCNN and BiLSTM (PAD id 0; masks rely on x != 0)."""
from __future__ import annotations

import torch
from torch import nn


def _emb_drop(p):
    """Embedding dropout; Identity at p=0 so default models are bit-identical to the pre-flag code."""
    return nn.Dropout(p) if p > 0 else nn.Identity()


class TextCNN(nn.Module):
    def __init__(self, vocab, n_classes, emb=128, nf=100, ks=(3, 4, 5), p=0.5, emb_dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.emb_drop = _emb_drop(emb_dropout)
        self.convs = nn.ModuleList(nn.Conv1d(emb, nf, k) for k in ks)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(nf * len(ks), n_classes)

    def forward(self, x, lengths=None):
        e = self.emb_drop(self.emb(x)).transpose(1, 2)
        h = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(h, 1)))


def _pool_classify(out, x, drop, fc):
    mask = (x != 0).unsqueeze(-1)
    mx = out.masked_fill(~mask, -1e9).max(1).values
    mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
    return fc(drop(torch.cat([mx, mean], 1)))


class BiLSTM(nn.Module):
    """Fast BiLSTM: two unidirectional LSTMs (oneDNN fused path) instead of a packed
    bidirectional one. Right-padded input; the backward LSTM sees each valid prefix
    reversed (pads stay at the end), so valid-position outputs equal the packed version."""

    def __init__(self, vocab, n_classes, emb=128, hid=128, p=0.5, emb_dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.emb_drop = _emb_drop(emb_dropout)
        self.rnn_f = nn.LSTM(emb, hid, batch_first=True)
        self.rnn_b = nn.LSTM(emb, hid, batch_first=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(hid * 4, n_classes)

    def forward(self, x, lengths):
        e = self.emb_drop(self.emb(x))
        B, T, _ = e.shape
        lengths = lengths.to(x.device)
        t = torch.arange(T, device=x.device).unsqueeze(0)
        ln = lengths.unsqueeze(1)
        rev = torch.where(t < ln, ln - 1 - t, t)  # [B,T]; involution on each row
        idx = rev.unsqueeze(-1).expand(-1, -1, e.size(-1))
        of, _ = self.rnn_f(e)
        ob, _ = self.rnn_b(e.gather(1, idx))
        ob = ob.gather(1, rev.unsqueeze(-1).expand(-1, -1, ob.size(-1)))
        return _pool_classify(torch.cat([of, ob], -1), x, self.drop, self.fc)


class BiLSTMPacked(nn.Module):
    """Standard packed bidirectional nn.LSTM (the original spec). Slow on CPU, used on CUDA."""

    def __init__(self, vocab, n_classes, emb=128, hid=128, p=0.5, emb_dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.emb_drop = _emb_drop(emb_dropout)
        self.rnn = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(hid * 4, n_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb_drop(self.emb(x)), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(-1)
        mx = out.masked_fill(~mask, -1e9).max(1).values
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(torch.cat([mx, mean], 1)))


def copy_packed_to_fast(packed, fast):
    """Copy weights from a BiLSTMPacked into a fast BiLSTM (embedding, both LSTMs, fc)."""
    with torch.no_grad():
        fast.emb.weight.copy_(packed.emb.weight)
        for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(fast.rnn_f, n + "_l0").copy_(getattr(packed.rnn, n + "_l0"))
            getattr(fast.rnn_b, n + "_l0").copy_(getattr(packed.rnn, n + "_l0_reverse"))
        fast.fc.weight.copy_(packed.fc.weight)
        fast.fc.bias.copy_(packed.fc.bias)
    return fast


def build_model(name, vocab_size, n_classes=5, emb=128, hid=128, nf=100, dropout=0.5, packed=False, emb_dropout=0.0):
    """packed=True selects the standard packed bidirectional nn.LSTM (use on CUDA); the default
    two-LSTM variant is faster on CPU. Both are numerically equivalent (see tests)."""
    if name == "cnn":
        return TextCNN(vocab_size, n_classes, emb=emb, nf=nf, p=dropout, emb_dropout=emb_dropout)
    if name == "bilstm":
        cls = BiLSTMPacked if packed else BiLSTM
        return cls(vocab_size, n_classes, emb=emb, hid=hid, p=dropout, emb_dropout=emb_dropout)
    raise ValueError(f"unknown model {name!r}")
