"""
DTransformer (No Diagnostic Knowledge States) + Frequency Decoupling Two-Stream + Neighbor-Decay Attention
========================================================================================================

This file implements a **no-diagnostic** version of DTransformer that keeps the original backbone
(embedding + masked multi-head attention with time-effect gamma + mask-dropout regularization),
but **removes** the diagnostic knowledge states (know_params / n_know slots / block4 query / readout).

On top of this no-diagnostic backbone, it implements the three innovations you requested:

1) **Frequency-domain learnable decoupling (stable/random)**
   - Treat the interaction embedding sequence as a signal.
   - Learn a frequency-domain (complex) response for a causal FIR filter.
   - Use iFFT to obtain a time-domain kernel and apply **depthwise causal conv1d**.
   - stable = low-pass-like filtered signal; random = residual (signal - stable).

2) **Two-stream latent knowledge representation (stable/random)**
   - Feed stable and random streams into two (lightweight) Transformer branches.
   - Fuse them with a learnable gate per timestep:
        z = g * z_stable + (1 - g) * z_random

3) **Neighbor-decay attention on the random branch**
   - Add a learnable per-head decay term to attention logits:
        score_ij <- score_ij - delta * |i - j|
   - This encourages the random stream to focus on recent interactions.

Notes:
- Causality / no label leakage is maintained by setting `peek_cur=False` on the final cross-attention
  used for prediction (same philosophy as AKT/SAKT). The model predicts s_t using only interactions < t.
- `n_know` is kept in the constructor for config compatibility, but is unused.

Author: generated for Aerfh Fxgc
"""

import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimum sequence length: skip contrastive learning if sequence too short
MIN_SEQ_LEN = 5


# ============================================================
# 1) Frequency-domain learnable causal filter
# ============================================================
class LearnableFreqCausalFilter(nn.Module):
    """
    Learn a causal depthwise FIR filter via frequency-domain parameters.

    - We learn half-spectrum complex coefficients H (per channel).
    - Use irfft to obtain a real time-domain kernel h of length K.
    - Apply h with depthwise causal conv1d (groups = d_model).

    Input/Output:
        x: (bs, T, d_model)
        y: (bs, T, d_model)
    """

    def __init__(self, d_model: int, kernel_size: int = 32, l1_normalize: bool = True):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.l1_normalize = l1_normalize

        n_freq = kernel_size // 2 + 1
        self.freq_real = nn.Parameter(torch.zeros(d_model, n_freq))
        self.freq_imag = nn.Parameter(torch.zeros(d_model, n_freq))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize as a simple moving-average smoother.
        This makes 'stable' start as a low-pass-like trend extractor.
        """
        h = torch.ones(self.kernel_size) / float(self.kernel_size)  # (K,)
        H = torch.fft.rfft(h)  # (K//2+1,) complex
        with torch.no_grad():
            self.freq_real.copy_(H.real[None, :].repeat(self.d_model, 1))
            self.freq_imag.copy_(H.imag[None, :].repeat(self.d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (bs, T, d_model)

        Returns:
            y: (bs, T, d_model)
        """
        bs, T, d = x.shape
        if d != self.d_model:
            raise ValueError(f"Expected last dim={self.d_model}, got {d}")

        # complex half-spectrum
        H = torch.complex(self.freq_real, self.freq_imag)  # (d_model, n_freq)
        # time-domain kernel
        kernel = torch.fft.irfft(H, n=self.kernel_size)  # (d_model, K)

        # Normalize for stability (optional)
        if self.l1_normalize:
            kernel = kernel / (kernel.abs().sum(dim=-1, keepdim=True) + 1e-8)

        # Depthwise causal conv:
        # y[t] = sum_{k=0..K-1} h[k] * x[t-k]
        weight = kernel.unsqueeze(1)  # (d_model, 1, K)

        xt = x.transpose(1, 2)  # (bs, d_model, T)
        xt = F.pad(xt, (self.kernel_size - 1, 0))  # left pad
        yt = F.conv1d(xt, weight, groups=self.d_model)  # (bs, d_model, T)

        return yt.transpose(1, 2)  # (bs, T, d_model)


# ============================================================
# 2) DTransformer backbone (no diagnostic knowledge states)
#    + 3) neighbor-decay attention (random stream)
# ============================================================
class DTransformer(nn.Module):
    """
    No-diagnostic DTransformer with:
      - frequency learnable decoupling (stable/random),
      - two-stream tracking,
      - neighbor-decay attention in random stream.

    Constructor args follow the original DTransformer for compatibility.
    """

    def __init__(
        self,
        n_questions: int,
        n_pid: int = 0,
        d_model: int = 128,
        d_fc: int = 256,
        n_heads: int = 8,
        n_know: int = 16,        # kept for compatibility, unused
        n_layers: int = 1,
        dropout: float = 0.05,
        lambda_cl: float = 0.1,
        proj: bool = False,
        hard_neg: bool = True,
        window: int = 1,
        shortcut: bool = False,
        # --------- new knobs for ablations ----------
        use_freq_decoupling: bool = True,
        freq_kernel_size: int = 32,
        use_two_stream: bool = True,
        use_neighbor_decay: bool = True,
        gate_hidden: Optional[int] = None,
        return_all_attn: bool = False,
    ):
        super().__init__()

        self.n_questions = n_questions
        self.n_pid = n_pid
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.lambda_cl = lambda_cl
        self.hard_neg = hard_neg
        self.window = window
        self.shortcut = shortcut
        self.n_know = n_know  # unused

        self.use_freq_decoupling = use_freq_decoupling
        self.use_two_stream = use_two_stream
        self.use_neighbor_decay = use_neighbor_decay
        self.return_all_attn = return_all_attn

        # ============ Embeddings ============
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)

        # ============ Difficulty-aware FiLM (optional) ============
        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.p_diff_embed = nn.Embedding(n_pid + 1, d_model)
            self.p_film_gamma = nn.Linear(d_model, d_model)
            self.p_film_beta = nn.Linear(d_model, d_model)

        # ============ (1) Frequency decoupling ============
        if self.use_freq_decoupling:
            self.decouple_norm = nn.LayerNorm(d_model)
            self.freq_filter = LearnableFreqCausalFilter(d_model, kernel_size=freq_kernel_size)
        else:
            self.decouple_norm = None
            self.freq_filter = None

        # ============ Transformer blocks ============
        # Shared question encoder (used when shortcut=True or n_layers>=3)
        self.q_block = DTransformerLayer(d_model, n_heads, dropout, neighbor_decay=False)

        # Stable stream blocks
        self.s_block1 = DTransformerLayer(d_model, n_heads, dropout, neighbor_decay=False)
        self.s_block2 = DTransformerLayer(d_model, n_heads, dropout, neighbor_decay=False)
        self.s_block3 = DTransformerLayer(d_model, n_heads, dropout, neighbor_decay=False)

        # Random stream blocks (neighbor decay enabled)
        self.r_block1 = DTransformerLayer(d_model, n_heads, dropout, neighbor_decay=use_neighbor_decay)
        self.r_block2 = DTransformerLayer(d_model, n_heads, dropout, neighbor_decay=use_neighbor_decay)
        self.r_block3 = DTransformerLayer(d_model, n_heads, dropout, neighbor_decay=use_neighbor_decay)

        # ============ (2) Two-stream fusion gate ============
        if gate_hidden is None:
            gate_hidden = d_model

        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )

        # ============ Output head ============
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, d_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc // 2, 1),
        )

        # ============ Contrastive learning projection (optional) ============
        if proj:
            self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.proj = None

    # ------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------
    def embedding(
        self, q: torch.Tensor, s: torch.Tensor, pid: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (bs, T), padded with -1
            s: (bs, T), 0/1, padded with -1
            pid: (bs, T), padded with -1

        Returns:
            q_emb: (bs, T, d_model)
            s_emb: (bs, T, d_model)
            lens:  (bs,)
            p_diff: (bs, T, d_model) or 0.0 tensor
        """
        lens = (s >= 0).sum(dim=1)

        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)

        q_emb = self.q_embed(q)
        s_emb = self.s_embed(s) + q_emb

        p_diff = torch.tensor(0.0, device=q.device)

        if pid is not None and hasattr(self, "p_diff_embed"):
            pid = pid.masked_fill(pid < 0, 0)
            p_diff = self.p_diff_embed(pid)

            gamma = 1.0 + torch.tanh(self.p_film_gamma(p_diff))
            beta = self.p_film_beta(p_diff)

            q_diff_emb = self.q_diff_embed(q)
            q_emb = q_emb + (q_diff_emb * gamma + beta)

            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb = s_emb + (s_diff_emb * gamma + beta)

        return q_emb, s_emb, lens, p_diff

    # ------------------------------------------------------------
    # (1) Frequency decoupling: stable/random
    # ------------------------------------------------------------
    def _decouple(self, s_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_freq_decoupling:
            return s_emb, torch.zeros_like(s_emb)

        x = self.decouple_norm(s_emb)  # normalize first for stability
        stable = self.freq_filter(x)
        random_part = x - stable
        return stable, random_part

    # ------------------------------------------------------------
    # Branch forward (stable or random)
    # ------------------------------------------------------------
    def _forward_branch(
        self, q_emb: torch.Tensor, s_emb: torch.Tensor, lens: torch.Tensor, branch: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z: (bs, T, d_model)
            q_scores: attention weights (for analysis)
        """
        if branch == "stable":
            b1, b2, b3 = self.s_block1, self.s_block2, self.s_block3
        elif branch == "random":
            b1, b2, b3 = self.r_block1, self.r_block2, self.r_block3
        else:
            raise ValueError("branch must be 'stable' or 'random'")

        # Shortcut path (AKT-like)
        if self.shortcut:
            hq, _ = self.q_block(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, _ = b2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            z, q_scores = b3(hq, hq, hs, lens, peek_cur=False)
            return z, q_scores

        # Original DTransformer selectable depth (but WITHOUT diagnostic states)
        if self.n_layers == 1:
            z, q_scores = b1(q_emb, q_emb, s_emb, lens, peek_cur=False)
            return z, q_scores

        if self.n_layers == 2:
            hs, _ = b1(s_emb, s_emb, s_emb, lens, peek_cur=True)
            z, q_scores = b2(q_emb, q_emb, hs, lens, peek_cur=False)
            return z, q_scores

        # n_layers >= 3
        hq, _ = self.q_block(q_emb, q_emb, q_emb, lens, peek_cur=True)
        hs, _ = b2(s_emb, s_emb, s_emb, lens, peek_cur=True)
        z, q_scores = b3(hq, hq, hs, lens, peek_cur=False)
        return z, q_scores

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, q_emb: torch.Tensor, s_emb: torch.Tensor, lens: torch.Tensor):
        """
        Args:
            q_emb: (bs, T, d_model)
            s_emb: (bs, T, d_model)
            lens:  (bs,)

        Returns:
            z: (bs, T, d_model)
            q_scores: attention weights (tensor) or (stable, random) tuple if return_all_attn=True
            k_scores: None (kept for compatibility)
        """
        if not self.use_two_stream:
            z, q_scores = self._forward_branch(q_emb, s_emb, lens, branch="stable")
            return z, q_scores, None

        s_stable, s_random = self._decouple(s_emb)
        z_s, q_scores_s = self._forward_branch(q_emb, s_stable, lens, branch="stable")
        z_r, q_scores_r = self._forward_branch(q_emb, s_random, lens, branch="random")

        g = torch.sigmoid(self.gate_net(torch.cat([z_s, z_r], dim=-1)))  # (bs, T, 1)
        z = g * z_s + (1.0 - g) * z_r

        if self.return_all_attn:
            return z, (q_scores_s, q_scores_r), None
        return z, q_scores_s, None

    # ------------------------------------------------------------
    # Prediction & Loss
    # ------------------------------------------------------------
    def predict(self, q: torch.Tensor, s: torch.Tensor, pid: Optional[torch.Tensor] = None, n: int = 1):
        """
        Predict logits for each timestep.

        Note:
            This no-diagnostic version supports only n=1 (standard KT alignment).
        """
        if n != 1:
            raise ValueError("No-diagnostic two-stream model currently supports only n=1.")

        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid)
        z, q_scores, _ = self(q_emb, s_emb, lens)

        query = q_emb  # (bs, T, d_model)
        h = z          # (bs, T, d_model)

        y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

        # Regularization (only when FiLM is enabled)
        if pid is not None and isinstance(p_diff, torch.Tensor) and p_diff.numel() > 1:
            reg_loss = (p_diff ** 2).mean() * 1e-3
        else:
            reg_loss = torch.tensor(0.0, device=q.device)

        return y, z, q_emb, reg_loss, (q_scores, None)

    def get_loss(self, q: torch.Tensor, s: torch.Tensor, pid: Optional[torch.Tensor] = None):
        logits, _, _, reg_loss, _ = self.predict(q, s, pid)

        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]

        return F.binary_cross_entropy_with_logits(masked_logits, masked_labels, reduction="mean") + reg_loss

    # ------------------------------------------------------------
    # Contrastive Learning (kept, adapted to d_model hidden states)
    # ------------------------------------------------------------
    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity matrix across batch and time.

        Args:
            z1: (bs, T, d_model)
            z2: (bs, T, d_model)

        Returns:
            (bs, bs, T)
        """
        z1 = z1.unsqueeze(1)  # (bs, 1, T, d)
        z2 = z2.unsqueeze(0)  # (1, bs, T, d)

        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)

        return F.cosine_similarity(z1, z2, dim=-1) / 0.05

    def get_cl_loss(self, q: torch.Tensor, s: torch.Tensor, pid: Optional[torch.Tensor] = None):
        """
        Returns:
            total_loss, pred_loss, cl_loss
        """
        bs = s.size(0)

        lens = (s >= 0).sum(dim=1)
        minlen = int(lens.min().item())
        if minlen < MIN_SEQ_LEN:
            return self.get_loss(q, s, pid)

        # --------- positive augmentation: swap adjacent positions ---------
        q_pos = q.clone()
        s_pos = s.clone()
        pid_pos = pid.clone() if pid is not None else None

        for b in range(bs):
            idx = random.sample(
                range(int(lens[b]) - 1), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                q_pos[b, i], q_pos[b, i + 1] = q_pos[b, i + 1], q_pos[b, i]
                s_pos[b, i], s_pos[b, i + 1] = s_pos[b, i + 1], s_pos[b, i]
                if pid_pos is not None:
                    pid_pos[b, i], pid_pos[b, i + 1] = pid_pos[b, i + 1], pid_pos[b, i]

        # --------- hard negative: flip responses ---------
        s_flip = s.clone() if self.hard_neg else s_pos
        for b in range(bs):
            idx = random.sample(
                range(int(lens[b])), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]
        if not self.hard_neg:
            s_pos = s_flip

        # --------- forward passes ---------
        logits, z1, q_emb, reg_loss, _ = self.predict(q, s, pid)
        masked_logits = logits[s >= 0]

        _, z2, *_ = self.predict(q_pos, s_pos, pid_pos)

        if self.hard_neg:
            _, z3, *_ = self.predict(q, s_flip, pid)

        # --------- contrastive loss ---------
        sim_pos = self.sim(z1[:, :minlen, :], z2[:, :minlen, :])  # (bs, bs, minlen)
        sim_all = sim_pos
        if self.hard_neg:
            sim_neg = self.sim(z1[:, :minlen, :], z3[:, :minlen, :])
            sim_all = torch.cat([sim_pos, sim_neg], dim=1)  # (bs, 2*bs, minlen)

        target = torch.arange(bs, device=q.device)[:, None].expand(-1, minlen)
        cl_loss = F.cross_entropy(sim_all, target)

        # --------- prediction loss ---------
        masked_labels = s[s >= 0].float()
        pred_loss = F.binary_cross_entropy_with_logits(masked_logits, masked_labels, reduction="mean")

        # Multi-window prediction (optional)
        for i in range(1, self.window):
            label = s[:, i:]
            if label.numel() == 0:
                continue
            query_i = q_emb[:, i:, :]
            h_i = z1[:, i:, :]
            y_i = self.out(torch.cat([query_i, h_i], dim=-1)).squeeze(-1)

            pred_loss = pred_loss + F.binary_cross_entropy_with_logits(
                y_i[label >= 0], label[label >= 0].float()
            )

        pred_loss = pred_loss / float(self.window)

        return pred_loss + cl_loss * self.lambda_cl + reg_loss, pred_loss, cl_loss

    # ------------------------------------------------------------
    # Simple tracing helper (no knowledge-slot tracing)
    # ------------------------------------------------------------
    @torch.no_grad()
    def tracing(self, q_1d: torch.Tensor, s_1d: torch.Tensor, pid_1d: Optional[torch.Tensor] = None):
        """
        For a single student sequence (1D tensors), return predicted probabilities over time.
        """
        q = q_1d.unsqueeze(0)
        s = s_1d.unsqueeze(0)
        pid = pid_1d.unsqueeze(0) if pid_1d is not None else None
        logits, *_ = self.predict(q, s, pid)
        return torch.sigmoid(logits.squeeze(0))


class DTransformerLayer(nn.Module):
    """
    One masked-attention layer from the original DTransformer codebase.

    Includes:
      - causal mask (peek_cur controls whether the diagonal is allowed),
      - training-time mask dropout (same as original),
      - residual + layer norm.

    neighbor_decay:
      If True, its internal attention uses the extra neighbor-decay term.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, kq_same: bool = True, neighbor_decay: bool = False):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same, neighbor_decay=neighbor_decay)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device

    def forward(self, query: torch.Tensor, key: torch.Tensor, values: torch.Tensor, lens: torch.Tensor, peek_cur: bool = False):
        seqlen = query.size(1)

        # causal mask: tril(0) includes diagonal, tril(-1) excludes diagonal
        mask = torch.ones(seqlen, seqlen, device=self.device()).tril(0 if peek_cur else -1).bool()[None, None, :, :]

        # training-time stochastic edge dropout in the causal mask
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()
            for b in range(query.size(0)):
                if lens[b] < MIN_SEQ_LEN:
                    continue
                idx = random.sample(
                    range(int(lens[b]) - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0

        query_attn, scores = self.masked_attn_head(query, key, values, mask, maxout=not peek_cur)
        query = query + self.dropout(query_attn)
        return self.layer_norm(query), scores


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with:
      - time-effect gamma (original DTransformer),
      - optional neighbor-decay delta (random stream).
    """

    def __init__(self, d_model: int, n_heads: int, kq_same: bool = True, bias: bool = True, neighbor_decay: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Time-effect parameters: one gamma per head
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        # (3) Neighbor-decay parameters: one delta per head
        self.neighbor_decay = neighbor_decay
        if neighbor_decay:
            self.deltas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            nn.init.constant_(self.deltas, 0.1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, maxout: bool = False):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        delta = self.deltas if self.neighbor_decay else None

        v_out, scores = attention(q, k, v, mask, gamma=self.gammas, delta=delta, maxout=maxout)

        concat = v_out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        out = self.out_proj(concat)
        return out, scores


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    gamma: Optional[torch.Tensor] = None,
    delta: Optional[torch.Tensor] = None,
    maxout: bool = False,
):
    """
    Scaled dot-product attention with:
      - time-effect gamma (original DTransformer),
      - optional neighbor-decay delta: score_ij <- score_ij - delta * |i-j|.

    Args:
        q,k,v: (bs, h, T, d_k)
        mask:  (bs, h, T, T) boolean
        gamma: (h, 1, 1)
        delta: (h, 1, 1)
    """
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, qlen, klen = scores.size()

    position_effect = None
    if gamma is not None or delta is not None:
        x1 = torch.arange(qlen, device=scores.device).float().unsqueeze(1).expand(qlen, klen)
        x2 = torch.arange(klen, device=scores.device).float().unsqueeze(0).expand(qlen, klen)
        position_effect = torch.abs(x1 - x2)[None, None, :, :]  # (1,1,qlen,klen)

    # (3) neighbor decay: subtract delta * distance in logit space
    if delta is not None:
        scores = scores - position_effect * delta.abs().unsqueeze(0)

    # Time effect (original DTransformer)
    if gamma is not None:
        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)

            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)

            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        gamma_eff = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma_eff).exp(), min=1e-5, max=1e5)
        scores = scores * total_effect

    # softmax
    scores = scores.masked_fill(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0.0)

    # max-out normalization (original DTransformer)
    if maxout:
        scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
        scores = scores * scale

    output = torch.matmul(scores, v)
    return output, scores
