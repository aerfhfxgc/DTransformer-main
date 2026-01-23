"""
DTransformer: 知识追踪的诊断Transformer模型
=============================================

论文: Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer (WWW'23)

核心创新:
1. 多知识状态建模 (n_know个可学习的知识状态参数)
2. 知识状态查询机制 (block4专门用于从历史中查询知识状态)
3. 时间效应建模 (gamma参数控制时间衰减)
4. 灵活的架构配置 (支持1/2/3层编码)

主要类:
- DTransformer: 主模型类
- DTransformerLayer: Transformer层
- MultiHeadAttention: 多头注意力机制

主要函数:
- attention: 注意力计算函数（含时间效应）
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 最小序列长度，用于跳过过短序列的对比学习
MIN_SEQ_LEN = 5


class DTransformer(nn.Module):
    """
    DTransformer: 诊断Transformer模型用于知识追踪
    
    该模型通过多个可学习的知识状态参数来追踪学生的知识掌握情况，
    而非仅仅学习答题模式。
    
    核心创新:
    1. 多知识状态 (know_params): n_know个可学习向量，每个代表一个知识维度
    2. 知识查询层 (block4): 用知识状态查询历史交互信息
    3. 灵活编码: 根据n_layers选择不同的编码策略
    """
    
    def __init__(
        self,
        n_questions,      # 题目类型数量（如138种题目类型）
        n_pid=0,          # 具体题目ID数量（如1084道具体题目，用于难度建模）
        d_model=128,      # 模型隐藏维度（向量表示的维度）
        d_fc=256,         # 全连接层的中间维度
        n_heads=8,        # 多头注意力的头数
        n_know=16,        # 【核心】知识状态数量（同时追踪的知识维度数）
        n_layers=1,       # 编码层数（1/2/3，控制block1-3的使用方式）
        dropout=0.05,     # Dropout率
        lambda_cl=0.1,    # 对比学习损失的权重
        proj=False,       # 是否在对比学习前使用投影层
        hard_neg=True,    # 是否使用hard negative样本
        window=1,         # 预测窗口大小
        shortcut=False,   # 是否使用AKT模式（捷径连接）
    ):
        super().__init__()
        
        # ============ 基础嵌入层 ============
        self.n_questions = n_questions
        # 题目类型嵌入：将题目类型ID映射为d_model维向量
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        # 响应嵌入：将答题结果(0/1)映射为d_model维向量
        self.s_embed = nn.Embedding(2, d_model)

        # ============ 难度感知嵌入（可选）============
        # 当提供题目ID时，使用难度信息增强表示
        if n_pid > 0:
            # 题目类型的难度嵌入
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            # 响应的难度嵌入
            self.s_diff_embed = nn.Embedding(2, d_model)
            # 题目ID的难度值（FiLM条件向量）
            self.p_diff_embed = nn.Embedding(n_pid + 1, d_model)
            # FiLM: 为每个具体题目(pid)学习一个条件向量，再映射为逐维scale(γ)与shift(β)
            self.p_film_gamma = nn.Linear(d_model, d_model)
            self.p_film_beta = nn.Linear(d_model, d_model)

        # ============ Transformer层（Block） ============
        self.n_heads = n_heads
        # block1-3: 序列编码层（根据n_layers选择使用）
        self.block1 = DTransformerLayer(d_model, n_heads, dropout)
        self.block2 = DTransformerLayer(d_model, n_heads, dropout)
        self.block3 = DTransformerLayer(d_model, n_heads, dropout)
        # 【核心创新】block4: 知识查询层，kq_same=False允许Q和K使用不同的线性层
        self.block4 = DTransformerLayer(d_model, n_heads, dropout, kq_same=False)
        # block5: 预留扩展，当前未使用
        self.block5 = DTransformerLayer(d_model, n_heads, dropout)

        # ============ 【核心创新】知识状态参数 ============
        # n_know个可学习的知识状态查询向量，每个向量代表一个知识维度
        # 例如: 知识状态1可能学习"基础概念掌握"，知识状态2可能学习"应用能力"
        self.n_know = n_know
        self.know_params = nn.Parameter(torch.empty(n_know, d_model))
        torch.nn.init.uniform_(self.know_params, -1.0, 1.0)

        # ============ 输出层 ============
        # 3层MLP：输入是问题嵌入+知识状态(d_model*2)，输出是预测概率(1)
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, d_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc // 2, 1),
        )

        # ============ 对比学习投影层（可选）============
        if proj:
            self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.proj = None

        # ============ 超参数 ============
        self.dropout_rate = dropout
        self.lambda_cl = lambda_cl    # 对比学习损失权重
        self.hard_neg = hard_neg       # 是否使用hard negative
        self.shortcut = shortcut       # 是否使用AKT模式
        self.n_layers = n_layers       # 编码层数
        self.window = window           # 预测窗口

    def forward(self, q_emb, s_emb, lens):
        """
        前向传播：从嵌入向量计算知识状态表示
        
        参数:
            q_emb: 题目类型嵌入 (batch_size, seq_len, d_model)
            s_emb: 答题响应嵌入 (batch_size, seq_len, d_model)
            lens: 每个序列的有效长度 (batch_size,)
        
        返回:
            z: 知识状态表示 (batch_size, seq_len, n_know * d_model)
            q_scores: 题目编码的注意力分数
            k_scores: 知识查询的注意力分数
        """
        
        # ============ AKT捷径模式（特殊模式）============
        if self.shortcut:
            # AKT模式：问题和响应分别自注意力，然后交叉注意力
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, scores = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            return self.block3(hq, hq, hs, lens, peek_cur=False), scores, None

        # ============ 序列编码阶段（根据n_layers选择策略）============
        
        if self.n_layers == 1:
            # 1层: 直接用问题查询响应（交叉注意力）
            # query=q_emb, key=q_emb, values=s_emb
            hq = q_emb  # 保持原始问题嵌入
            p, q_scores = self.block1(q_emb, q_emb, s_emb, lens, peek_cur=True)
            
        elif self.n_layers == 2:
            # 2层: 先编码响应，再用问题查询
            # block1: 响应自注意力编码 (s_emb自己和自己交互)
            # block2: 用问题查询编码后的响应 (交叉注意力)
            hq = q_emb
            hs, _ = self.block1(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block2(hq, hq, hs, lens, peek_cur=True)
            
        else:  # n_layers >= 3
            # 3层: 分别编码问题和响应，再交互
            # block1: 问题自注意力编码 (q_emb自己和自己交互)
            # block2: 响应自注意力编码 (s_emb自己和自己交互)
            # block3: 问题和响应交互 (交叉注意力)
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)

        # ============ 【核心创新】知识状态查询阶段 ============
        bs, seqlen, d_model = p.size()
        n_know = self.n_know

        # 准备知识状态查询向量
        # 将know_params扩展到batch和序列长度维度
        # 形状变化: (n_know, d_model) -> (bs, n_know, seqlen, d_model) -> (bs*n_know, seqlen, d_model)
        query = (
            self.know_params[None, :, None, :]  # 增加batch和时间维度
            .expand(bs, -1, seqlen, -1)          # 扩展到所有batch和时间步
            .contiguous()
            .view(bs * n_know, seqlen, d_model)  # reshape为(bs*n_know, seqlen, d_model)
        )
        
        # 将问题表示和交互表示也扩展到n_know维度
        # 每个知识状态都会查询相同的历史信息，但用不同的query向量
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)

        # 【核心】使用block4进行知识状态查询
        # query: n_know个知识状态查询向量
        # key: 编码后的问题表示
        # values: 问题和响应的交互表示
        # peek_cur=False: 不能看到当前时间步（预测模式）
        z, k_scores = self.block4(
            query, hq, p, torch.repeat_interleave(lens, n_know), peek_cur=False
        )
        
        # 重塑知识状态表示的形状
        # (bs*n_know, seqlen, d_model) -> (bs, n_know, seqlen, d_model)
        # -> (bs, seqlen, n_know, d_model) -> (bs, seqlen, n_know*d_model)
        z = (
            z.view(bs, n_know, seqlen, d_model)  # 拆分batch和知识维度
            .transpose(1, 2)                      # 交换知识维度和时间维度
            .contiguous()
            .view(bs, seqlen, -1)                 # 展平知识状态，最终形状(bs, seqlen, n_know*d_model)
        )
        
        # 重塑知识查询的注意力分数
        k_scores = (
            k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)
            .permute(0, 2, 3, 1, 4)  # (bs, n_heads, seqlen, n_know, seqlen)
            .contiguous()
        )
        
        return z, q_scores, k_scores

    def embedding(self, q, s, pid=None):
        """
        嵌入层：将原始输入转换为向量表示
        
        参数:
            q: 题目类型ID序列 (batch_size, seq_len)
            s: 答题响应序列(0/1) (batch_size, seq_len)
            pid: 题目ID序列（可选） (batch_size, seq_len)
        
        返回:
            q_emb: 题目嵌入 (batch_size, seq_len, d_model)
            s_emb: 响应嵌入 (batch_size, seq_len, d_model)
            lens: 有效序列长度 (batch_size,)
            p_diff: pid条件向量（FiLM用；用于正则化）
        """
        # 计算每个序列的有效长度（s>=0的位置数）
        lens = (s >= 0).sum(dim=1)
        
        # 处理padding：将负值填充为0（对应嵌入层的第0个位置）
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)

        # 基础嵌入
        q_emb = self.q_embed(q)           # 题目类型 -> 向量
        s_emb = self.s_embed(s) + q_emb   # 响应嵌入 = 响应向量 + 题目向量
                                          # 这样响应嵌入就包含了"哪个题目"和"答对/答错"的信息

        p_diff = 0.0

        # ============ 难度/题目区分因子增强（可选，FiLM: scale + shift）============
        # 当提供具体题目ID(pid)时：
        # - 为每个pid学习一个条件向量 p_diff (d_model维)
        # - 由p_diff生成逐维scale(γ)与shift(β)，去调制难度方向向量（q_diff/s_diff）
        if pid is not None and hasattr(self, 'p_diff_embed'):
            # padding位置统一映射到0号embedding，避免出现负索引
            pid = pid.masked_fill(pid < 0, 0)
            # pid条件向量（用于正则化，也用于生成FiLM参数）
            p_diff = self.p_diff_embed(pid)  # (batch_size, seq_len, d_model)

            # FiLM参数：gamma用于逐维缩放，beta用于逐维平移
            # gamma初始化接近1，尽量不破坏原始DTransformer的初始行为
            gamma = 1.0 + torch.tanh(self.p_film_gamma(p_diff))  # (bs, seqlen, d_model) ∈ (0, 2)
            beta = self.p_film_beta(p_diff)                       # (bs, seqlen, d_model)

            # 难度方向向量：由题目类型/响应类型提供“变化方向”，再由FiLM控制变化强度与偏置
            q_diff_emb = self.q_diff_embed(q)
            q_emb = q_emb + (q_diff_emb * gamma + beta)

            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb = s_emb + (s_diff_emb * gamma + beta)
        return q_emb, s_emb, lens, p_diff

    def readout(self, z, query):
        """
        【核心】知识状态聚合：使用注意力机制聚合多个知识状态
        
        从n_know个知识状态中，根据当前问题query动态选择相关的知识状态。
        
        参数:
            z: 知识状态表示，重塑为(bs*seqlen, n_know, d_model)
            query: 当前问题嵌入 (bs, seqlen, d_model)
        
        返回:
            聚合后的知识表示 (bs, seqlen, d_model)
        
        机制:
            使用know_params作为key，query作为query，z作为value
            计算注意力权重alpha，然后加权求和
        """
        bs, seqlen, _ = query.size()
        
        # 准备key: 使用know_params作为key
        # 形状: (n_know, d_model) -> (bs, seqlen, n_know, d_model) -> (bs*seqlen, n_know, d_model)
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )
        
        # 准备value: 使用z作为value
        value = z.reshape(bs * seqlen, self.n_know, -1)

        # 计算注意力权重: beta = key^T × query
        # 形状: (bs*seqlen, n_know, d_model) × (bs*seqlen, d_model, 1) -> (bs*seqlen, 1, n_know)
        beta = torch.matmul(
            key,
            query.reshape(bs * seqlen, -1, 1),
        ).view(bs * seqlen, 1, self.n_know)
        
        # Softmax归一化得到注意力权重
        alpha = torch.softmax(beta, -1)  # (bs*seqlen, 1, n_know)
        
        # 加权求和: alpha × value
        # (bs*seqlen, 1, n_know) × (bs*seqlen, n_know, d_model) -> (bs*seqlen, 1, d_model)
        return torch.matmul(alpha, value).view(bs, seqlen, -1)

    def predict(self, q, s, pid=None, n=1):
        """
        预测函数：给定历史交互，预测未来答题表现
        
        参数:
            q: 题目类型ID序列
            s: 答题响应序列
            pid: 题目ID序列（可选）
            n: 预测未来第n步（T+n预测）
        
        返回:
            y: 预测的logits
            z: 知识状态表示
            q_emb: 题目嵌入
            reg_loss: 正则化损失
            scores: 注意力分数
        """
        # 步骤1: 嵌入
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid)
        
        # 步骤2: 前向传播得到知识状态
        z, q_scores, k_scores = self(q_emb, s_emb, lens)

        # 步骤3: 预测T+n
        if self.shortcut:
            # AKT模式：直接使用z
            assert n == 1, "AKT does not support T+N prediction"
            h = z
        else:
            # 标准模式：使用readout聚合知识状态
            query = q_emb[:, n - 1 :, :]  # 从第n-1个位置开始的问题嵌入
            h = self.readout(z[:, : query.size(1), :], query)

        # 步骤4: 输出层预测
        # 拼接问题嵌入和知识表示，通过MLP得到预测
        y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

        # 返回结果（包含正则化损失）
        # p_diff: pid条件向量（FiLM用）；若未提供pid或未启用n_pid，则为float(0.0)
        if pid is not None and isinstance(p_diff, torch.Tensor):
            reg_loss = (p_diff ** 2).mean() * 1e-3
        else:
            reg_loss = 0.0
        return y, z, q_emb, reg_loss, (q_scores, k_scores)
    def get_loss(self, q, s, pid=None):
        """
        标准损失函数：二元交叉熵 + 正则化
        
        用于不使用对比学习的训练
        """
        logits, _, _, reg_loss, _ = self.predict(q, s, pid)
        
        # 只计算有效位置（s>=0）的损失
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        
        return (
            F.binary_cross_entropy_with_logits(
                masked_logits, masked_labels, reduction="mean"
            )
            + reg_loss
        )

    def get_cl_loss(self, q, s, pid=None):
        """
        对比学习损失函数：预测损失 + 对比学习损失 + 正则化
        
        流程:
        1. 数据增强：生成增强样本（序列重排、响应翻转）
        2. 前向传播：计算原始样本和增强样本的表示
        3. 对比学习：相同样本的表示应该接近，不同样本的表示应该远离
        4. 预测损失：标准的二元交叉熵
        
        返回:
            总损失, 预测损失, 对比学习损失
        """
        bs = s.size(0)

        # 检查序列长度，过短则跳过对比学习
        lens = (s >= 0).sum(dim=1)
        minlen = lens.min().item()
        if minlen < MIN_SEQ_LEN:
            return self.get_loss(q, s, pid)

        # ============ 数据增强：生成正样本对 ============
        q_ = q.clone()
        s_ = s.clone()

        if pid is not None:
            pid_ = pid.clone()
        else:
            pid_ = None

        # 序列重排：随机交换相邻位置
        for b in range(bs):
            # 随机选择dropout_rate比例的位置
            idx = random.sample(
                range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                # 交换位置i和i+1
                q_[b, i], q_[b, i + 1] = q_[b, i + 1], q_[b, i]
                s_[b, i], s_[b, i + 1] = s_[b, i + 1], s_[b, i]
                if pid_ is not None:
                    pid_[b, i], pid_[b, i + 1] = pid_[b, i + 1], pid_[b, i]

        # ============ Hard Negative：生成负样本 ============
        # 翻转响应（答对变答错，答错变答对）
        s_flip = s.clone() if self.hard_neg else s_
        for b in range(bs):
            idx = random.sample(
                range(lens[b]), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]  # 0->1, 1->0
        if not self.hard_neg:
            s_ = s_flip

        # ============ 前向传播：计算表示 ============
        # 原始样本
        logits, z_1, q_emb, reg_loss, _ = self.predict(q, s, pid)
        masked_logits = logits[s >= 0]

        # 增强样本（正样本对）
        _, z_2, *_ = self.predict(q_, s_, pid_)

        # Hard negative样本
        if self.hard_neg:
            _, z_3, *_ = self.predict(q, s_flip, pid)

        # ============ 对比学习损失 ============
        # 计算原始样本和增强样本的相似度
        input = self.sim(z_1[:, :minlen, :], z_2[:, :minlen, :])
        
        # 如果使用hard negative，也计算与负样本的相似度
        if self.hard_neg:
            hard_neg = self.sim(z_1[:, :minlen, :], z_3[:, :minlen, :])
            input = torch.cat([input, hard_neg], dim=1)
        
        # 目标：每个样本应该与自己的增强版本最相似
        # target[i] = i（对角线）
        target = (
            torch.arange(s.size(0))[:, None]
            .to(self.know_params.device)
            .expand(-1, minlen)
        )
        cl_loss = F.cross_entropy(input, target)

        # ============ 预测损失 ============
        masked_labels = s[s >= 0].float()
        pred_loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction="mean"
        )

        # 多窗口预测（如果window>1）
        for i in range(1, self.window):
            label = s[:, i:]
            query = q_emb[:, i:, :]
            h = self.readout(z_1[:, : query.size(1), :], query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

            pred_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )
        pred_loss /= self.window

        # ============ 总损失 ============
        # 总损失 = 预测损失 + λ_cl × 对比学习损失 + 正则化损失
        return pred_loss + cl_loss * self.lambda_cl + reg_loss, pred_loss, cl_loss

    def sim(self, z1, z2):
        """
        计算两个知识状态表示的相似度矩阵
        
        使用余弦相似度，温度系数为0.05
        
        参数:
            z1: 第一组知识状态 (bs, seqlen, n_know*d_model)
            z2: 第二组知识状态 (bs, seqlen, n_know*d_model)
        
        返回:
            相似度矩阵 (bs, bs, seqlen)，其中[i,j,t]表示样本i和样本j在时间步t的相似度
        """
        bs, seqlen, _ = z1.size()
        
        # 重塑为(bs, 1, seqlen, n_know, d_model)和(1, bs, seqlen, n_know, d_model)
        z1 = z1.unsqueeze(1).view(bs, 1, seqlen, self.n_know, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seqlen, self.n_know, -1)
        
        # 可选的投影层
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        
        # 对n_know维度求平均，然后计算余弦相似度
        # 除以0.05是温度系数，使相似度分布更陡峭
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / 0.05

    def tracing(self, q, s, pid=None):
        """
        知识追踪：可视化学生的知识状态演变
        
        返回每个时间步在每个知识状态上的掌握程度（概率）
        
        返回:
            y: (n_know, seq_len+1) 每个知识状态在每个时间步的掌握概率
        """
        # 添加一个假的padding，生成最后一个时间步的追踪结果
        pad = torch.tensor([0]).to(self.know_params.device)
        q = torch.cat([q, pad], dim=0).unsqueeze(0)
        s = torch.cat([s, pad], dim=0).unsqueeze(0)
        if pid is not None:
            pid = torch.cat([pid, pad], dim=0).unsqueeze(0)

        with torch.no_grad():
            # 嵌入和前向传播
            q_emb, s_emb, lens, _ = self.embedding(q, s, pid)
            z, _, _ = self(q_emb, s_emb, lens)
            
            # 使用所有知识状态作为query查询
            query = self.know_params.unsqueeze(1).expand(-1, z.size(1), -1).contiguous()
            z = z.expand(self.n_know, -1, -1).contiguous()
            
            # 聚合和预测
            h = self.readout(z, query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)
            y = torch.sigmoid(y)  # 转换为概率

        return y


class DTransformerLayer(nn.Module):
    """
    DTransformer的基础Transformer层
    
    组成:
        - MultiHeadAttention: 多头注意力
        - Dropout: 正则化
        - 残差连接: 缓解梯度消失
        - LayerNorm: 层归一化
    """
    
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        """
        参数:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout率
            kq_same: 是否共享Query和Key的线性层
                    True: 自注意力（Q和K用同一个线性层）
                    False: 交叉注意力（Q和K用不同的线性层）
        """
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        """返回模型所在设备"""
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False):
        """
        前向传播
        
        参数:
            query: 查询向量 (batch_size, seq_len, d_model)
            key: 键向量 (batch_size, seq_len, d_model)
            values: 值向量 (batch_size, seq_len, d_model)
            lens: 有效序列长度
            peek_cur: 是否允许看到当前时间步
                     True: 编码模式（位置i可以看到位置<=i）
                     False: 预测模式（位置i只能看到位置<i）
        
        返回:
            输出向量, 注意力分数
        """
        # ============ 构建因果掩码 ============
        seqlen = query.size(1)
        # tril(0): 下三角（含对角线）- 用于peek_cur=True
        # tril(-1): 严格下三角（不含对角线）- 用于peek_cur=False
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.device())

        # ============ 训练时的掩码Dropout（正则化）============
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()

            for b in range(query.size(0)):
                # 跳过过短的序列
                if lens[b] < MIN_SEQ_LEN:
                    continue
                
                # 随机选择一些位置，屏蔽它们对后续位置的影响
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    # 屏蔽位置i对位置i+1之后的影响
                    mask[b, :, i + 1 :, i] = 0

        # ============ 多头注意力计算 ============
        query_, scores = self.masked_attn_head(
            query, key, values, mask, maxout=not peek_cur
        )
        
        # ============ 残差连接 + Dropout ============
        query = query + self.dropout(query_)
        
        # ============ 层归一化 ============
        return self.layer_norm(query), scores


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    将d_model维的向量分成n_heads个头，每个头独立计算注意力，
    最后拼接所有头的结果。
    
    核心思想: 多个"观察角度"并行工作，每个头关注不同的模式
    """
    
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        """
        参数:
            d_model: 模型维度
            n_heads: 注意力头数（d_model必须能被n_heads整除）
            kq_same: 是否共享Q和K的线性层
            bias: 线性层是否使用偏置
        """
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads  # 每个头的维度
        self.h = n_heads

        # Q, K, V的线性变换
        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            # 自注意力：Q和K共享参数
            self.k_linear = self.q_linear
        else:
            # 交叉注意力：Q和K使用不同参数
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # 【创新】时间衰减参数：每个头一个gamma值
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, maxout=False):
        """
        多头注意力前向传播
        
        步骤:
        1. 线性变换: Q, K, V
        2. 分头: 分成n_heads个头
        3. 计算注意力: 每个头独立计算
        4. 拼接: 合并所有头
        5. 输出投影
        """
        bs = q.size(0)

        # ============ 步骤1: 线性变换并分头 ============
        # (bs, seq_len, d_model) -> (bs, seq_len, n_heads, d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # ============ 步骤2: 转置以便并行计算 ============
        # (bs, seq_len, n_heads, d_k) -> (bs, n_heads, seq_len, d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # ============ 步骤3: 计算注意力（含时间效应）============
        v_, scores = attention(
            q,
            k,
            v,
            mask,
            self.gammas,  # 时间衰减参数
            maxout,
        )

        # ============ 步骤4: 拼接多头 ============
        # (bs, n_heads, seq_len, d_k) -> (bs, seq_len, n_heads, d_k) -> (bs, seq_len, d_model)
        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # ============ 步骤5: 输出投影 ============
        output = self.out_proj(concat)

        return output, scores


def attention(q, k, v, mask, gamma=None, maxout=False):
    """
    【核心】注意力计算函数（含时间效应）
    
    创新点:
    1. 时间效应建模: 使用gamma参数控制时间衰减
    2. Max-out机制: 防止注意力过度集中
    
    参数:
        q: Query (bs, n_heads, seq_len, d_k)
        k: Key (bs, n_heads, seq_len, d_k)
        v: Value (bs, n_heads, seq_len, d_k)
        mask: 掩码 (bs, n_heads, seq_len, seq_len)
        gamma: 时间衰减参数 (n_heads, 1, 1)
        maxout: 是否使用max-out归一化
    
    返回:
        output: 注意力输出
        scores: 注意力权重
    """
    # ============ 缩放点积注意力 ============
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    # ============ 【创新】时间效应建模 ============
    if gamma is not None:
        # 创建位置矩阵
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            # 计算注意力分布（用于估计时间效应）
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)

            # 计算累积注意力和位置距离
            distcum_scores = torch.cumsum(scores_, dim=-1)  # 累积注意力
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)  # 总注意力
            position_effect = torch.abs(x1 - x2)[None, None, :, :]  # 位置距离
            
            # 距离分数 = (总注意力 - 累积注意力) × 位置距离
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        # 应用时间衰减: exp(dist_scores × gamma)
        # gamma为负值，距离越远衰减越多
        gamma = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        # 调整注意力分数
        scores *= total_effect

    # ============ Softmax归一化 ============
    scores.masked_fill_(mask == 0, -1e32)  # 掩码位置设为负无穷
    scores = F.softmax(scores, dim=-1)      # Softmax归一化
    scores = scores.masked_fill(mask == 0, 0)  # 硬设为0，防止信息泄露

    # ============ 【创新】Max-out机制 ============
    # 防止某个位置的注意力权重过大
    if maxout:
        scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
        scores *= scale

    # ============ 加权求和 ============
    output = torch.matmul(scores, v)
    return output, scores