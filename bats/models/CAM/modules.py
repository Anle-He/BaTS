import torch
import torch.nn as nn


class GTR(nn.Module):
    def __init__(self, seq_len, period_len=24):
        super().__init__()

        self.period_len = period_len

        # 优化1: Bottleneck 结构
        # 作用：将全局查询向量映射到与输入数据相似的特征空间
        # 同时通过降维减少参数量，增加非线性表达能力
        self.mapping = nn.Sequential(
            nn.Linear(seq_len, seq_len // 4),
            nn.GELU(),
            nn.Linear(seq_len // 4, seq_len),
        )

        # 计算卷积核大小：保持覆盖一个完整周期的逻辑
        kernel_size = 1 + 2 * (self.period_len // 2)

        # 优化2: 双通道卷积 + 门控机制
        # in_channels=2: 分别对应 [原始输入 x, 全局查询 q]
        # out_channels=2: 分别对应 [融合特征 feat, 门控权重 gate]
        self.conv1d = nn.Conv1d(
            in_channels=2,
            out_channels=2,
            kernel_size=kernel_size,
            padding='same',  # 自动保持序列长度不变
            bias=False,
        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x_in: torch.Tensor, q_in: torch.Tensor) -> torch.Tensor:
        """
        x_in: (batch_size, num_nodes, seq_len)
        q_in: (batch_size, num_nodes, seq_len)
        """
        B, N, L = x_in.shape

        # --- Step 1: Mapping ---
        # 将全局参数 q 投影到数据空间
        global_query = self.mapping(q_in)  # (B, N, L)

        # --- Step 2: Alignment & Reshape ---
        # [关键修正] 使用 stack 在 dim=2 堆叠
        # 结果形状: (B, N, 2, L)，内存中 x 和 q 是交错对齐的
        combined = torch.stack([x_in, global_query], dim=2)

        # 展平批次和节点维度以进行并行卷积
        # 形状: (B*N, 2, L)
        combined = combined.view(B * N, 2, L)

        # --- Step 3: Convolution & Gating ---
        conv_out = self.conv1d(combined)  # (B*N, 2, L)

        # 分离特征通道和门控通道
        feat, gate = conv_out[:, 0:1, :], conv_out[:, 1:2, :]

        # 使用 Sigmoid 激活门控，范围 [0, 1]
        gate = torch.sigmoid(gate)

        # 门控融合：模型自动学习保留多少融合信息
        conv_out = feat * gate  # (B*N, 1, L)

        # --- Step 4: Output ---
        # 恢复形状并应用 Dropout
        out = self.dropout(conv_out.view(B, N, L))

        return out


class SeriesEmbedding(nn.Module):
    def __init__(self, seq_len_in, d_model, dropout):
        super().__init__()

        self.proj = nn.Linear(seq_len_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, L, N) -> (B, N, L) -> proj(x) -> (B, N, D)
        x = x.transpose(1, 2)
        return self.dropout(self.proj(x))


class SinkhornProjection(nn.Module):
    """Sinkhorn-Knopp 算法：将任意矩阵投影到双随机矩阵流形上"""

    def __init__(self, iterations=5):
        super().__init__()
        self.iterations = iterations

    def forward(self, A):
        # Log-Sinkhorn: 在对数域进行归一化，数值更稳定
        log_M = A.clone()
        for _ in range(self.iterations):
            # 行归一化: 减去 log-sum-exp
            log_M = log_M - torch.logsumexp(log_M, dim=-1, keepdim=True)
            # 列归一化
            log_M = log_M - torch.logsumexp(log_M, dim=-2, keepdim=True)
        return torch.exp(log_M)


class MHCBlock(nn.Module):
    """
    Manifold Hyper-Complex Block
    """

    def __init__(self, ssm, ssm_r, d_model, d_ff, dropout, num_streams):
        super().__init__()

        self.num_streams = num_streams

        self.ssm = ssm
        self.ssm_r = ssm_r

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout),
        )

        # 流聚合权重 (用于将多个流合并为一个流)
        self.agg_weight = nn.Parameter(torch.ones(num_streams) / num_streams)

        # MHC 流混合参数
        self.theta_ssm = nn.Parameter(torch.eye(num_streams) * 2.0)  # SSM 混合矩阵参数
        self.phi_ssm = nn.Parameter(torch.ones(num_streams) * 0.5)  # SSM 残差门控
        self.theta_ffn = nn.Parameter(torch.eye(num_streams) * 2.0)  # FFN 混合矩阵参数
        self.phi_ffn = nn.Parameter(torch.ones(num_streams) * 0.5)  # FFN 残差门控

        self.sinkhorn = SinkhornProjection()

    def aggregate_streams(self, x_stream, w_agg):
        # x_stream: (B, S, N, E), w_agg: (S,)
        return torch.einsum('s,bsne->bne', w_agg, x_stream)

    def stream_residual(self, x_stream, sublayer_out, theta, phi):
        """
        MHC 流混合残差连接
        Args:
            x_stream: (B, S, N, E) - 当前的流
            sublayer_out: (B, N, E) - 子层输出（需要广播到每个流）
            theta: (S, S) - 流混合矩阵参数
            phi: (S,) - 残差门控参数
        Returns:
            x_new: (B, S, N, E) - 混合后的新流
        """
        S = x_stream.shape[1]

        # Sinkhorn 投影得到双随机流混合矩阵
        W = self.sinkhorn(theta)  # (S, S)

        # 残差门控: (S,) -> (1, S, 1, 1)
        gate = torch.sigmoid(phi).view(1, S, 1, 1)

        # 流混合: (B, S, N, E) * (S, S) -> (B, S, N, E)
        # ij,bjne->bine: i=输出流, j=输入流
        x_mixed = torch.einsum('ij,bjne->bine', W, x_stream)

        # 残差连接: 混合流 + 门控 * 子层输出
        return x_mixed + gate * sublayer_out.unsqueeze(1)

    def forward(self, x_stream):
        """
        Args:
            x_stream: (B, S, N, E) - Batch, Streams, Nodes/Channels, Embedding_dim
        Returns:
            x_stream: (B, S, N, E)
        """
        # 预计算聚合权重（只计算一次，提高效率）
        w_agg = torch.softmax(self.agg_weight, dim=0)  # (S,)

        # ========== MHC SSM Sublayer ==========
        # 1. 聚合流 -> 单一流
        sub_in = self.aggregate_streams(x_stream, w_agg)  # (B, N, E)

        # 2. SSM 处理（双向）
        ssm_in_norm = self.norm1(sub_in)
        ssm_out = self.ssm(ssm_in_norm) + self.ssm_r(ssm_in_norm.flip(dims=[1])).flip(
            dims=[1]
        )

        # 3. MHC 流混合残差连接
        x_stream = self.stream_residual(x_stream, ssm_out, self.theta_ssm, self.phi_ssm)

        # ========== MHC FFN Sublayer ==========
        # 1. 聚合流 -> 单一流
        sub_in_ffn = self.aggregate_streams(x_stream, w_agg)  # (B, N, E)

        # 2. FFN 处理
        sub_in_ffn_norm = self.norm2(sub_in_ffn)
        ffn_out = self.ffn(sub_in_ffn_norm)

        # 3. MHC 流混合残差连接
        x_stream = self.stream_residual(x_stream, ffn_out, self.theta_ffn, self.phi_ffn)

        return x_stream


class CrossDomainAlignment(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # 通道级残差权重
        # 关键修正：初始化为 -5.0，使得 sigmoid(output) 约为 0.007，接近 0
        # 这样初始状态 out ≈ t_feat + 0 * f_orth = t_feat
        # 保证了训练初期的稳定性
        self.gate_alpha = nn.Parameter(-5.0 * torch.ones(d_model))

    def forward(self, t_feat, f_feat):
        """
        Args:
            t_feat: [B, N, D] (Anchor)
            f_feat: [B, N, D] (To be aligned)
        Logic:
            out = t_feat + alpha * (f_aligned - t_feat)
            初始 alpha=0 -> out = t_feat
        """
        # --- Step 1: 正交对齐 (高效，无参) ---
        # 计算 f 在 t 上的投影
        numerator = (t_feat * f_feat).sum(dim=-1, keepdim=True)
        denominator = (t_feat * t_feat).sum(dim=-1, keepdim=True) + 1e-6
        proj_scale = numerator / denominator
        f_proj_t = proj_scale * t_feat

        # 获取频域正交分量 (频域独有的新信息)
        f_orthogonal = f_feat - f_proj_t

        # --- Step 2: 稳定融合 ---
        # 使用 Sigmoid 将 alpha 约束在 [0, 1] 之间
        # alpha 初始化为 0，sigmoid(0) = 0.5。
        # 为了让初始输出严格等于 t_feat，我们使用 2 * sigmoid - 1 的变体，
        # 或者更直观地：out = t + lambda * f_orth
        # 初始化 lambda = 0 即可。

        # 这里为了符合你 "w1*t + w2*aligned" 的想法，我们可以转换一下公式：
        # 假设 aligned = t + f_orth
        # out = (1-w)*t + w*aligned = t - w*t + w*t + w*f_orth = t + w*f_orth
        # 所以只需要一个参数 w 即可控制两者比例，且 w=0 时完全为 t。

        # 为了让模型学习更灵活，我们使用 Sigmoid 激活
        gate = torch.sigmoid(self.gate_alpha)  # [D]

        # 融合
        # 这里的逻辑是：基础是 t_feat，然后根据 gate 决定每个维度吸收多少 f_orthogonal
        # Base(t) + Correction(f_orth)
        out = t_feat + gate * f_orthogonal

        return out


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):

        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError

        return x

    def _get_statistics(self, x):

        dim2reduce = tuple(range(1, x.ndim - 1))

        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):

        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):

        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev

        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x
