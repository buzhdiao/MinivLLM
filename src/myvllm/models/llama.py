from myvllm.layers import *  # 导入自定义层与算子：并行线性层、Attention、RotaryEmbedding、SiluAndMul、LayerNorm、VocabParallelEmbedding、ParallelLMHead 及 dist

from typing import Tuple  # 类型注解：DecoderLayer.forward 返回 (x, residual) 二元组

import torch  # PyTorch 主库：张量运算与设备管理
import torch.nn as nn  # 神经网络基础组件（nn.Module、nn.ModuleList）

# LlamaAttn：Llama 的 GQA（Grouped-Query Attention，分组查询注意力）层
# 职责：QKV 融合投影（列并行，在此按 head 完成权重分片）→ 切分 Q/K/V → RoPE 旋转位置编码
#       → PagedAttention（结合分页 KV cache 实现 GQA）→ o_proj 行并行输出投影（内部 all_reduce 聚合）
class LlamaAttn(nn.Module):
    # 初始化：按 tensor parallel 切分本卡的 Q/KV head，并构建 QKV 投影、RoPE、attention、o_proj 子模块
    def __init__(
        self,
        hidden_size: int,  # 隐藏层维度（注意力的输入/输出特征维度）
        head_dim: int,  # 每个注意力 head 的维度
        num_qo_heads: int,  # Q head 总数（全局值）
        num_kv_heads: int,  # KV head 总数（GQA：多个 Q head 共享一组 KV，通常远小于 Q head 数）
        has_attn_bias: bool = False,  # QKV 投影是否带 bias（Llama 默认为 False）
        rms_norm_epsilon: float = 1e-5,  # RMSNorm 的 epsilon（数值稳定项）
        rope_base: int = 500000,  # RoPE 频率基 theta（Llama 3 系列为 500000）
        max_position_embeddings: int = 131072,  # 最大上下文长度（决定 RoPE 预计算表大小）
        block_size: int = 256,  # 分页 KV cache 中每个物理块可容纳的 token 数
    ):
        super().__init__()  # 调用父类 nn.Module 的构造函数
        self.tp_size = dist.get_world_size()  # tensor parallel 并行度（进程组世界大小）

        self.total_num_heads = num_qo_heads  # 全局 Q head 数（所有卡合计）
        self.num_heads = num_qo_heads // self.tp_size  # 本卡分到的 Q head 数（列并行按 head 切分）

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_qo_heads  # 全局 KV head 数（未指定时退化为 MHA）
        # self.num_kv_heads is per-GPU value (divided by tp_size)  # 说明：num_kv_heads 为本卡值（已除以 tp_size）
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size  # 本卡分到的 KV head 数（GQA 下 KV 的显存与带宽开销显著降低）

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_qo_heads  # head 维度（缺省时按 hidden_size 均分推导）

        self.qkv_projection = QKVColumnParallelLinear(  # QKV 融合投影：列并行层，将 Q/K/V 三组权重沿输出维拼接成一个大矩阵并切分到各卡
            input_size=hidden_size,  # 输入维度
            head_size=head_dim,  # 每个 head 的维度
            num_heads=num_qo_heads,  # Q head 总数
            num_kv_heads=num_kv_heads,  # KV head 总数
            bias=has_attn_bias,  # 是否含 bias
        )

        self.q_size = head_dim * self.num_heads  # 本卡切分后 Q 段的宽度
        self.kv_size = head_dim * self.num_kv_heads  # 本卡切分后单份 K（或 V）段的宽度
        
        # Llama 3.2 does not have q_norm or k_norm  # Llama 3.2 没有 q_norm/k_norm（这是与 Qwen3 的结构差异之一）

        self.rotary_emb = RotaryEmbedding(  # RoPE 旋转位置编码模块：对 Q/K 做旋转变换以注入位置信息
            base=rope_base,  # 频率基 theta
            rotary_embedding=head_dim,  # 旋转作用的维度（等于 head_dim）
            max_position=max_position_embeddings,  # 最大预计算位置数
            is_llama3=True  # 采用 Llama3 风格的频率设置
        )
        self.attention = Attention(  # 注意力核心算子：负责分页 KV cache 的读写与 GQA attention 计算
            num_heads=num_qo_heads,  # Q head 总数
            head_dim=head_dim,  # head 维度
            num_kv_heads=num_kv_heads,  # KV head 总数（计算时广播到共享它的 Q head 组）
            block_size=block_size,  # 分页块大小
        )
        self.o_proj = RowParallelLinear(  # 输出投影：行并行层（按输入维切分），各卡部分和经 all_reduce 求和
            input_size= head_dim * num_qo_heads,  # 输入维度：全部 head 拼接宽度
            output_size=hidden_size,  # 输出恢复为 hidden_size
            bias=False,  # 不带 bias
        )

    # 前向传播：依次完成 QKV 投影、RoPE、GQA attention 与输出投影
    def forward(
        self, 
        x: torch.Tensor,  # 输入隐藏状态
        positions: torch.Tensor,  # 每个 token 的绝对位置（供 RoPE 使用）
    ) -> torch.Tensor:
        # Input: x shape (B, N, hidden_size) - REPLICATED on all GPUs  # 输入形状 (B, N, hidden_size)，在所有 GPU 上为完整复制（未分片）

        # ===== QKV Projection (Column Parallel - THIS IS WHERE SHARDING HAPPENS) =====  # QKV 投影（列并行——权重分片发生在这一步）
        # Output shape PER GPU: (B, N, head_dim * (num_heads + 2*num_kv_heads))  # 每 GPU 的输出形状
        # where num_heads = total_num_heads/tp_size  # 其中 num_heads 为每卡的 Q head 数
        #       num_kv_heads = total_num_kv_heads/tp_size  # num_kv_heads 为每卡的 KV head 数
        qkv = self.qkv_projection(x)  # 计算融合 QKV 投影，本卡得到 (Q段; K段; V段) 的拼接

        # ===== Split QKV =====  # 切分 QKV
        # q_size = head_dim * num_heads           - Per-GPU size!  # q_size 是本卡尺寸
        # kv_size = head_dim * num_kv_heads       - Per-GPU size!  # kv_size 是本卡尺寸
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)  # 沿最后一维把融合输出切成 Q/K/V 三段

        # Handle both batched (3D) and varlen (2D) inputs  # 同时支持批处理(3D)与变长拼接(2D)两种输入布局
        # Varlen: q shape: (total_tokens, q_size) where q_size = num_heads * head_dim  # 变长模式：整个 batch 的 token 拼接为一维
        # Batched: q shape: (B, N, q_size)  # 批处理模式：标准 (B, N, C) 形状
        if q.dim() == 2:  # 判断是否为变长（2D）输入
            # Varlen mode: (total_tokens, q_size) -> (total_tokens, num_heads, head_dim)  # 变长模式：reshape 为 (token 数, head 数, head 维)
            q = q.view(-1, self.num_heads, self.head_dim)  # Q 变形为按 head 分组
            k = k.view(-1, self.num_kv_heads, self.head_dim)  # K 变形为按 KV head 分组
            v = v.view(-1, self.num_kv_heads, self.head_dim)  # V 变形为按 KV head 分组
        else:
            # Batched mode: (B, N, q_size) -> (B, N, num_heads, head_dim)  # 批处理模式：reshape 为 (B, N, head 数, head 维)
            B, N = q.size(0), q.size(1)  # 读取批大小与序列长度
            q = q.view(B, N, self.num_heads, self.head_dim)  # Q 变形为按 head 分组
            k = k.view(B, N, self.num_kv_heads, self.head_dim)  # K 变形为按 KV head 分组
            v = v.view(B, N, self.num_kv_heads, self.head_dim)  # V 变形为按 KV head 分组

        q, k = self.rotary_emb(positions, q, k)  # 对 Q/K 应用 RoPE：按位置旋转，使 attention 具备相对位置感知能力

        o = self.attention(q, k, v)  # 执行 GQA attention：读/写分页 KV cache，得到每个 token 的注意力输出
        # o shape: (B*N, num_heads, head_dim)     - Per-GPU, different heads per GPU  # o 为本卡结果，各卡持有不同的 head

        # ===== Output Projection (Row Parallel - COMMUNICATION HAPPENS HERE by dist.all_reduce) =====  # 输出投影（行并行——通信由 dist.all_reduce 在此发生）
        o = self.o_proj(o)  # 行并行线性层：各卡算出部分和后 all_reduce 求和，恢复完整 hidden_size
        # Input: (B*N, num_heads * head_dim) sharded across GPUs  # 输入在各卡间按输入维分片
        # Output: (B*N, hidden_size) REPLICATED on all GPUs (after all_reduce)  # 输出经 all_reduce 后在所有 GPU 上完整复制

        return o  # 返回注意力层输出（各卡内容一致）
    
# LlamaMLP
# gate_up
# activateion
# gate_down
# LlamaMLP：前馈网络（SwiGLU 结构）
# 职责：gate_up 合并投影（列并行，一次算出 gate 与 up 两路）→ SiLU 门控逐元素相乘 → down_proj 降维（行并行）
class LlamaMLP(nn.Module):
    # 初始化：构建 gate/up 合并列并行层、SiLU 门控激活与行并行降维层
    def __init__(
        self,
        hidden_size: int,  # 输入/输出维度
        intermediate_size: int,  # FFN 中间层维度（通常为 hidden_size 的数倍）
        bias: bool = True,  # 线性层是否带 bias
    ):
        super().__init__()  # 调用父类构造函数
        self.gate_up = MergedColumnParallelLinear(  # 将 gate_proj 与 up_proj 两路权重合并为一张大矩阵，按输出维切分到各卡
            input_size=hidden_size,  # 输入维度
            output_sizes=[intermediate_size] * 2,  # 两路输出各为 intermediate_size（gate 与 up）
            bias=bias,  # 是否带 bias
        )
        self.activation = SiluAndMul()  # SwiGLU 激活：前半段（gate）经 SiLU 后逐元素乘以后半段（up）
        self.down_proj = RowParallelLinear(  # 降维投影：行并行，内部 all_reduce 聚合
            input_size=intermediate_size,  # 输入为中间层维度
            output_size=hidden_size,  # 输出回到 hidden_size
            bias=bias,  # 是否带 bias
        )

    # 前向传播：SwiGLU 前馈计算
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_proj(self.activation(self.gate_up(x)))  # 先升维(gate/up 两路) → SiLU 门控相乘 → 再降维
        return x  # 返回 FFN 输出

# LlamaDecoderLayer
# input_layernorm, also consider residual
# self_attn
# layer_norm post attention
# mlp
# LlamaDecoderLayer：单个 Transformer 解码层（Pre-Norm + 残差结构）
# 职责：输入 RMSNorm（融合残差）→ 自注意力 → 注意力后 RMSNorm（融合残差）→ SwiGLU MLP
class LlamaDecoderLayer(nn.Module):
    # 初始化：构建两个归一化层、自注意力层与 MLP 层
    def __init__(
        self,
        hidden_size: int = 2048,  # 隐藏层维度
        head_dim: int = 64,  # 每个 head 的维度
        num_qo_heads: int = 32,  # Q head 总数
        num_kv_heads: int = 8,  # KV head 总数（GQA 分组数）
        has_attn_bias: bool = False,  # 注意力 QKV 投影是否带 bias
        rms_norm_epsilon: float = 1e-05,  # RMSNorm 的 epsilon
        rope_base: int = 500000,  # RoPE 频率基 theta
        max_position_embeddings: int = 131072,  # 最大上下文长度
        intermediate_size: int = 8192,  # FFN 中间层维度
        ffn_bias: bool = False,  # FFN 线性层是否带 bias
        block_size: int = 256,  # 分页 KV cache 块大小
    ):
        super().__init__()  # 调用父类构造函数
        gamma = torch.ones(hidden_size)  # 初始化归一化的缩放参数 gamma 为全 1
        self.input_layernorm = LayerNorm(gamma)  # 注意力前的归一化层（支持与残差融合计算）
        self.self_attn = LlamaAttn(  # 构建 Llama 自注意力层
            hidden_size=hidden_size,  # 隐藏层维度
            head_dim=head_dim,  # head 维度
            num_qo_heads=num_qo_heads,  # Q head 数
            num_kv_heads=num_kv_heads,  # KV head 数
            has_attn_bias=has_attn_bias,  # 是否带注意力 bias
            rms_norm_epsilon=rms_norm_epsilon,  # RMSNorm epsilon
            rope_base=rope_base,  # RoPE theta
            max_position_embeddings=max_position_embeddings,  # 最大上下文长度
            block_size=block_size,  # 分页块大小
        )
        self.post_attention_layernorm = LayerNorm(gamma)  # 注意力之后的归一化层
        self.mlp = LlamaMLP(  # 构建 SwiGLU 前馈网络
            hidden_size=hidden_size,  # 隐藏层维度
            intermediate_size=intermediate_size,  # FFN 中间层维度
            bias=ffn_bias  # FFN 是否带 bias
        )

    # 前向传播：Pre-Norm 残差流水线（注意力 + MLP），并计算本层所需的位置编码
    def forward(
        self,
        x: torch.Tensor,  # 输入隐藏状态
        residual: torch.Tensor | None = None,  # 跨层传递的残差流（首层为 None）
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:  # 非首层：残差流已存在
            x, residual = self.input_layernorm(x, residual)  # RMSNorm 与残差融合计算：residual 累加上一层输出，x 为归一化结果
        else:
            residual = x  # Save BEFORE normalization  # 首层：先保存原始输入作为残差（归一化之前）
            x = self.input_layernorm(x)  # 对输入做归一化
        # Compute positions based on context (respecting sequence boundaries for batched prefill)  # 依据上下文计算位置（批量 prefill 时尊重各序列边界）
        from myvllm.utils import get_context  # 函数内导入：获取当前调度步的全局上下文
        context = get_context()  # 取出当前批次的推理上下文信息
        if context.is_prefill and context.cu_seqlens_q is not None:  # 批量 prefill：多序列以 cu_seqlens 拼接（varlen 布局）
            # For batched prefill, create positions that restart at 0 for each sequence  # 批量 prefill：每个序列的位置都从 0 重新开始
            positions = []  # 收集所有 token 的位置索引
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()  # 取各序列的累积长度边界（转回 CPU 列表）
            for i in range(len(cu_seqlens) - 1):  # 遍历每一条序列
                seq_len = cu_seqlens[i+1] - cu_seqlens[i]  # 当前序列的长度
                positions.extend(range(seq_len))  # 该序列内位置依次为 0..seq_len-1
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)  # 转为与输入同设备的张量
        elif context.is_prefill:  # 单序列 prefill
            # For single sequence prefill, use sequential positions  # 单序列 prefill：使用连续递增的位置
            positions = torch.arange(x.size(0), device=x.device)  # 0..token 数-1
        else:
            # For decode, use context_lens - 1 as positions (current position for each sequence)  # decode 阶段：以各序列已有长度-1 作为当前位置
            positions = context.context_lens - 1  # 每条序列当前生成的位置索引

        x = self.self_attn(x, positions=positions)  # 自注意力计算（内部含 RoPE 与 KV cache 读写）
        # Residual connection always on for attention output  # 注意力输出的残差连接始终开启
        x, residual = self.post_attention_layernorm(x, residual)  # 将注意力输出累加进残差流，并对结果做归一化
        x = self.mlp(x)  # 经过 SwiGLU 前馈网络
        return x, residual  # 返回 (归一化后的输出, 更新后的残差流) 供下一层使用
    
# LlamaModel
# embedding
# layers stack
# final layer norm
# LlamaModel：Transformer 主体（词嵌入 + N 个解码层 + 最终归一化）
# 职责：token id → 词嵌入 → 逐层解码（残差流贯穿）→ 最终 RMSNorm
class LlamaModel(nn.Module):
    # 初始化：构建词嵌入、num_layers 个解码层与最终归一化层
    def __init__(
        self,
        vocab_size: int = 128256,  # 词表大小
        hidden_size: int = 2048,  # 隐藏层维度
        head_dim: int = 64,  # 每个 head 的维度
        num_qo_heads: int = 32,  # Q head 总数
        num_kv_heads: int = 8,  # KV head 总数
        has_attn_bias: bool = False,  # 注意力是否带 bias
        rms_norm_epsilon: float = 1e-5,  # RMSNorm epsilon
        rope_base: int = 500000,  # RoPE 频率基 theta
        max_position_embeddings: int = 131072,  # 最大上下文长度
        intermediate_size: int = 8192,  # FFN 中间层维度
        ffn_bias: bool = False,  # FFN 是否带 bias
        num_layers: int = 16,  # 解码层层数
        block_size: int = 256,  # 分页 KV cache 块大小
    ):
        super().__init__()  # 调用父类构造函数

        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入：按词表维度做 vocab 并行切分，各卡只持有一部分词表
            num_embeddings=vocab_size,  # 词表大小
            embedding_dim=hidden_size,  # 嵌入维度
        )
        self.layers = nn.ModuleList([  # 堆叠 num_layers 个解码层
            LlamaDecoderLayer(  # 每层均为相同的结构配置
                hidden_size=hidden_size,  # 隐藏层维度
                head_dim=head_dim,  # head 维度
                num_qo_heads=num_qo_heads,  # Q head 数
                num_kv_heads=num_kv_heads,  # KV head 数
                has_attn_bias=has_attn_bias,  # 注意力 bias
                rms_norm_epsilon=rms_norm_epsilon,  # RMSNorm epsilon
                rope_base=rope_base,  # RoPE theta
                max_position_embeddings=max_position_embeddings,  # 最大上下文长度
                intermediate_size=intermediate_size,  # FFN 中间层维度
                ffn_bias=ffn_bias,  # FFN bias
                block_size=block_size,  # 分页块大小
            ) for _ in range(num_layers)  # 按层数重复构建
        ])
        gamma = torch.ones(hidden_size)  # 最终归一化的缩放参数初始化为全 1
        self.norm = LayerNorm(gamma)  # 最终的归一化层（输出前）

    # 前向传播：嵌入 → 逐层解码（残差流贯穿全程）→ 最终归一化
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)  # 将 token id 映射为词向量
        residual = None  # 残差流初始化为 None（首层会从输入建立）
        for layer in self.layers:  # 依次通过每个解码层
            x, residual = layer(x, residual)  # 每层返回 (归一化输出, 累积残差)
        x, _ = self.norm(x, residual)  # 最终归一化：把最后的残差累加进来并归一化（丢弃新的残差）
        return x  # 返回最终的隐藏状态


# LlamaForCausalLM：顶层因果语言模型（LlamaModel + lm_head）
# 职责：组合主体模型与输出投影头，支持词嵌入权重绑定（tie_word_embeddings）
class LlamaForCausalLM(nn.Module):
    # 初始化：构建主体模型与 lm_head，可选地与词嵌入共享权重
    def __init__(
            self,
            vocab_size: int = 128256,  # 词表大小
            hidden_size: int = 2048,  # 隐藏层维度
            head_dim: int = 64,  # head 维度
            num_qo_heads: int = 32,  # Q head 总数
            num_kv_heads: int = 8,  # KV head 总数
            has_attn_bias: bool = False,  # 注意力是否带 bias
            rms_norm_epsilon: float = 1e-5,  # RMSNorm epsilon
            rope_base: int = 500000,  # RoPE theta
            max_position_embeddings: int = 131072,  # 最大上下文长度
            intermediate_size: int = 8192,  # FFN 中间层维度
            ffn_bias: bool = False,  # FFN 是否带 bias
            num_layers: int = 16,  # 层数
            block_size: int = 256,  # 分页块大小
            tie_word_embeddings: bool = True  # 是否将 lm_head 与词嵌入权重绑定（共享）
        ):
        super().__init__()  # 调用父类构造函数
        self.model = LlamaModel(  # 构建 Transformer 主体
            vocab_size=vocab_size,  # 词表大小
            hidden_size=hidden_size,  # 隐藏层维度
            head_dim=head_dim,  # head 维度
            num_qo_heads=num_qo_heads,  # Q head 数
            num_kv_heads=num_kv_heads,  # KV head 数
            has_attn_bias=has_attn_bias,  # 注意力 bias
            rms_norm_epsilon=rms_norm_epsilon,  # RMSNorm epsilon
            rope_base=rope_base,  # RoPE theta
            max_position_embeddings=max_position_embeddings,  # 最大上下文长度
            intermediate_size=intermediate_size,  # FFN 中间层维度
            ffn_bias=ffn_bias,  # FFN bias
            num_layers=num_layers,  # 层数
            block_size=block_size,  # 分页块大小
        )
        self.lm_head = ParallelLMHead(  # 输出头：词表并行的线性层，把 hidden_size 映射回词表 logits
            num_embeddings=vocab_size,  # 词表大小
            embedding_dim=hidden_size,  # 输入维度
        )
        if tie_word_embeddings:  # 若启用权重绑定
            self.lm_head.weight = self.model.embed_tokens.weight  # lm_head 直接复用词嵌入权重（小模型常用，省参数）

    # 前向传播：只到隐藏状态为止（logits 由 compute_logits 按需计算）
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model(input_ids)  # 通过主体模型得到最终隐藏状态
        return x  # 返回隐藏状态 

    # 计算下一个 token 的 logits（与主体解耦，便于只对最后位置调用）
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)  # 用 lm_head 将隐藏状态投影到词表空间
        return logits  # 返回 logits
