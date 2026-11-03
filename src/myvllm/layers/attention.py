import triton  # 导入 Triton JIT 编译框架，用于编写 GPU kernel
import triton.language as tl  # 导入 Triton 的设备端语言 API
from myvllm.utils import get_context  # 导入获取当前推理上下文（slot_mapping、cu_seqlens 等）的工具
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块基类

@triton.jit  # 将函数编译为 Triton GPU kernel
def store_kvcache_kernel(
    key_ptr, # pointer to what we want to store  # 指向待写入的 K 数据
    value_ptr,  # 指向待写入的 V 数据
    k_cache_ptr, # pointer to where we want to store  # 指向 paged KV cache 的 K 部分
    v_cache_ptr,  # 指向 paged KV cache 的 V 部分
    slot_mapping_ptr,  # 指向 slot_mapping：每个 token 映射到的 cache 物理槽位
    num_kv_heads: tl.constexpr,  # KV 头数（编译期常量）
    head_dim: tl.constexpr,  # 每头维度（编译期常量）
    block_size: tl.constexpr  # 每个 page block 可容纳的 token 数（编译期常量）
):
    """
    Store keys and values into paged KV cache.
    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)
    Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    # thread ID, in dimension 0
    token_idx = tl.program_id(0) # each GPU thread processes one token  # 每个 program 处理一个 token
    # slot ID, where in cache to store this token
    slot_idx = tl.load(slot_mapping_ptr + token_idx)  # 查表得到该 token 应写入的物理槽位编号
    
    if slot_idx == -1:  # slot_mapping 中 -1 表示 padding token，无需写入
        return  # 直接结束该 program
    
    # Calculate which block and position within block
    block_idx = slot_idx // block_size  # 物理槽位所在的 page block 编号
    block_offset = slot_idx % block_size  # 在该 block 内的偏移位置
    
    # Process each head
    # program_id(0) = which token
    # program_id(1) = which head
    head_idx = tl.program_id(1)  # 第二维 program id 指定当前处理的 KV 头
    
    # it creates a vector [0, 1, ..., head_dim-1]
    # Load key and value for this token and head
    head_offsets = tl.arange(0, head_dim)  # 生成 [0,1,...,head_dim-1] 的向量，用于遍历每头内部的元素
    # Input: (num_tokens, num_kv_heads, head_dim)
    # example: input_offset = 5 * (8 * 128) + 3 * 128 + [0, 1, 2, ..., 127]
    #         = 5120 + 384 + [0, 1, 2, ..., 127]
    #         = [5504, 5505, 5506, ..., 5631]
    input_offset = (token_idx * num_kv_heads * head_dim + # skip previous tokens  # 输入张量中跳过前面的 token
                    head_idx * head_dim + # skip previous heads  # 跳过前面的头
                    head_offsets)  # 加上头内元素偏移，得到该 token 该头在输入中的线性地址

    # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim + # skip previous blocks  # cache 中跳过前面的 page block
                   block_offset * num_kv_heads * head_dim + # skip previous positions in block  # 跳过 block 内前面的槽位
                   head_idx * head_dim + # skip previous heads  # 跳过前面的头
                   head_offsets)  # 得到该 token 该头在 paged cache 中的线性地址
    
    # load key and value value floats from the pointers's memory
    key = tl.load(key_ptr + input_offset)  # 从输入读出该 token 该头的 K 向量
    value = tl.load(value_ptr + input_offset)  # 读出对应的 V 向量
    
    # store into cache
    tl.store(k_cache_ptr + cache_offset, key)  # 把 K 写入 paged cache 的对应物理槽位
    tl.store(v_cache_ptr + cache_offset, value)  # 把 V 写入对应槽位


def store_kvcache(
    key: torch.Tensor,  # 当前批次的 K 张量
    value: torch.Tensor,  # 当前批次的 V 张量
    k_cache: torch.Tensor,  # paged K cache
    v_cache: torch.Tensor,  # paged V cache
    slot_mapping: torch.Tensor,  # token -> 物理槽位映射表
    block_size: int  # 每个 page block 的 token 数
):
    """
    Store key-value pairs into paged cache.
    
    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        block_size: number of tokens per block
    """
    num_tokens, num_kv_heads, head_dim = key.shape  # 解包输入形状
    
    # Make contiguous if needed
    if not key.is_contiguous():  # Triton 按线性地址访问，需保证内存连续
        key = key.contiguous()  # K 不连续时整理为连续
    if not value.is_contiguous():  # 同样检查 V
        value = value.contiguous()  # V 不连续时整理为连续
    
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"  # K/V cache 形状必须一致
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"  # 映射表长度须等于 token 数
    
    grid = (num_tokens, num_kv_heads)  # grid 布局：第一维 token，第二维 KV 头
    # launch num_tokens x num_kv_heads threads
    store_kvcache_kernel[grid](  # 启动 kernel，共 num_tokens * num_kv_heads 个 program
        key, # tensors are automatically converted to pointers by triton  # Triton 自动把张量转为指针
        value,  # V 指针
        k_cache,  # K cache 指针
        v_cache,  # V cache 指针
        slot_mapping,  # 槽位映射表指针
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size
    )


@triton.jit  # 编译为 Triton GPU kernel
def flash_attention_varlen_kernel(
    Q, K, V, O,  # Q/K/V/输出指针（varlen 拼接格式）
    cu_seqlens_q_ptr,  # 各序列在拼接维度上的累积长度（前缀和），用于确定每条序列的边界
    scale,  # 注意力缩放因子 1/sqrt(d)
    num_heads: tl.constexpr,  # Q 头数
    num_kv_heads: tl.constexpr,  # KV 头数（GQA 时少于 Q 头数）
    head_dim: tl.constexpr,  # 每头维度
    BLOCK_M: tl.constexpr,  # 每个 program 处理的 query 行数
    BLOCK_N: tl.constexpr,  # 每次迭代处理的 K/V token 数
):
    """
    Flash Attention kernel for variable-length sequences.
    Each program processes one block of queries for one head in one sequence.
    """
    # Program IDs
    start_m = tl.program_id(0) # block index  # 当前 program 负责的 query block 编号
    off_h = tl.program_id(1) # head index  # 当前处理的 Q 头编号
    seq_idx = tl.program_id(2) # sequence index  # 当前处理的序列编号

    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)  # GQA 映射：每 num_heads/num_kv_heads 个 Q 头共享一个 KV 头
    
    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)  # 本序列在拼接 token 中的起始位置
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)  # 本序列的结束位置
    seq_len = seq_end - seq_start  # 本序列长度
    
    # Early exit if this block is beyond sequence length
    if start_m * BLOCK_M >= seq_len:  # 该 query block 完全超出序列长度
        return  # 提前结束，避免无效计算
    
    # Offset for this block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 本 block 内各 query 的序列内位置索引
    offs_d = tl.arange(0, head_dim)  # 头内维度索引 [0, head_dim)
    
    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]  # 计算 Q block 各元素的线性地址（二维形式）

    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < seq_len  # 序列内位置有效性掩码
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)  # 加载 Q block，越界位置填 0
    
    # Initialize output accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # online softmax 的分母累加器（softmax 的 sum 项）
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10  # 每行已见 score 的最大值（初值 -1e10）
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)  # 输出累加器（加权 V 的累积和）
    
    # Number of blocks to process
    num_blocks = tl.cdiv(seq_len, BLOCK_N)  # 需要遍历的 K/V block 数（向上取整）
    
    # Loop over K, V blocks
    for block_n in range(num_blocks):  # Flash Attention 的核心：分块迭代 K/V，online softmax 边算边归一化
        start_n = block_n * BLOCK_N  # 本 block K/V 的起始位置
        offs_n = start_n + tl.arange(0, BLOCK_N)  # 本 block 内各 K/V 的序列内位置索引
        
        # Mask for valid positions
        mask_n = offs_n < seq_len  # K/V 位置有效性掩码
        
        # K pointers: K has shape (total_tokens, num_kv_heads, head_dim)
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]  # 计算 K^T block 的线性地址（转置布局）
        
        # Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)  # 加载 K block，越界位置填 0
        
        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)  # 矩阵乘计算注意力分数 QK^T
        qk = qk * scale  # 乘以 1/sqrt(head_dim) 缩放
        
        # Apply causal mask: only attend to positions <= current position
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)  # 因果掩码：query 只能看到自身及之前的位置
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)  # 无效/未来位置填 -1e10，softmax 后权重为 0
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)  # 本 block 每行 score 的最大值
        m_i_new = tl.maximum(m_i, m_ij)  # 与历史最大值取 max，得到新的行最大值
        alpha = tl.exp(m_i - m_i_new)  # 修正因子：旧累加值需按新最大值重新缩放
        p = tl.exp(qk - m_i_new[:, None])  # 当前 block 的未归一化 softmax 权重（减最大值防溢出）
        
        # Rescale previous accumulator
        acc = acc * alpha[:, None]  # 按新最大值修正之前的输出累加器
        
        # Load V block - shape (BLOCK_N, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]  # 计算 V block 的线性地址
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # 加载 V block，越界位置填 0
        
        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)  # 累加当前 block 的加权 V
        
        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)  # 同步更新 softmax 分母累加器
        m_i = m_i_new  # 更新行最大值供下一轮使用
    
    # Final normalization
    acc = acc / l_i[:, None]  # 除以 softmax 分母，得到归一化输出
    
    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]  # 计算输出的线性地址
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])  # 写回输出，越界位置不写


def flash_attention_prefill(
    q: torch.Tensor,  # Q 张量 (total_tokens, num_heads, head_dim)
    k: torch.Tensor,  # K 张量 (total_tokens, num_kv_heads, head_dim)
    v: torch.Tensor,  # V 张量 (total_tokens, num_kv_heads, head_dim)
    cu_seqlens: torch.Tensor,  # 各序列累积长度（varlen 前缀和）
    scale: float,  # 注意力缩放因子
    num_heads: int,  # Q 头数
    num_kv_heads: int,  # KV 头数
    head_dim: int,  # 每头维度
) -> torch.Tensor:
    """
    Optimized Flash Attention for prefill phase with variable-length sequences.
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: cumulative sequence lengths
        scale: attention scale factor
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()  # 保证 Q 内存连续，供 Triton 线性寻址
    k = k.contiguous()  # 保证 K 连续
    v = v.contiguous()  # 保证 V 连续
    
    # Allocate output
    output = torch.empty_like(q)  # 按相同形状分配输出张量
    
    # Conservative block sizes to avoid OOM on shared memory
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (for float32 attention scores)
    # + BLOCK_M * head_dim * 4 (for Q)
    # + BLOCK_N * head_dim * 4 (for K, V)
    # Want to keep total < 48KB for most GPUs
    
    if head_dim <= 64:  # head 较小时可用较大的 block
        BLOCK_M = 64  # 每个 program 处理 64 行 query
        BLOCK_N = 64  # 每次迭代处理 64 个 K/V token
    elif head_dim <= 128:  # 中等 head 维度
        BLOCK_M = 32  # 缩小 block 以控制共享内存占用
        BLOCK_N = 32
    else:  # 大 head 维度
        BLOCK_M = 16  # 进一步缩小 block 防止共享内存溢出
        BLOCK_N = 16
    
    # Number of sequences
    num_seqs = cu_seqlens.shape[0] - 1  # 前缀和长度减 1 即为序列条数
    
    # Find max sequence length to determine grid size
    cu_seqlens_cpu = cu_seqlens.cpu()  # 前缀和搬到 CPU 计算（避免 GPU->CPU 反复同步）
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()  # 最长序列长度，决定 grid 第一维大小
    
    # Calculate grid dimensions - launch all kernels at once
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)  # grid: (query block 数, 头数, 序列数)
    
    flash_attention_varlen_kernel[grid](  # 启动 varlen Flash Attention kernel
        q, k, v, output,  # 传入 Q/K/V 与输出指针
        cu_seqlens,  # 序列边界前缀和
        scale,  # 缩放因子
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output  # 返回注意力输出


@triton.jit  # 编译为 Triton GPU kernel
def paged_attention_decode_kernel(
    output_ptr,  # 输出指针
    query_ptr,  # 当前批次的 Q 指针
    k_cache_ptr,  # paged K cache 指针
    v_cache_ptr,  # paged V cache 指针
    block_tables_ptr,  # block table：每条序列的逻辑 block -> 物理 block 映射表
    context_lens_ptr,  # 每条序列已生成的上下文长度
    scale: tl.constexpr,  # 注意力缩放因子
    num_heads: tl.constexpr,  # Q 头数
    num_kv_heads: tl.constexpr,  # KV 头数（GQA）
    head_dim: tl.constexpr,  # 每头维度
    block_size: tl.constexpr,  # 每个 page block 的 token 数
    max_num_blocks: tl.constexpr,  # 每条序列最多占用的 block 数
    BLOCK_N: tl.constexpr,  # 每次迭代处理的 KV token 数（chunk 大小）
):
    """
    Optimized paged attention kernel for decode phase.
    Processes KV cache in chunks.

    The block table is gathered per token rather than per chunk, so a chunk may
    straddle any number of blocks and no relation between BLOCK_N and block_size
    is assumed. Each lane resolves its own token independently:

        token t  ->  block_tables[batch, t // block_size], slot t % block_size
    """
    batch_idx = tl.program_id(0)  # 第一维 program id：当前批样本
    head_idx = tl.program_id(1)  # 第二维 program id：当前 Q 头
    
    # Determine which KV head this query head uses (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)  # GQA 映射：该 Q 头共享的 KV 头编号
    
    # Load context length
    context_len = tl.load(context_lens_ptr + batch_idx)  # 读取本样本的上下文长度
    
    # Load query: (batch_size, num_heads, head_dim)
    offs_d = tl.arange(0, head_dim)  # 头内维度索引
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d  # 本 batch 本头的 Q 向量线性地址
    q = tl.load(query_ptr + q_offset)  # 加载 Q 向量（decode 阶段每样本仅 1 个 token）
    
    # Initialize accumulators
    acc = tl.zeros([head_dim], dtype=tl.float32)  # 输出累加器（加权 V 的累积和）
    l_i = 0.0  # softmax 分母累加器
    m_i = -1e10  # 已见 score 的最大值（初值 -1e10）
    
    # Calculate total number of chunks to process
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)  # 按 chunk 遍历 KV cache 的总迭代次数上限
    
    # Process all tokens in chunks
    for chunk_idx in range(max_chunks):  # 分 chunk 扫描整段 KV cache，online softmax 增量更新
        # Global token index for this chunk
        token_start = chunk_idx * BLOCK_N  # 本 chunk 覆盖的首个 KV token 的逻辑索引
        
        # Only process if within valid range
        if token_start < context_len:  # 本 chunk 起点已在上下文长度之外则跳过
            # Determine which tokens in this chunk are valid
            offs_n = token_start + tl.arange(0, BLOCK_N)  # 本 chunk 内各 KV token 的逻辑位置索引
            logical_block = offs_n // block_size  # 各 token 所属的逻辑 page block
            # physical_block * block_size already moves the base pointer to the
            # start of the block, so what is added is the offset *within* the
            # block, not the global token index.
            offs_in_block = offs_n % block_size  # 各 token 在物理 block 内的槽位偏移

            in_range = (offs_n < context_len) & (logical_block < max_num_blocks)  # token 未超出上下文且 block 编号合法

            # One block-table entry per token: a chunk is free to span several
            # blocks, and those blocks need not be adjacent in the cache.
            physical_block = tl.load(
                block_tables_ptr + batch_idx * max_num_blocks + logical_block,  # 每条 lane 各自查 block table：逻辑 block -> 物理 block
                mask=in_range, other=-1)  # 越界 lane 填 -1 表示无效
            valid = in_range & (physical_block != -1)  # 综合有效性掩码
            # Masked-out lanes still take part in the address arithmetic, so give
            # them block 0 to keep every computed offset inside the cache.
            physical_block = tl.where(valid, physical_block, 0).to(tl.int64)  # 无效 lane 用 block 0 占位，保证地址不越界；转 int64 防止大 cache 溢出

            # Cache: (num_blocks, block_size, num_kv_heads, head_dim)
            kv_offset = (physical_block[None, :] * (block_size * num_kv_heads * head_dim)  # 跳到对应物理 block
                         + offs_in_block[None, :] * (num_kv_heads * head_dim)  # 跳到 block 内对应槽位
                         + kv_head_idx * head_dim  # 跳到对应 KV 头
                         + offs_d[:, None])  # 头内维度（转置布局以便与 Q 做内积）

            # Compute attention scores for this chunk
            k = tl.load(k_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)  # 加载本 chunk 的 K block
            k = tl.cast(k, tl.float32)  # 转 float32 计算保证精度
            score = tl.sum(q[:, None] * k, axis=0) * scale  # Q 与各 K 向量做内积并缩放，得到注意力分数
            qk = tl.where(valid, score, -1e10)  # 无效位置填 -1e10，softmax 后权重为 0

            # Online softmax
            m_ij = tl.max(qk)  # 本 chunk 分数的最大值
            m_i_new = tl.maximum(m_i, m_ij)  # 更新全局行最大值
            alpha = tl.exp(m_i - m_i_new)  # 旧累加值的修正因子
            p = tl.exp(qk - m_i_new)  # 本 chunk 的未归一化 softmax 权重

            # Rescale accumulator
            acc = acc * alpha  # 修正之前的输出累加器
            l_i = l_i * alpha  # 修正之前的分母累加器

            # Accumulate weighted values
            v = tl.load(v_cache_ptr + kv_offset, mask=valid[None, :], other=0.0)  # 加载本 chunk 的 V block
            v = tl.cast(v, tl.float32)  # 转 float32 计算
            weight = tl.where(valid, p, 0.0)  # 无效位置的权重清零
            acc = acc + tl.sum(weight[None, :] * v, axis=1)  # 累加加权 V
            l_i = l_i + tl.sum(weight)  # 累加 softmax 分母

            m_i = m_i_new  # 保存新的最大值供下一 chunk 使用
    
    # Normalize
    output = acc / l_i  # 除以 softmax 分母，得到归一化输出
    
    # Store output
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d  # 本 batch 本头的输出线性地址
    tl.store(output_ptr + output_offset, output)  # 写回输出


def paged_attention_decode(
    query: torch.Tensor,  # Q 张量 (batch_size, num_heads, head_dim)
    k_cache: torch.Tensor,  # paged K cache
    v_cache: torch.Tensor,  # paged V cache
    block_tables: torch.Tensor,  # 每条序列的逻辑->物理 block 映射表
    context_lens: torch.Tensor,  # 各序列上下文长度
    scale: float,  # 注意力缩放因子
    num_heads: int,  # Q 头数
    num_kv_heads: int,  # KV 头数
    head_dim: int,  # 每头维度
    block_size: int  # 每个 page block 的 token 数
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.
    
    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
        scale: attention scale factor
    
    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]  # 批大小
    max_num_blocks = block_tables.shape[1]  # 每条序列最多占用的物理 block 数
    
    # Make contiguous
    query = query.contiguous()  # 保证 Q 内存连续供 Triton 寻址
    
    output = torch.empty_like(query)  # 按相同形状分配输出
    
    # Chunk size for processing KV tokens
    BLOCK_N = 64 if head_dim <= 128 else 32  # 根据头维度选择 chunk 大小，平衡共享内存与并行度
    
    grid = (batch_size, num_heads)  # grid: 每个批样本的每个 Q 头一个 program
    
    paged_attention_decode_kernel[grid](  # 启动 decode 阶段的 paged attention kernel
        output,  # 输出指针
        query,  # Q 指针
        k_cache,  # K cache 指针
        v_cache,  # V cache 指针
        block_tables,  # block table 指针
        context_lens,  # 上下文长度指针
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    
    return output  # 返回注意力输出


# Attention：统一的多头注意力层，prefill 用 varlen Flash Attention，decode 用 PagedAttention
class Attention(nn.Module):
    def __init__(  # 构造函数
        self,
        num_heads: int,  # Q 头数
        head_dim: int,  # 每头维度
        scale: float = 1.0,  # 缩放因子基准（实际会再除以 sqrt(head_dim)）
        num_kv_heads: int = None,  # KV 头数（None 表示与 Q 头数相同，即 MHA；小于则为 GQA/MQA）
        block_size: int = 16,  # paged KV cache 每个 block 的 token 数
    ):
        super().__init__()  # 调用父类 nn.Module 的初始化
        self.num_heads = num_heads  # 保存 Q 头数
        self.head_dim = head_dim  # 保存每头维度
        self.scale = scale  # 保存缩放因子基准
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads  # 未指定 KV 头数时退化为标准 MHA
        self.block_size = block_size  # 保存 page block 大小
        self.k_cache = self.v_cache = torch.tensor([])  # KV cache 初始为空张量，由外部引擎分配后再挂载

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # 前向：根据 prefill/decode 分派不同注意力实现
        context = get_context()  # 获取当前推理上下文（slot_mapping、cu_seqlens、block_tables 等）
        k_cache, v_cache = self.k_cache, self.v_cache  # 取出当前挂载的 paged KV cache

        # Store current k, v into cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:  # cache 已分配且提供了槽位映射时才写入
            # Ensure k, v are in the right shape: (num_tokens, num_kv_heads, head_dim)
            if k.dim() == 4:  # 输入是 4D 批处理格式
                # Batched: (B, N, num_kv_heads, head_dim) -> reshape to (B*N, num_kv_heads, head_dim)
                B, N, num_kv_heads, head_dim = k.shape  # 解包形状
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()  # 摊平为 varlen 格式并整理内存
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()  # 对 V 同样处理
            else:
                # Already in correct shape (num_tokens, num_kv_heads, head_dim)
                k_to_store = k.contiguous()  # 已是 3D，直接保证内存连续
                v_to_store = v.contiguous()  # 对 V 同样处理
            
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)  # 启动 Triton kernel 把 K/V 写入 paged cache 的物理槽位

        scale = self.scale / (self.head_dim ** 0.5)  # 计算最终缩放因子 1/sqrt(head_dim)

        if context.is_prefill:  # prefill 阶段：整段 prompt 一次性计算，用 varlen Flash Attention
            # Prefill: use flash attention
            # Varlen mode: (total_tokens, num_heads, head_dim)
            cu_seqlens = context.cu_seqlens_q  # 取各序列的累积长度前缀和
            if cu_seqlens is None:  # prefill 必须提供 varlen 边界信息
                raise ValueError("cu_seqlens_q must be provided for varlen attention")  # 缺失则报错
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale,   # 调用 Flash Attention kernel（内含因果掩码与 GQA 头映射）
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)  # 把多头输出合并为单个 hidden 维度返回
        else:  # decode 阶段：每样本只算一个新 token 的注意力，Q 来自当前步，K/V 从 paged cache 中读取
            o = paged_attention_decode(
                q,   # 当前步的 Q
                k_cache,   # paged K cache
                v_cache,   # paged V cache
                context.block_tables,   # 各序列的 block table
                context.context_lens,   # 各序列上下文长度
                scale,   # 缩放因子
                self.num_heads,   # Q 头数
                self.num_kv_heads,   # KV 头数
                self.head_dim,   # 每头维度
                self.block_size   # page block 大小
            )
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)  # 合并多头输出维度返回


if __name__ == "__main__":  # 作为脚本直接运行时执行基准测试
    # Example usage
    layer = Attention(num_heads=8, head_dim=64).cuda()  # 实例化注意力层（8 头，每头 64 维）并移到 GPU
    B, N, D = 4, 1024, 512  # 批大小、序列长度、QKV 合并维度（8 头 x 64 维）
    q = torch.randn(B, N, D).cuda()  # 生成随机 Q 并移到 GPU
    k = torch.randn(B, N, D).cuda()  # 生成随机 K
    v = torch.randn(B, N, D).cuda()  # 生成随机 V
    layer.k_cache = torch.zeros(B, N, D).cuda()  # 手动分配简单的 K cache（此处按非分页布局演示）
    layer.v_cache = torch.zeros(B, N, D).cuda()  # 分配 V cache
    slot_mapping = torch.arange(N).cuda()  # 槽位映射：token i 写入槽位 i

    for _ in range(10):  # Warm-up iterations  # 预热 10 次，触发 Triton JIT 编译并消除首跑开销
        _ = layer(q, k, v)  # 执行前向计算（预热）

    import time  # 导入时间模块用于计时
    times = []  # 记录每次耗时
    for _ in range(100):  # Timing iterations  # 正式计时 100 次迭代
        torch.cuda.synchronize()  # 同步 CUDA 流保证计时准确
        start_time = time.time()  # 记录开始时间
        output_tensor = layer(q, k, v)  # 执行一次前向计算
        torch.cuda.synchronize()  # 等待 GPU 计算完成
        end_time = time.time()  # 记录结束时间
        times.append(end_time - start_time)  # 累加本次耗时
    avg_time = sum(times) / len(times)  # 计算平均耗时
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")  # 打印平均推理耗时
