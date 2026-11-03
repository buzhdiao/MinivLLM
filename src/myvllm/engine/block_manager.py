import xxhash  # 导入 xxhash，用于前缀缓存的非加密快速哈希
import numpy as np  # 导入 numpy，将 token_ids 转字节流参与哈希
from collections import deque  # 导入 deque，作为 O(1) 的空闲块队列

from myvllm.engine.sequence import Sequence  # 导入 Sequence，块管理以序列为单位操作


# Block：物理 KV cache 块的元数据（不存 KV 张量本身，张量在统一的大 cache 池中，
# 这里只维护块的引用计数、哈希与所含 token，用于 PagedAttention 的块管理与前缀缓存）
class Block:
    # 初始化块元数据：块号、哈希、引用计数与所含 token
    def __init__(self, block_id):
        self.block_id = block_id  # 物理 KV cache 的块编号
        self.hash = -1  # 块内容的前缀哈希；-1 表示未满块或未计算（不参与前缀缓存）
        self.ref_count = 0  # 引用计数：有多少序列正在使用该块（0 表示空闲可复用）
        self.token_ids = []  # 该块内已写入的 token 列表，用于前缀缓存匹配


    # 用新的哈希与 token 内容更新块（写入满块内容时调用）
    def update(self, h: int, token_ids: list[int]):
        self.hash = h  # 记录该块的前缀哈希值
        self.token_ids = token_ids  # 记录该块包含的 token，供前缀缓存比对

    # 重置块为"已分配但为空"的状态（分配时调用）
    def reset(self):
        self.hash = -1  # 清空哈希：新内容尚未定型，不参与前缀缓存匹配
        # reset() is only reached via BlockManager._allocate_block, which takes the
        # block off the free list on behalf of exactly one sequence. Allocation is
        # therefore the first reference: leaving this at 0 makes the matching
        # deallocate() drive ref_count to -1, so the block is never freed.  # reset() 只会经由 _allocate_block 被调用：分配即代表该块获得第一个引用，故这里直接置 1，否则配对的 deallocate 会把 ref_count 减到 -1 导致块永远不释放
        self.ref_count = 1
        self.token_ids = []  # 清空 token 内容

# BlockManager：PagedAttention 的核心显存管理器。把整块 KV cache 池切成固定大小
# 的块（block）按需分配给各序列，并通过哈希表实现 prefix caching（前缀缓存复用）
class BlockManager:
    # 初始化：创建 num_blocks 个块并全部入空闲队列
    def __init__(self, num_blocks: int, block_size: int):
        # block_size: number of tokens per block  # 每个块容纳的 token 数
        self.block_size: int = block_size
        # list of all blocks  # 所有物理块的元数据列表，下标即块号
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash to block id: this is for prefix caching  # 前缀哈希 -> 块号 的映射，用于 prefix caching 命中查询
        self.hash_to_block_id: dict[int, int] = {}
        # free block ids  # 空闲块号队列（左侧取出分配、右侧归还回收）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used block ids  # 已被占用的块号集合，快速判断块是否在使用中
        self.used_block_ids: set[int] = set()

    # given token_ids, compute the hash value  # 给定 token 序列，计算该块的前缀哈希
    # use prefix_hash_value to compute the hash in a context-sensitive way  # 把前一块的哈希作为种子，使哈希"上下文敏感"：只有完整前缀都相同，块哈希才相同
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()  # 创建 64 位 xxhash 哈希器
        if prefix_hash_value != -1:  # 有前缀哈希时，先混入前缀（-1 表示本块是第一块）
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())  # 再混入本块的 token 字节
        return h.intdigest()  # 返回 64 位整数哈希

    # move this block to used list  # 从空闲队列取出指定块并标记为已占用，返回块对象
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]  # 取出块对象
        assert block.ref_count == 0, "Block is already allocated"  # 断言该块当前空闲
        block.reset()  # 重置为"已分配但为空"，引用计数置 1
        self.free_block_ids.remove(block_id)  # 从空闲队列移除
        self.used_block_ids.add(block_id)  # 加入已占用集合
        return block

    # 回收块：引用计数为 0 时归还到空闲队列
    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"  # 断言无序列引用该块
        block = self.blocks[block_id]  # 取出块对象
        # Clearing token_ids deliberately keeps the prefix cache scoped to blocks that
        # are still referenced: a freed block can no longer match in allocate(), so a
        # cache hit never spans a finished sequence. Reuse across sequences is not
        # enabled yet because the prefill path cannot consume it -- the Triton kernel
        # in layers/attention.py attends only over the K/V computed in that pass and
        # ignores context.block_tables, and qwen3 derives RoPE positions from
        # cu_seqlens_q, so both would be wrong by num_cached_tokens. Enabling reuse
        # means a paged prefill kernel (cu_seqlens_q != cu_seqlens_k) plus a position
        # offset; until then this line is what keeps the engine correct.  # 刻意清空 token_ids：已释放块不再参与前缀匹配，避免命中跨越已结束的序列。跨序列复用尚未启用——prefill 路径的 Triton kernel 只关注本次计算的 K/V（忽略 block_tables），且 RoPE 位置取自 cu_seqlens_q，若复用二者都会偏移 num_cached_tokens；启用复用需要 paged prefill kernel（cu_seqlens_q != cu_seqlens_k）加位置偏移
        block.token_ids = []
        self.used_block_ids.remove(block_id)  # 从已占用集合移除
        self.free_block_ids.append(block_id)  # 归还到空闲队列尾部

    # whether we can allocate a block for this sequence  # 判断空闲块是否足够放下整个序列（prefill 调度前的准入检查）
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks  # 空闲块数 >= 序列所需块数才可分配


    # 为序列分配全部所需块：优先尝试前缀缓存命中，未命中的块从空闲队列分配
    def allocate(self, seq: Sequence) -> None:
        h = -1  # 前缀哈希种子，-1 表示第一块没有前缀
        for i in range(seq.num_blocks):  # 逐块处理序列的每个逻辑块
            no_cache_found = False  # 标记本块是否前缀缓存未命中

            token_ids = seq.block(i)  # 取出第 i 块覆盖的 token 片段
            # only compute hash for full blocks, always -1 for partial blocks  # 只有满块才计算前缀哈希；未满块（最后一块）哈希恒为 -1，不参与缓存
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)  # 查前缀缓存：该哈希是否已有对应块（-1 表示未命中）
            
            # if cache miss or hash collision  # 未命中，或命中了但发生哈希碰撞（内容不同）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True  # 按缓存未命中处理

            if not no_cache_found:  # 前缀缓存命中：直接复用已有块
                # update sequence information  # 更新序列信息：跳过已缓存的 token，prefill 时不再重复计算
                seq.num_cached_tokens += self.block_size # which == len(token_ids)
                # update block information, considering the edge case that the block is not allocated yet but with hash code  # 更新块信息，需处理"有哈希记录但块尚未被占用"的边界情况
                if block_id not in self.used_block_ids:
                    # Unreachable while _deallocate_block clears token_ids: a freed
                    # block has token_ids == [] and fails the match above. Kept for
                    # when cross-sequence reuse is enabled.  # 当前不可达：释放时已清空 token_ids，命中分支不会走到这里；保留以备将来启用跨序列复用
                    block = self._allocate_block(block_id)
                else:
                    # update block information  # 块已在使用：引用计数加一（多个序列共享同一物理块，即前缀共享）
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:  # 缓存未命中：分配新的物理块
                block = self._allocate_block(self.free_block_ids[0])  # 取空闲队列队首块
                block.update(h=h, token_ids=token_ids)  # 写入哈希与 token 内容
                if h != -1:  # 满块才登记进前缀缓存表
                    self.hash_to_block_id[h] = block.block_id
            seq.block_table.append(block.block_id)  # 把物理块号追加进序列的 block_table（块表）
        
    # 释放序列占用的所有块：引用计数减一，减到 0 则物理回收
    def deallocate(self, seq: Sequence) -> None:
        # update block information  # 逐个处理序列块表中的块
        for block_id in seq.block_table:
            block = self.blocks[block_id]  # 取出块对象
            block.ref_count -= 1  # 引用计数减一
            if block.ref_count == 0:  # 无人引用时物理回收该块
                self._deallocate_block(block_id)
        # update sequence information  # 清空序列的块表与缓存计数（preemption 复用前重置状态）
        seq.block_table = []
        seq.num_cached_tokens = 0

    # this is to check whether we can append tokens to this sequence  # 检查能否向该序列追加一个新 token（decode 每步一个 token）
    # when that token would require allocating a new block.  # 只有当 token 需要新块时才可能失败
    def can_append(self, seq: Sequence) -> bool:
        # Called after the new token is already counted in seq.num_tokens, so the
        # condition must match append()'s allocation branch: a fresh block is only
        # needed when that token is the first of a new block (num_tokens % size == 1).
        # At num_tokens % size == 0 the token still fits in the block the sequence
        # already holds, and append() merely finalizes its hash.  # 调用时新 token 已计入 num_tokens：只有当它是新块的首个 token（num_tokens % size == 1）才需要新块；余数为 0 时 token 仍在现有末块内，append() 只会补算哈希
        if seq.num_tokens % self.block_size == 1:
            return len(self.free_block_ids) > 0  # 需要新块：只要有空闲块即可
        return True  # 不需要新块：一定可以追加

    # this is the actual work to append tokens to this sequence  # 实际执行追加：为新 token 落实 KV cache 块（此时 token 已计入序列信息，但块尚未落实）
    # this is called when the new token has been added to the seq information  # 
    # but no block in gpu has yet allocate for it  # 
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table  # 取出序列的块表
        last_block_for_seq_id = block_tables[-1]  # 末块（最后 token 所在块）的块号

        # if the last block is now full, compute hash  # 末块刚好写满：登记其内容哈希，纳入前缀缓存
        if seq.num_tokens % self.block_size == 0:
            h = self.compute_hash(token_ids = seq.block(seq.num_blocks - 1), prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)  # 计算末块的前缀哈希（第一块无前缀，否则链上前一块哈希）
            block = self.blocks[last_block_for_seq_id]  # 取出末块对象
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))  # 写入满块哈希与内容
            self.hash_to_block_id[h] = block.block_id  # 登记到前缀缓存映射表
        # if one new block is needed  # 新 token 是新块的首个 token：需要分配新块
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized  # 断言上一块已定型（满块必有哈希）
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])  # 从空闲队列分配新块
            block_tables.append(block.block_id)  # 新块号追加进块表
        # else, do nothing  # 其余情况：token 落在现有未满块内，无需新块
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"  # 断言末块已分配
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"  # 断言末块是未满块（哈希为 -1）
