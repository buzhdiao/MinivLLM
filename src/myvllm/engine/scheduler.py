from collections import deque  # 导入 deque，作为 waiting/running 两个 FIFO 队列的容器
from myvllm.engine.sequence import Sequence, SequenceStatus  # 导入序列类与状态枚举
from myvllm.engine.block_manager import BlockManager  # 导入块管理器，负责 KV cache 块的分配与回收


# Scheduler：continuous batching 的核心。每一步从 waiting 队列调度 prefill、
# 从 running 队列调度 decode，在 token budget（max_num_batched_tokens）与
# KV cache 块数量的约束下拼装每步的批次，块不足时通过 preemption 抢占回收块
class Scheduler:
    # 初始化：创建块管理器并设置调度上限
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # block manager  # 创建 KV cache 块管理器（max_cached_blocks 为可用物理块总数）
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens  # 每步允许调度的最大 token 数（token budget）
        self.max_num_sequences = max_num_sequences  # 每步批次中允许的最大序列数
        # sequence queue  # 双队列：waiting 等待 prefill，running 处于 continuous batching 中
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos  # 结束符 token id，用于判断停止条件


    # 判断引擎是否已无任何待处理序列（waiting 与 running 均为空）
    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    # 新序列入队：先做容量准入检查，再进入 waiting 队列
    def add_sequence(self, sequence: Sequence):
        # Reject up front what the block manager could never satisfy, otherwise the
        # sequence sits in `waiting` forever and only surfaces as a stalled engine.  # 提前拒绝块管理器永远无法满足的请求，否则序列会永远卡在 waiting 队列，引擎表现为停滞
        capacity = len(self.block_manager.blocks)  # KV cache 的总块数
        if sequence.num_blocks > capacity:  # 序列所需块数超过总容量：必然无法调度
            raise ValueError(
                f"Sequence {sequence.seq_id} needs {sequence.num_blocks} blocks "
                f"({len(sequence)} tokens at block_size={self.block_manager.block_size}) "
                f"but the KV cache only holds {capacity}. "
                f"Raise max_cached_blocks or block_size, or shorten the prompt."
            )  # 报错说明缺多少块，并给出调大 max_cached_blocks/block_size 或缩短 prompt 的建议
        self.waiting.append(sequence)  # 通过准入检查，加入 waiting 队尾


    # 核心：调度下一步的批次。优先调度 waiting 中的 prefill，若无则调度 running 中的 decode
    # 返回 (被调度的序列列表, 是否为 prefill 批次)
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []  # 本步被选中的序列
        current_scheduled_tokens = 0  # 本步已用的 token budget
        # An empty schedule is only legitimate when this call freed blocks by
        # preempting, so the next call can make progress. See the guard below.  # 空调度只有在本次通过 preemption 释放了块时才合法，下一步才有进展（见下方守卫）
        preempted = False
        # try schedule for prefilling from waiting queue if not exceeding limits  # 阶段一：尝试从 waiting 队列调度 prefill（受块数与 token budget 限制）
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]  # 窥探队首序列（FIFO，避免饥饿）
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting  # 空闲块充足且 token budget 够用：从 waiting 队列移除
                self.block_manager.allocate(seq)  # 为其分配 KV cache 块（含 prefix cache 命中处理）
                seq.status = SequenceStatus.RUNNING  # 状态置为运行中
                self.running.append(seq)  # 加入 running 队列参与后续 decode
                scheduled_sequences.append(seq)  # 加入本步批次
                current_scheduled_tokens += len(seq)  # prefill 一次计入整条序列的全部 token
            else:
                break  # 队首放不下则停止（FIFO 严格序，不跳过以避免饥饿）
        if scheduled_sequences:  # 本步成功调度了 prefill：整步作为 prefill 批次执行
            return scheduled_sequences, True
        
        # try schedule for completion from running queue  # 阶段二：从 running 队列调度 decode（每序列每步只生成一个 token）
        while self.running:
            seq = self.running.popleft()  # 取出队首序列
            # use can_append to check whether we can append one more token  # 检查追加一个 token 是否需要新块、且有空闲块
            if not self.block_manager.can_append(seq):
                preempted = True  # 块不足：标记发生过抢占，本次空调度才合法
                if self.running:  # 队列中还有其他序列：抢占最后一条，把块让给当前序列
                    self.running.appendleft(seq)  # 当前序列放回队首
                    self.preempt(self.running.pop())  # 抢占队尾序列（释放其全部块）
                else:  # 只剩当前序列且块不足：抢占它自己并结束本轮
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)  # token budget 或批内序列数已满：放回队首，结束本轮
                    break
                # append one token  # 正常调度：为该序列追加一个 token 的块位
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)  # 加入本步 decode 批次
                current_scheduled_tokens += 1 # only one token for completion  # decode 每序列每步只消耗 1 个 token budget

        # re-add to running queue in the same order  # 把本步被调度的序列按原顺序放回 running 队列（保持公平轮转）
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))
        elif not preempted and (self.waiting or self.running):
            # Nothing was scheduled and nothing was preempted, so no engine state
            # changed: every later schedule() would take the same decisions and
            # LLMEngine.generate() would spin forever. Fail loudly instead.  # 什么都没调度也没抢占，引擎状态无变化，后续每次 schedule() 都会是同样的结果，generate() 会死循环；这里显式报错而非静默自旋
            raise RuntimeError(
                "Scheduler made no progress: "
                f"{len(self.waiting)} waiting and {len(self.running)} running sequences, "
                f"{len(self.block_manager.free_block_ids)} of "
                f"{len(self.block_manager.blocks)} blocks free. "
                "This means either a sequence that cannot fit in the KV cache, or "
                "blocks leaked because their ref_count never returned to 0."
            )  # 报错原因：要么有条序列装不下 KV cache，要么块泄漏（ref_count 未归零）

        return scheduled_sequences, False  # 返回 decode 批次（is_prefill=False）


    # preemption（抢占）：回收序列的全部 KV cache 块并放回 waiting 队首，
    # 让更高优先级的请求先行；该序列稍后重新 prefill（可借 prefix cache 减少重算）
    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)  # 释放该序列占用的所有块
        seq.status = SequenceStatus.WAITING  # 状态回退为等待
        self.waiting.appendleft(seq)  # 放回 waiting 队首，优先重新调度


    # postprocess after generation to check whether sequences are finished  # 生成后的后处理：把采样的 token 写回各序列并判断是否触发停止条件
    # if finished, deallocate blocks  # 已完成的序列释放其 KV cache 块
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):  # 逐序列处理采样结果
            seq.append_token(token_id)  # 把新 token 追加进序列
            # Check stopping conditions:  # 检查停止条件
            # EOS token  # 条件一：生成了 EOS（且未被 ignore_eos 忽略）
            # Reached max_tokens limit (number of completion tokens)  # 条件二：补全 token 数达到 max_tokens 上限
            # Reached max_model_length limit (total sequence length including prompt)  # 条件三：总长度（含 prompt）达到 max_model_length 上限
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:  # 任一条件满足即结束
                seq.status = SequenceStatus.FINISHED  # 状态置为已完成
                self.block_manager.deallocate(seq)  # 释放该序列占用的全部 KV cache 块
                self.running.remove(seq)  # 从 running 队列移除
