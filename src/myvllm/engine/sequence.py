from enum import Enum, auto  # 导入 Enum 与 auto，用于定义序列状态枚举
import math  # 导入 math，用于按 block_size 向上取整计算块数
from itertools import count  # 导入 count，用于生成全局自增的序列 ID
from myvllm.sampling_parameters import SamplingParams  # 导入采样参数类，提供 temperature/max_tokens 等默认值
from copy import copy  # 导入浅拷贝，使 token_ids 是独立的新列表，不受外部修改影响


# SequenceStatus：序列状态枚举，描述请求在调度器（continuous batching）中的生命周期阶段
class SequenceStatus(Enum):
    WAITING = auto()  # 等待中：位于 waiting 队列，尚未被调度做 prefill
    RUNNING = auto()  # 运行中：已分配 KV cache 块，正在被调度做 decode
    FINISHED = auto()  # 已完成：触发停止条件（EOS / max_tokens / max_model_length），已释放块


# Sequence：一条请求的抽象。维护 token_ids、KV cache 的 block_table（块表，
# 记录该序列每个逻辑块对应的物理块号）以及采样参数，是调度与显存管理的基本单元
class Sequence:
    counter = count()  # 类级别计数器，为每条 Sequence 生成全局唯一自增的 seq_id

    # 初始化序列：记录 token_ids、状态、块表和采样相关信息
    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        self.block_size = block_size # number of tokens per block  # 每个块容纳的 token 数
        # record sequence id  # 记录全局唯一的序列 ID
        self.seq_id = next(Sequence.counter)
        # status  # 初始状态为等待调度
        self.status = SequenceStatus.WAITING
        # token ids, need copy so that it is a new list, won't be affected by outside changes  # 浅拷贝一份 token 列表，避免外部修改影响
        self.token_ids = copy(token_ids)
        # last token  # 记录最后一个 token（decode 阶段的输入）
        self.last_token = self.token_ids[-1] if self.token_ids else None
        # num_tokens, num_prompt_tokens  # 当前 token 总数与 prompt token 数（初始相等）
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        # num_cached_tokens = 0  # 命中 prefix cache 的 token 数，初始为 0
        self.num_cached_tokens = 0
        # block_table  # 块表：该序列各逻辑块对应的物理 KV cache 块号，初始为空
        self.block_table = []
        # sampling_params' related things  # 从采样参数中取出本序列需要的字段
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length

    # 返回序列当前 token 总数（含 prompt 与已生成部分），使 len(seq) 可用
    def __len__(self):
        return self.num_tokens

    # 支持下标访问，直接透传到 token_ids
    def __getitem__(self, idx):
        return self.token_ids[idx]

    # 判断序列是否已完成
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 已生成的补全（completion）token 数 = 当前总数 - prompt 数
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    # 返回 prompt 部分的 token（前 num_prompt_tokens 个）
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    # 返回已生成的补全部分 token（num_prompt_tokens 之后的）
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 命中 prefix cache 的块数：cache 按"完整块"复用，故对 cached token 数向上取整
    @property
    def num_cached_blocks(self):
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    # 序列当前占用的总块数：token 数按 block_size 向上取整
    @property
    def num_blocks(self):
        return int(math.ceil(self.num_tokens / self.block_size))

    # 最后一个块内的 token 数（不满一块时也正确）
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - max(self.num_blocks - 1, 0) * self.block_size

    # 返回第 i 个块对应的 token 片段；最后一块为尾部剩余部分
    def block(self, i):
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"  # 校验块下标合法
        if i == self.num_blocks - 1:  # 最后一块：取 token_ids 尾部 last_block_num_tokens 个
            return self.token_ids[-self.last_block_num_tokens:]
        else:  # 非最后一块：按块大小精确切片
            start_idx = i * self.block_size  # 本块起始下标
            end_idx = start_idx + self.block_size  # 本块结束下标（不含）
            return self.token_ids[start_idx : end_idx]

    # 将新生成的 token 追加进序列，并同步更新 last_token 与 token 计数
    def append_token(self, token_id):
        self.token_ids.append(token_id)  # 追加到 token 序列末尾
        self.last_token = token_id  # 更新最后 token，供下一轮 decode 使用
        self.num_tokens += 1  # token 总数加一

    # 自定义 pickle 序列化内容：decode 阶段只传 last_token 而非全部 token_ids，
    # 以减少主进程与 worker 进程间（经共享内存/管道）的序列化开销
    def __getstate__(self):
        return (
            self.num_tokens,  # token 总数
            self.num_prompt_tokens,  # prompt token 数
            self.num_cached_tokens,  # prefix cache 命中的 token 数
            self.block_table,  # KV cache 块表
            self.token_ids if self.num_completion_tokens == 0 else self.last_token  # prefill 时传完整 token 列表，decode 时只传最后一个 token
        )

    # 从 pickle 状态恢复序列字段（与 __getstate__ 配对）
    def __setstate__(self, state):
        (
            self.num_tokens,  # token 总数
            self.num_prompt_tokens,  # prompt token 数
            self.num_cached_tokens,  # prefix cache 命中的 token 数
            self.block_table,  # KV cache 块表
            last_token_or_ids  # 末尾字段：可能是完整 token 列表或单个 last_token
        ) = state
        # Check if this is prefill (num_completion_tokens == 0) or decode phase  # 判断是 prefill（尚无补全 token）还是 decode 阶段
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens  # 计算补全 token 数
        if num_completion_tokens == 0:
            # Prefill: last_token_or_ids is the full token_ids list  # prefill：末尾字段即完整的 token_ids 列表
            self.token_ids = last_token_or_ids
        else:
            # Decode: last_token_or_ids is just the last token  # decode：末尾字段只是最后一个 token，重建为单元素列表
            self.token_ids = [last_token_or_ids]
        # Restore last_token attribute  # 恢复 last_token 属性
        self.last_token = self.token_ids[-1] if self.token_ids else None
