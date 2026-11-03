import torch  # 导入 PyTorch 主库
from torch import nn  # 导入神经网络模块基类
import torch.nn.functional as F  # 导入函数式接口（embedding 等）
import torch.distributed as dist  # 导入 PyTorch 分布式通信模块

from myvllm.utils import get_context  # 导入获取当前推理上下文（prefill/decode 等信息）的工具函数


# vocabparallelembedding
# shard over the number of vocab, not the embedding size

# VocabParallelEmbedding：词表并行嵌入层——按词表维度（而非 embedding 维度）在 TP 各 rank 间分片
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):  # num_embeddings 为词表大小，embedding_dim 为词向量维度
        super().__init__()  # 调用父类 nn.Module 的初始化
        self.tp_size = dist.get_world_size()  # 获取张量并行（TP）的总进程数
        self.tp_rank = dist.get_rank()  # 获取当前进程在 TP 组中的 rank

        # keep the original num_embeddings
        self.num_embeddings = num_embeddings  # 保存原始词表大小（分片后需要用它截断 padding 部分）
        # pad to make it divisible by tp_size
        self.padded_num_embeddings = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size  # 向上取整补齐词表大小，使其能被 tp_size 整除
        # this is the num_embeddings per partition in this current GPU
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size  # 每个 GPU 分片持有的词表行数
        self.embedding_dim = embedding_dim  # 保存词向量维度

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))  # 每个 rank 只分配自己分片的嵌入矩阵 (vocab/tp, dim)
        self.weight.weight_loader = self.weight_loader  # 挂载加载函数，从完整 checkpoint 中切出本 rank 的分片

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):  # 从完整词表权重中加载本 rank 对应的分片
        param_data = param.data  # 取出本 rank 参数的实际存储

        offset = self.tp_rank * self.num_embeddings_per_partition  # 本分片在完整词表中的起始行号
        shard_size = self.num_embeddings_per_partition  # 本分片的行数

        # calculate how much of the original vocab falls in this partition
        actual_start = min(offset, self.num_embeddings)  # 分片起点与原始词表大小的较小值（处理越界）
        actual_end = min(offset + shard_size, self.num_embeddings)  # 分片终点截断到原始词表大小
        actual_size = max(0, actual_end - actual_start)  # 本分片真正覆盖原始词表的行数（可能为 0）

        if actual_size > 0:  # 若本分片覆盖部分真实词表
            # load the actual weights
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size)  # 从完整权重中切出本分片负责的行
            param_data[:actual_size].copy_(sharded_weights)  # 拷贝到本 rank 的参数中

        # pad the rest with zeros if needed
        if actual_size < shard_size:  # 若因 padding 导致分片比真实词表大
            param_data[actual_size:].zero_()  # 多余的 padding 行用 0 填充

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向：x 为 token id 张量
        # mask for tokens in this partition's range and within original vocab size
        # 逐条件构造掩码：token id 需落在本 rank 负责的 [下界, 上界) 区间内，且未超出原始词表大小（排除 padding 区间）
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
               (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) & \
               (x < self.num_embeddings)
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)  # 把全局 id 平移为本分片内的局部 id；区间外的 id 被置 0（仅占位）
        output = F.embedding(x, self.weight)  # 用本 rank 的分片嵌入矩阵查表，得到局部嵌入

        if dist.get_world_size() > 1:  # 多卡时才需要跨 rank 汇总
            # need to mask again, otherwise the embedding for the out-of-range ids will be the embedding of id 0
            output = mask.unsqueeze(1) * output  # 再次置零：区间外的 token 查到的是局部 id 0 的嵌入，必须清掉
            dist.all_reduce(output, op=dist.ReduceOp.SUM)  # 各 rank 的结果求和（每个 token 只在一个 rank 上非零），得到完整嵌入
        return output  # 返回完整的词向量

# weight tying with embedding layer
# ParallelLMHead：语言模型输出头（与嵌入层权重绑定），同样按词表并行分片，输出 logits
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):  # 参数含义与嵌入层相同
        super().__init__(num_embeddings, embedding_dim)  # 复用词表并行嵌入层的分片逻辑与权重

    # x: [batch_size, seq_len, hidden_size]
    # weight: [vocab_size_per_partition, hidden_size]
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x 为各 token 的隐状态
        context = get_context()  # 获取当前推理上下文
        if context.is_prefill:  # prefill 阶段只需取每条序列最后一个 token 的隐状态来预测下一个 token
            # cu_seqlens_q = [0, 5, 8, 12]
            # last_indices = [5, 8, 12] - 1 = [4, 7, 11]
            last_token = context.cu_seqlens_q[1:] - 1  # exclude the first element which is 0  # varlen 拼接下各序列的累积偏移减 1，即每条序列最后一个 token 的索引
            x = x[last_token].contiguous()  # 取出各序列最后一个 token 的隐状态并整理为连续内存

        # logits: [batch_size, seq_len, vocab_size_per_partition]
        # F.linear automatically transpose the weight
        logits = torch.nn.functional.linear(x, self.weight)  # 与本 rank 分片词表做矩阵乘，得到局部 logits（F.linear 内部会转置权重）
        if self.tp_size > 1:  # 多卡时把各 rank 的局部 logits 拼成完整 logits
            # prepare for all_gather only for GPU 0 which is the main GPU
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None  # 只有 rank 0 需要预先分配接收缓冲区，其余 rank 传 None
            # dist.gather collects the logits from all GPUs to GPU 0
            dist.gather(logits, gather_list=all_logits, dst=0)  # 把各 rank 的局部 logits 收集到 GPU 0
            # concatenate
            if self.tp_rank == 0:  # 仅 rank 0 做拼接与截断
                # [batch_size, seq_len, padded_vocab_size]
                logits = torch.cat(all_logits, dim=-1)  # 沿词表维拼接各分片的 logits
                # trim to original vocab size
                logits = logits[..., :self.num_embeddings]  # 截掉 padding 部分，恢复原始词表大小

        return logits  # 返回（rank 0 上完整的）logits
