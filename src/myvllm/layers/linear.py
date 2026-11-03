import torch.nn as nn  # 导入神经网络模块基类
import torch  # 导入 PyTorch 主库
import torch.distributed as dist  # 导入分布式通信模块（all_reduce/gather 等）
import os  # 导入 os 模块，读取 torchrun 注入的环境变量

# LinearBase：各类张量并行线性层的公共基类，负责按 TP 分片创建权重并挂载 weight_loader
class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(  # 构造函数
        self, 
        input_size: int,  # 输入维度
        output_size: int,  # 输出维度（可传入本 rank 分片后的尺寸）
        bias: bool = True,  # 是否使用偏置
        tp_dim: int | None = None  # 分片所在维度：0 表示按输出维（列并行）分片，1 表示按输入维（行并行）分片
    ):
        super().__init__()  # 调用父类 nn.Module 的初始化
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim  # 保存分片维度信息
        self.tp_rank = dist.get_rank()  # 当前进程在 TP 组中的 rank
        self.tp_size = dist.get_world_size()  # TP 组的进程总数
        
        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))  # 创建权重矩阵 (output_size, input_size)，每个 rank 只持有自己的分片
        self.weight.weight_loader = self.weight_loader  # 挂载自定义加载函数，用于从完整 checkpoint 中切出本 rank 分片

        # create bias parameter
        if bias:  # 需要偏置时
            self.bias = nn.Parameter(torch.zeros(output_size))  # 创建全零偏置
            self.bias.weight_loader = self.weight_loader  # 同样挂载加载函数
        else:  # 不需要偏置
            self.register_parameter('bias', None)  # 显式注册 bias=None，保证模块结构一致

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):  # 权重分片加载接口，由子类实现
        raise NotImplementedError("Subclasses should implement this method.")  # 基类不实现，强制子类覆写

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向计算接口，由子类实现
        raise NotImplementedError("Subclasses should implement this method.")  # 基类不实现，强制子类覆写

"""
these functions are for is that we deploy a maybe randomly initialized model on GPU using some tensor/pipeline parallel method
then we wanna load a saved model checkpoint to it

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""

# the simpliest Linear layer: ReplicatedLinear(LinearBase)
# where we simply copy the weight as the weight_loader
# and run the forward as a normal linear layer
# ReplicatedLinear：不做任何分片、每个 rank 持有完整权重的普通线性层
class ReplicatedLinear(LinearBase):
    def __init__(  # 构造函数
        self, 
        input_size: int,  # 输入维度
        output_size: int,  # 输出维度
        bias: bool = True  # 是否使用偏置
    ):
        super().__init__(input_size, output_size, bias)  # 调用基类构造，不指定 tp_dim（不分片）

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):  # 直接整份加载权重
        param.data.copy_(loaded_weights)  # 把完整 checkpoint 权重原样拷贝到本参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向：普通线性变换
        return nn.functional.linear(x, self.weight, self.bias)  # y = xW^T + b，权重不分片，无需通信

# columnsplit Linear layer: ColumnParallelLinear(LinearBase)
# get the original full parameter
# compute the starting index of the column split
# compute the dim size of the full parameter
# copy the parameter slice to the local parameter
# ColumnParallelLinear：列并行线性层——把权重按输出维（列）切成 tp_size 份，各 rank 持有列分片
class ColumnParallelLinear(LinearBase):
    def __init__(  # 构造函数
        self, 
        input_size: int,  # 输入维度（不做分片，各 rank 相同）
        output_size: int,  # 完整输出维度，将被均分给各 rank
        bias: bool = True,  # 是否使用偏置
    ):
        tp_size = dist.get_world_size()  # TP 进程总数
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."  # 输出维度必须能被 tp_size 整除，否则无法均分
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)  # 本 rank 只创建 output_size/tp_size 行的权重；tp_dim=0 表示按输出维分片

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):  # 从完整权重中切出本 rank 负责的列分片
        param_data = param.data  # 本 rank 参数的实际存储
        # full_dim on the output column
        full_data_output_size = loaded_weights.size(0)  # 完整权重的输出维尺寸
        # dim size after sharding
        shard_size = full_data_output_size // self.tp_size  # 每个 rank 分到的行数
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."  # 校验分片尺寸与本 rank 参数一致
        # starting index
        start_index = self.tp_rank * shard_size  # 本 rank 分片在完整权重中的起始行
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)  # 沿输出维（dim 0）切出本 rank 的分片
        param_data.copy_(slided_weight)  # 拷贝到本 rank 参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向：各 rank 独立计算自己的输出列分片
        return nn.functional.linear(x, self.weight, self.bias)  # 输入完整、输出按列分片；无需通信（激活函数可紧随其后在分片上计算）

# an extension of ColumnParallelLinear by merging several matrices
# MergedColumnParallelLinear：把多个列并行矩阵（如 Q、K、V）合并成一个大矩阵一次计算，加载时再按偏移切分
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(  # 构造函数
        self, 
        input_size: int,  # 输入维度
        output_sizes: list[int], # e.g. merge QKV matrices to compute MM together and then split  # 各子矩阵的输出维度列表，例如 [Q, K, V] 各自的输出尺寸
        bias: bool = True,  # 是否使用偏置
    ):
        self.output_sizes = output_sizes  # 保存各子矩阵输出维度，weight_loader 需要用它计算偏移
        super().__init__(input_size, sum(output_sizes), bias)  # 按总输出维度创建列并行层

    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # the index of merged matrices (e.g. it's 0 for Q, 1 for K, 2 for V assuming QKV are merged together)
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):  # loaded_weight_id 指明加载的是第几个子矩阵（0=Q,1=K,2=V）
        """
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),  
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        load to 
        merged_layer = Linear(
            input_size=4096,
            output_sizes=sum([4096, 4096, 4096]),  # Q, K, V
        ) which is also sharded by tp_size
        """
        param_data = param.data  # 本 rank 合并后的参数存储
        # compute offset 
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size  # 前面若干子矩阵在分片后参数中的行偏移
        # compute size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size  # 当前子矩阵在本 rank 的分片行数
        # find the correct slice to be loaded in the sharded parameter
        param_data = param_data.narrow(0, offset, shard_size)  # 在本 rank 参数中定位该子矩阵对应的行区间
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size  # 完整子矩阵权重中本 rank 分片的起始行
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)  # 切出本 rank 的分片
        param_data.copy_(shard_weights)  # 写入参数对应区间


# QKVColumnParallelLinear：面向 GQA 的 QKV 合并列并行层，Q 用 num_heads、K/V 用 num_kv_heads 分片
class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(  # 构造函数
        self,
        input_size: int,  # 输入维度
        head_size: int,  # 每个注意力头的维度
        num_heads: int,  # 完整的 Q 头数
        num_kv_heads: int | None = None,  # KV 头数（GQA 时小于 num_heads；None 则退化为 MHA）
        bias: bool = False,  # 是否使用偏置
    ):
        self.tp_size = dist.get_world_size()  # TP 进程总数
        num_kv_heads = num_kv_heads or num_heads  # 未指定 KV 头数时与 Q 头数相同（标准 MHA）
        self.head_size = head_size  # 保存每头维度
        self.num_heads = num_heads // self.tp_size  # 本 rank 分到的 Q 头数
        self.num_kv_heads = num_kv_heads // self.tp_size  # 本 rank 分到的 KV 头数（GQA 分组对应切分）
        # Calculate per-GPU output size
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)  # 本 rank 输出维度 = Q 头 + K 头 + V 头的总维度
        # Pass TOTAL output size to parent (it will divide by tp_size)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)  # 完整输出维度，交给父类按 tp_size 均分
        super().__init__(input_size, total_output_size, bias=bias)  # 调用父类构造完成分片参数创建

    # load_weight_id: q, k, v
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):  # load_weight_id 指明加载 'q'/'k'/'v' 哪个权重
        # batch_size * num_heads * num_token * head_size
        param_data = param.data  # 本 rank 合并后的 QKV 参数存储
        # loaded_weights: batch_size * num_token * (head_size*num_heads)
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"  # 校验权重标识合法
        # compute offset
        if load_weight_id == 'q':  # Q 位于合并参数的最前面
            offset = 0  # Q 的偏移为 0
            shard_size = self.head_size * self.num_heads  # Q 分片的行数（本 rank Q 头总维度）
        elif load_weight_id == 'k':  # K 紧跟在 Q 之后
            offset = self.head_size * self.num_heads  # K 的行偏移 = Q 区间大小
            shard_size = self.head_size * self.num_kv_heads  # K 分片的行数（本 rank KV 头总维度）
        elif load_weight_id == 'v':  # V 紧跟在 K 之后
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads  # V 的行偏移 = Q + K 区间大小
            shard_size = self.head_size * self.num_kv_heads  # V 分片的行数
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")  # 理论上不会到达（前面已 assert）

        param_data = param_data.narrow(0, offset, shard_size)  # 在本 rank 参数中定位该矩阵对应的行区间
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size  # 完整权重中本 rank 分片的起始行
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)  # 切出本 rank 的分片

        param_data.copy_(shard_weights)  # 写入参数对应区间


# RowParallelLinear：行并行线性层——把权重按输入维（行）切成 tp_size 份，输入也分片，输出需 all_reduce 求和
class RowParallelLinear(LinearBase):
    def __init__(  # 构造函数
        self,
        input_size: int,  # 完整输入维度，将被均分给各 rank
        output_size: int,  # 输出维度（不分片，各 rank 相同）
        bias: bool = True,  # 是否使用偏置（只在 all_reduce 后加一次才正确，故通常各 rank 只存一份）
    ):
        tp_size = dist.get_world_size()  # TP 进程总数
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."  # 输入维度必须能被 tp_size 整除
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)  # 本 rank 只创建 input_size/tp_size 列的权重；tp_dim=1 表示按输入维分片

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):  # 从完整权重中切出本 rank 负责的行分片
        param_data = param.data  # 本 rank 参数的实际存储
        # full_dim on the input row
        full_data_input_size = loaded_weights.size(1)  # 完整权重的输入维尺寸
        # dim size after sharding
        shard_size = full_data_input_size // self.tp_size  # 每个 rank 分到的列数
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."  # 校验分片尺寸与本 rank 参数一致
        # starting index
        start_index = self.tp_rank * shard_size  # 本 rank 分片在完整权重中的起始列
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)  # 沿输入维（dim 1）切出本 rank 的分片
        param_data.copy_(slided_weight)  # 拷贝到本 rank 参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向：输入需先按行分片，各 rank 算部分和后归约
        result = nn.functional.linear(x, self.weight, self.bias)  # 用本 rank 的行分片计算部分和
        if self.tp_size > 1:  # 多卡时需要跨 rank 求和
            dist.all_reduce(result, op=dist.ReduceOp.SUM)  # all_reduce 汇总各 rank 的部分和，得到完整输出
        return result  # 返回完整输出（各 rank 相同）




if __name__ == "__main__":  # 作为脚本直接运行时执行各并行层的正确性测试
    # how to run?
    # 1. cd src/myvllm/layers
    # 2. CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 linear.py # 4 GPUs

    def _init_dist():  # 初始化分布式进程组，返回 rank/世界规模/本地 rank/设备
        rank = int(os.environ["RANK"])  # 读取全局 rank（torchrun 注入）
        world_size = int(os.environ["WORLD_SIZE"])  # 读取进程总数（torchrun 注入）
        local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 读取本机 rank，缺省为 0
        backend = "nccl" if torch.cuda.is_available() else "gloo"  # 有 GPU 用 nccl，否则用 gloo（CPU）

        if torch.cuda.is_available():  # GPU 环境
            torch.cuda.set_device(local_rank)  # 绑定当前进程使用的 GPU
            device = torch.device("cuda", local_rank)  # 指定设备为本机对应 GPU
            dist.init_process_group(  # 初始化进程组
                backend=backend,  # 通信后端 nccl
                init_method="env://",  # 通过环境变量获取 rendezvous 信息
                device_id=local_rank,  # 绑定设备 id
                ) 
        else:  # CPU 环境
            device = torch.device("cpu")  # 使用 CPU 设备
            dist.init_process_group( backend=backend, init_method="env://", )  # 用 gloo 后端初始化进程组
        return rank, world_size, local_rank, device  # 返回初始化结果

    # Single linear layer and column parallel test
    @torch.no_grad()  # 测试不需要梯度
    def test_column_parallel(device):  # 列并行与单卡参考结果的一致性测试
        tp_rank = dist.get_rank()  # 当前 rank
        tp_size = dist.get_world_size()  # Number of parallel GPUs  # 并行 GPU 数

        in_features = 1024 * tp_size  # 输入维度取 tp_size 的倍数保证可整除
        out_features = 1024 * tp_size  # 输出维度同样保证可整除
        batch = 4  # 批大小

        # Ensure that each rank gets exactly the same full input/weight
        g = torch.Generator(device="cpu").manual_seed(2026)  # 固定随机种子，保证所有 rank 生成相同的完整数据
        x_full = torch.randn(batch, in_features, generator=g)  # 生成完整输入
        w_full = torch.randn(out_features, in_features, generator=g)  # 生成完整权重
        b_full = torch.randn(out_features, generator=g)  # 生成完整偏置

        x_full = x_full.to(device)  # 完整输入移到本 rank 设备
        w_full = w_full.to(device)  # 完整权重移到本 rank 设备
        b_full = b_full.to(device)  # 完整偏置移到本 rank 设备

        # reference (single GPU)
        single_layer = ReplicatedLinear(in_features, out_features, bias=True).to(device)  # 构建不分片的参考层
        single_layer.weight.weight_loader(single_layer.weight, w_full)  # 整份加载参考权重
        single_layer.bias.weight_loader(single_layer.bias, b_full)  # 整份加载参考偏置
        y_single = single_layer(x_full)  # 计算单卡参考输出

        # TP layer (each rank stores out_features/tp)
        col_tp_layer = ColumnParallelLinear(in_features, out_features, bias=True).to(device)  # 构建列并行层（每个 rank 存 out_features/tp）
        col_tp_layer.weight.weight_loader(col_tp_layer.weight, w_full)  # 从完整权重切出本 rank 分片加载
        col_tp_layer.bias.weight_loader(col_tp_layer.bias, b_full)  # 加载本 rank 的偏置分片

        # forward
        y_col_tp = col_tp_layer(x_full)  # [batch, out_features/tp]  # 列并行前向：得到本 rank 的输出列分片

        # Restore full output: all_gather+concat
        y_parts = [torch.empty_like(y_col_tp) for _ in range(tp_size)]  # 为每个 rank 的分片分配接收缓冲区
        dist.all_gather(y_parts, y_col_tp)  # 收集所有 rank 的输出分片
        y_full = torch.cat(y_parts, dim=-1)  # [batch, out_features]  # 沿输出维拼接还原完整输出

        # Alignment check (print only at rank0)
        max_err = (y_full - y_single).abs().max().item()  # 计算与参考结果的最大绝对误差
        ok = torch.allclose(y_full, y_single, rtol=1e-4, atol=1e-4)  # 判断结果是否一致
        if tp_rank == 0:  # 只在 rank 0 打印
            print(f"[ColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")  # 输出测试结果


    # MergedColumnParallelLinear test
    @torch.no_grad()  # 测试不需要梯度
    def test_merged_column_parallel(device):  # 合并式列并行（Q/K/V）与单卡参考的一致性测试
        tp_rank = dist.get_rank()  # 当前 rank
        tp_size = dist.get_world_size()  # 并行 GPU 数

        # Make the dimension automatically adapt to tp_size to ensure divisibility
        in_features = 1024 * tp_size  # 输入维度随 tp_size 缩放保证可整除
        out_each = 512 * tp_size  # 每个子矩阵的输出维度
        out_sizes = [out_each, out_each, out_each]  # Combination of Q, K and V matrices  # Q、K、V 三个子矩阵的输出维度
        batch = 4  # 批大小

        g = torch.Generator(device="cpu").manual_seed(2026)  # 固定种子保证各 rank 数据一致
        x_full = torch.randn(batch, in_features, generator=g)  # 生成完整输入
        w_q = torch.randn(out_sizes[0], in_features, generator=g)  # 生成完整 Q 权重
        w_k = torch.randn(out_sizes[1], in_features, generator=g)  # 生成完整 K 权重
        w_v = torch.randn(out_sizes[2], in_features, generator=g)  # 生成完整 V 权重

        x_full = x_full.to(device)  # 输入移到本 rank 设备
        w_q = w_q.to(device)  # Q 权重移到本 rank 设备
        w_k = w_k.to(device)  # K 权重移到本 rank 设备
        w_v = w_v.to(device)  # V 权重移到本 rank 设备

        # Reference: single-card equivalent output (Q | K | V concat)
        y_ref = torch.cat(  # 参考输出：单卡分别计算 Q/K/V 再拼接
            [
                nn.functional.linear(x_full, w_q, None),  # Q 的单卡计算
                nn.functional.linear(x_full, w_k, None),  # K 的单卡计算
                nn.functional.linear(x_full, w_v, None),  # V 的单卡计算
            ],
            dim=-1,  # 沿输出维拼接
        )

        # TP merged layer
        # NOTE: your MergedColumnParallelLinear defines weight_loader(param, loaded_weights, loaded_weight_id),
        # so bias loader signature doesn't match base (param, loaded_weights). Therefore bias=False here.
        merged = MergedColumnParallelLinear(in_features, out_sizes, bias=False).to(device)  # 构建合并列并行层（bias 加载签名不匹配，故关闭 bias）
        merged.weight_loader(merged.weight, w_q, 0)  # 加载 Q 到合并参数的对应分片区间
        merged.weight_loader(merged.weight, w_k, 1)  # 加载 K
        merged.weight_loader(merged.weight, w_v, 2)  # 加载 V

        y_local = merged(x_full)  # [batch, sum(out_sizes)/tp], layout: [q_local, k_local, v_local]  # 本 rank 的合并输出，布局为 [Q|K|V 的分片]

        # all_gather then re-pack to [Q_all | K_all | V_all]
        y_parts = [torch.empty_like(y_local) for _ in range(tp_size)]  # 为各 rank 输出分配缓冲区
        dist.all_gather(y_parts, y_local)  # 收集所有 rank 的输出分片

        ql = out_sizes[0] // tp_size  # 每个 rank 的 Q 分片宽度
        kl = out_sizes[1] // tp_size  # 每个 rank 的 K 分片宽度
        vl = out_sizes[2] // tp_size  # 每个 rank 的 V 分片宽度

        q_full = torch.cat([p[:, :ql] for p in y_parts], dim=-1)  # 从各 rank 分片中取 Q 部分拼接成完整 Q 输出
        k_full = torch.cat([p[:, ql : ql + kl] for p in y_parts], dim=-1)  # 取 K 部分拼接
        v_full = torch.cat([p[:, ql + kl : ql + kl + vl] for p in y_parts], dim=-1)  # 取 V 部分拼接
        y_full = torch.cat([q_full, k_full, v_full], dim=-1)  # 再拼成完整输出

        max_err = (y_full - y_ref).abs().max().item()  # 计算最大绝对误差
        ok = torch.allclose(y_full, y_ref, rtol=1e-4, atol=1e-4)  # 判断结果是否一致
        if tp_rank == 0:  # 只在 rank 0 打印
            print(f"[MergedColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")  # 输出测试结果


    # QKVColumnParallelLinear test
    @torch.no_grad()  # 测试不需要梯度
    def test_qkv_column_parallel(device):  # GQA 版 QKV 合并列并行与单卡参考的一致性测试
        tp_rank = dist.get_rank()  # 当前 rank
        tp_size = dist.get_world_size()  # 并行 GPU 数

        # make input dim divisible by tp_size
        input_size = 1024 * tp_size  # 输入维度随 tp_size 缩放保证可整除
        head_size = 16  # 每头维度
        num_heads = 4 * tp_size  # Q 头数（随 tp_size 缩放）
        num_kv_heads = 2 * tp_size  # KV 头数（Q 的一半，模拟 GQA）
        batch = 4  # 批大小

        g = torch.Generator(device="cpu").manual_seed(2026)  # 固定种子保证各 rank 数据一致
        x_full = torch.randn(batch, input_size, generator=g)  # 生成完整输入
        w_q = torch.randn(head_size * num_heads, input_size, generator=g)  # 生成完整 Q 权重
        w_k = torch.randn(head_size * num_kv_heads, input_size, generator=g)  # 生成完整 K 权重（KV 头数较少）
        w_v = torch.randn(head_size * num_kv_heads, input_size, generator=g)  # 生成完整 V 权重

        x_full = x_full.to(device)  # 输入移到本 rank 设备
        w_q = w_q.to(device)  # Q 权重移到本 rank 设备
        w_k = w_k.to(device)  # K 权重移到本 rank 设备
        w_v = w_v.to(device)  # V 权重移到本 rank 设备

        # reference: full Q|K|V
        y_ref = torch.cat(  # 参考输出：单卡分别计算 Q/K/V 再拼接
            [
                nn.functional.linear(x_full, w_q, None),  # Q 的单卡计算
                nn.functional.linear(x_full, w_k, None),  # K 的单卡计算
                nn.functional.linear(x_full, w_v, None),  # V 的单卡计算
            ],
            dim=-1,  # 沿输出维拼接
        )

        # TP QKV layer: rank output layout [q_local | k_local | v_local]
        qkv = QKVColumnParallelLinear(  # 构建面向 GQA 的 QKV 合并列并行层
            input_size=input_size,  # 输入维度
            head_size=head_size,  # 每头维度
            num_heads=num_heads,  # 完整 Q 头数
            num_kv_heads=num_kv_heads,  # KV 头数（GQA 分组）
            bias=False,  # 关闭偏置
        ).to(device)  # 移到本 rank 设备

        qkv.weight_loader(qkv.weight, w_q, "q")  # 加载 Q 分片到合并参数的 Q 区间
        qkv.weight_loader(qkv.weight, w_k, "k")  # 加载 K 分片
        qkv.weight_loader(qkv.weight, w_v, "v")  # 加载 V 分片

        y_local = qkv(x_full)  # [batch, head_size*(local_h + 2*local_kv)]  # 本 rank 的输出，布局 [q_local|k_local|v_local]

        # all_gather then re-pack to [Q_all | K_all | V_all]
        y_parts = [torch.empty_like(y_local) for _ in range(tp_size)]  # 为各 rank 输出分配缓冲区
        dist.all_gather(y_parts, y_local)  # 收集所有 rank 的输出分片

        ql = head_size * (num_heads // tp_size)  # 每个 rank 的 Q 分片宽度
        kl = head_size * (num_kv_heads // tp_size)  # 每个 rank 的 K 分片宽度
        vl = head_size * (num_kv_heads // tp_size)  # 每个 rank 的 V 分片宽度

        q_full = torch.cat([p[:, :ql] for p in y_parts], dim=-1)  # 取各 rank 的 Q 部分拼接成完整 Q 输出
        k_full = torch.cat([p[:, ql : ql + kl] for p in y_parts], dim=-1)  # 取 K 部分拼接
        v_full = torch.cat([p[:, ql + kl : ql + kl + vl] for p in y_parts], dim=-1)  # 取 V 部分拼接
        y_full = torch.cat([q_full, k_full, v_full], dim=-1)  # 拼成完整输出

        max_err = (y_full - y_ref).abs().max().item()  # 计算最大绝对误差
        ok = torch.allclose(y_full, y_ref, rtol=1e-4, atol=1e-4)  # 判断结果是否一致
        if tp_rank == 0:  # 只在 rank 0 打印
            print(f"[QKVColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")  # 输出测试结果


    # RowParallelLinear test
    @torch.no_grad()  # 测试不需要梯度
    def test_row_parallel(device):  # 行并行与单卡参考的一致性测试
        tp_rank = dist.get_rank()  # 当前 rank
        tp_size = dist.get_world_size()  # 并行 GPU 数

        in_features = 128 * tp_size  # 输入维度随 tp_size 缩放保证可整除
        out_features = 256  # 输出维度（不分片）
        batch = 4  # 批大小

        g = torch.Generator(device="cpu").manual_seed(2026)  # 固定种子保证各 rank 数据一致
        x_full = torch.randn(batch, in_features, generator=g)  # 生成完整输入
        w_full = torch.randn(out_features, in_features, generator=g)  # 生成完整权重
        b_full = torch.randn(out_features, generator=g)  # 生成完整偏置

        x_full = x_full.to(device)  # 输入移到本 rank 设备
        w_full = w_full.to(device)  # 权重移到本 rank 设备
        b_full = b_full.to(device)  # 偏置移到本 rank 设备

        # reference
        single = ReplicatedLinear(in_features, out_features, bias=True).to(device)  # 构建不分片的参考层
        single.weight.weight_loader(single.weight, w_full)  # 整份加载参考权重
        single.bias.weight_loader(single.bias, b_full)  # 整份加载参考偏置
        y_ref = single(x_full)  # 计算单卡参考输出

        # RowParallel
        row_tp = RowParallelLinear(in_features, out_features, bias=True).to(device)  # 构建行并行层
        row_tp.weight.weight_loader(row_tp.weight, w_full)  # 从完整权重按输入维切出本 rank 分片

        if row_tp.bias is not None:  # 行并行的 bias 不分片，需手动处理
            row_tp.bias.data.copy_(b_full / tp_size)  # 预先除以 tp_size，使 all_reduce 求和后偏置恰好正确

        shard = in_features // tp_size  # 每个 rank 的输入分片宽度
        start = tp_rank * shard  # 本 rank 输入分片的起始位置
        x_part = x_full.narrow(-1, start, shard)  # 按行并行的约定把输入切成本 rank 的分片

        y_row = row_tp(x_part)  # 行并行前向：内部部分和 + all_reduce 得到完整输出

        max_err = (y_row - y_ref).abs().max().item()  # 计算最大绝对误差
        ok = torch.allclose(y_row, y_ref, rtol=1e-4, atol=1e-4)  # 判断结果是否一致
        if tp_rank == 0:  # 只在 rank 0 打印
            print(f"[RowParallel] allclose={ok}, max_abs_err={max_err:.6f}")  # 输出测试结果

    rank, world_size, local_rank, device = _init_dist()  # 初始化分布式环境
    if rank == 0:  # 只在 rank 0 打印
        print(f"Running TP tests with world_size={world_size} on device={device}")  # 打印测试环境信息

    # The test output 'allclose=True' means passed.
    test_column_parallel(device)  # 运行列并行测试
    test_merged_column_parallel(device)  # 运行合并列并行测试
    test_qkv_column_parallel(device)  # 运行 GQA QKV 列并行测试
    test_row_parallel(device)  # 运行行并行测试

    dist.barrier()  # 同步所有进程，确保测试全部完成
    dist.destroy_process_group()  # 销毁进程组，释放分布式资源
