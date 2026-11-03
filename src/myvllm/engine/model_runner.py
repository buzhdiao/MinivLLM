import math  # 导入 math，用于向上取整计算最大块数
import torch  # 导入 PyTorch，张量与 CUDA 操作
import pickle  # 导入 pickle，序列化跨进程（经共享内存）传输的方法调用
import torch.distributed as dist  # 导入分布式模块，多卡 NCCL 通信
from pathlib import Path  # 导入 Path，从模型路径提取模型名
from multiprocessing.synchronize import Event  # 导入进程事件类型，仅用于类型注解
from multiprocessing.shared_memory import SharedMemory  # 导入共享内存，rank 0 与 worker 间传递调用

from myvllm.models.qwen3 import Qwen3ForCausalLM  # 导入 Qwen3 模型实现
from myvllm.models.llama import LlamaForCausalLM  # 导入 Llama 模型实现
from myvllm.layers.sampler import SamplerLayer  # 导入采样层，按 temperature 采样 token
from myvllm.engine.sequence import Sequence  # 导入序列类，批次的基本单元
from myvllm.utils import *  # 导入工具集（含 attention 上下文 set_context/get_context/reset_context）


# ModelRunner：模型执行层。负责加载模型与权重、预热、分配 PagedAttention 的
# KV cache 块池、捕获 CUDA graph，以及把一批 Sequence 拼装成 prefill/decode
# 前向所需的输入（input_ids、slot_mapping、block_tables 等）并执行
class ModelRunner:
    # 初始化：设置分布式环境、按模型名构建模型、加载权重、预热并分配 KV cache
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config  # 保存全局配置
        self.event = event  # 单个事件（worker 用）或事件列表（rank 0 用），同步共享内存读写

        # set distributed config  # 读取分布式相关配置
        self.block_size = config['block_size']  # KV cache 每块容纳的 token 数
        self.world_size = config['world_size']  # 总卡数（rank 数）
        self.enforce_eager = config.get('enforce_eager', False)  # 为 True 时不捕获 CUDA graph，始终 eager 执行

        self.rank = rank  # 本进程的 rank 编号
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)  # 初始化 NCCL 进程组（集合通信屏障：所有 rank 到齐才继续）
        torch.cuda.set_device(rank)  # 把当前 CUDA 设备绑定到本 rank 对应的 GPU

        # set model  # 按模型路径末段名称选择模型实现
        path_str = self.config['model_name_or_path']  # 模型路径
        model_name = Path(path_str).name  # 取路径最后一段作为模型名
        match model_name:  # 按模型名分发构建
            case 'Qwen3-0.6B':  # Qwen3：按其结构参数构建
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],  # 词表大小
                    hidden_size=config['hidden_size'],  # 隐藏层维度
                    num_heads=config['num_heads'],  # 注意力 query 头数
                    head_dim=config['head_dim'],  # 每头维度
                    scale=config['scale'],  # 注意力缩放因子
                    num_kv_heads=config['num_kv_heads'],  # KV 头数（GQA）
                    rms_norm_epsilon=config['rms_norm_epsilon'],  # RMSNorm 的 epsilon
                    qkv_bias=config['qkv_bias'],  # QKV 投影是否带偏置
                    base=config['base'],  # RoPE 的 base
                    max_position=config['max_position'],  # 最大位置编码长度
                    intermediate_size=config['intermediate_size'],  # FFN 中间维度
                    ffn_bias=config['ffn_bias'],  # FFN 是否带偏置
                    num_layers=config['num_layers'],  # Transformer 层数
                    tie_word_embeddings=config['tie_word_embeddings'],  # 是否绑定输入输出 embedding
                    block_size=self.block_size,  # KV cache 块大小（传给注意力层）
                )
            case 'Llama-3.2-1B-Instruct':  # Llama 3.2：按其结构参数构建
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],  # 词表大小
                    hidden_size=config['hidden_size'],  # 隐藏层维度
                    head_dim=config['head_dim'],  # 每头维度
                    num_qo_heads=config['num_qo_heads'],  # query 头数
                    num_kv_heads=config['num_kv_heads'],  # KV 头数（GQA）
                    has_attn_bias=config['has_attn_bias'],  # 注意力是否带偏置
                    rms_norm_epsilon=config['rms_norm_epsilon'],  # RMSNorm 的 epsilon
                    rope_base=config['rope_base'],  # RoPE 的 base
                    max_position_embeddings=config['max_position_embeddings'],  # 最大位置编码长度
                    intermediate_size=config['intermediate_size'],  # FFN 中间维度
                    ffn_bias=config['ffn_bias'],  # FFN 是否带偏置
                    num_layers=config['num_layers'],  # Transformer 层数
                    block_size=self.block_size,  # KV cache 块大小
                    tie_word_embeddings=config['tie_word_embeddings'],  # 是否绑定 embedding
                )
            case _:  # 未知模型名：报错
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Load weights in GPU (model moved to GPU before loading weights)  # 把模型搬到本 rank 的 GPU（先搬再加载权重）
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided  # 若给了模型路径则加载预训练权重
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint  # 延迟导入权重加载器
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])  # 从 checkpoint 读取并写入模型参数

        # Load weights in CPU (move the model to GPU after loading weights)  # 备选方案：先在 CPU 加载权重再搬 GPU（此处未启用）
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()  # 创建采样层（rank 0 用于按 temperature 采出下一个 token）

        # Store default dtype before it's needed in allocate_kv_cache  # 先保存默认 dtype，分配 KV cache 时要用它算块的字节数
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step  # 首个 decode 步的调试标志
        self._first_decode = False

        # warm up model so that we know peak memory usage  # 空跑一次最大批次预热：既触发 kernel 编译，也测出峰值显存
        self.warmup_model()
        # allocate kv cache  # 依据剩余显存分配 KV cache 块池
        self.allocate_kv_cache()
        # capture cuda graph for decoding  # 为 decode 捕获各批大小的 CUDA graph（enforce_eager 时跳过）
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device(f'cuda:{rank}')  # 之后新建张量默认落在本 rank 的 GPU 上
        torch.set_default_dtype(self.default_dtype)  # 恢复默认 dtype

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop  # 共享内存与屏障必须放在所有模型初始化之后：保证两个 rank 都完成预热/分配后，rank 1 才进入事件循环
        if self.world_size > 1:  # 多卡时才需要共享内存通道
            # Synchronize before setting up shared memory  # 先同步一次，确保双方都初始化完毕
            dist.barrier()
            if self.rank == 0:  # rank 0 负责创建共享内存
                # Try to clean up existing shared memory first  # 先清理可能残留的同名共享内存
                try:
                    old_shm = SharedMemory(name='myvllm')  # 尝试打开旧共享内存
                    old_shm.close()  # 关闭映射
                    old_shm.unlink()  # 删除旧段
                except FileNotFoundError:  # 不存在则无需清理
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)  # 创建 1MB 共享内存段作为调用通道
                # Barrier to ensure rank 1 waits until shared memory is created  # 再次同步：保证 rank 1 打开时共享内存已创建
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory  # worker：等 rank 0 创建好共享内存再打开
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')  # 打开 rank 0 创建的共享内存段
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__  # 这里不能进入 loop()，由外层 worker_process 在 __init__ 之后调用，否则会卡死在初始化里

    # only use read when rank != 0  # worker 端：从共享内存读取 rank 0 下发的方法调用（阻塞直到事件通知）
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"  # 仅 worker 可读
        self.event.wait()  # 阻塞等待 rank 0 写入完成的事件通知
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length  # 前 4 字节是本条消息的长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化出方法名与参数
        self.event.clear()  # 清除事件，等待下一次通知
        return method_name, args

    # only use write when rank == 0  # rank 0 端：把方法调用序列化后写入共享内存并通知所有 worker
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"  # 仅 rank 0 可写
        # encode the length first  # 先写长度再写内容
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)  # 把 (方法名, 参数元组) 摊平成 (方法名, *参数) 再序列化
        data = pickle.dumps((method_name, *args))
        n = len(data)  # 序列化后的字节数
        self.shm.buf[:4] = n.to_bytes(4, 'little')  # 前 4 字节写入长度
        self.shm.buf[4:n+4] = data  # 紧接着写入消息内容
        for event in self.event:  # 逐个 set 事件，通知每个 worker 有新调用
            event.set()

    # close shared memory, destroy process group, delete graphs  # 退出清理：关闭共享内存、销毁进程组、删除 CUDA graph
    def exit(self):
        if self.world_size > 1:  # 多卡时才存在共享内存
            self.shm.close()  # 关闭本进程的映射
            if self.rank == 0:  # 由 rank 0 负责删除共享内存段
                self.shm.unlink()
        if not self.enforce_eager:  # 捕获过 CUDA graph 才需要释放
            del self.graphs  # 删除所有已捕获的 graph
            del self.graph_vars  # 删除 graph 的输入/输出缓冲张量
        torch.cuda.synchronize()  # 等待所有 CUDA 工作完成
        # Check if process group exists before destroying  # 进程组存在才销毁
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory  # worker 主循环：阻塞等待并读取方法与参数
    # execute the method with args  # 执行对应方法
    # write results back to shared memory  # 结果通过方法返回值所在对象自然共享（张量在各 rank GPU 上同步）
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"  # 仅 worker 进入
        while True:  # 持续服务直到收到 exit
            method_name, args = self.read_shm()  # 读取一次调用
            self.call(method_name, *args) # Unpack args when calling  # 解包参数并执行
            if method_name == 'exit':  # 收到退出指令
                self.exit()  # 清理资源
                break  # 退出循环，进程结束

    # will be called by both rank == 0 and rank != 0  # 统一调用入口：rank 0 与 worker 都用它
    # given method name and args from shared memory  # 多卡时 rank 0 先把调用广播给 worker
    # execute the method and return results  # 然后各自在本 rank 执行同名方法
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine  # rank 0 在主引擎中调用：先把调用写入共享内存
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)  # 按名字取出本对象的方法
        if method:  # 存在则执行
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")  # 方法名不存在：报错

    # cleanup memory  # 预热流程说明：
    # compute max number of sequence based on max token and max model length  # 用 max_num_batched_tokens / max_model_length 估算最大并发批大小
    # run empty sequence to warm up the model  # 用全 0 token 的假序列空跑一次 prefill，触发 kernel 编译并测峰值显存
    # clear memory  # 结束后清缓存，防止碎片影响后续分配
    def warmup_model(self):
        torch.cuda.empty_cache()  # 先清空缓存
        torch.cuda.reset_peak_memory_stats()  # 重置峰值显存统计
        max_tokens = self.config['max_num_batch_tokens']  # 每步最大 token 数
        max_model_length = self.config['max_model_length']  # 单序列最大长度
        batch_size = max_tokens // max_model_length  # 最坏情况下（每条都占满长度）的最大并发序列数
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]  # 构造 batch_size 条满长度的假序列
        self.run(seqs, is_prefill=True)  # 以 prefill 方式空跑一遍
        torch.cuda.empty_cache()  # 清理预热产生的缓存

    # allocate kv cache memory blocks for model  # 分配 PagedAttention 的 KV cache 块池：一次分配一整块大张量，再按需切块给各序列
    def allocate_kv_cache(self):
        # find all available memory  # 统计可用显存
        free_mem, total_mem = torch.cuda.mem_get_info()  # 查询当前 GPU 空闲显存
        total_free_mem = free_mem * self.config['gpu_memory_utilization']  # 按 gpu_memory_utilization 比例打折，给其他开销留余量
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']  # 预热测得的峰值显存占用
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']  # 当前显存占用
        # reserve some room for peak memory usage during model execution  # 扣除"峰值-当前"的差值，为执行期临时显存预留空间
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # find parameters to compute kv cache size  # 取出计算每块字节数所需的参数
        num_layers = self.config['num_layers']  # Transformer 层数（每层独立 KV）
        num_kv_heads = self.config['num_kv_heads'] // self.world_size  # KV 头数按卡数切分（tensor parallel）
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']  # 每头维度（缺省时由 hidden_size/头数推得）

        # check whether the current free memory can hold at least one block  # 校验至少能放下一块
        # compute the actual byte required of each block  # 每块字节数 = 块内 token 数 × K/V 两份 × 层数 × KV 头数 × 每头维度 × dtype 大小
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        num_available_kv_blocks = int(available_mem // block_bytes)  # 可分配的块总数
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'  # 至少 1 块，否则无法运行
        
        # Synchronize max_cached_blocks across all ranks.  # 跨 rank 同步 max_cached_blocks：
        # Each rank independently computed num_available_kv_blocks from its own  # 每个 rank 各自基于自己的空闲显存算出块数，
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead  # 各 rank 可能略有差异：rank 0 额外背负 NCCL 缓冲等开销，空闲往往更少。
        # (NCCL buffers, process-group state) so it often has less free memory than  # 若不同步，调度器（只在 rank 0）会按本地值分配，
        # workers. Without sync, the scheduler (which runs only on rank-0) would use  # 可能超出某个 rank 的承载，导致该 rank 写 KV 时 OOM。
        # rank-0's local value and could allocate more blocks than some rank can hold,  # 
        # causing an OOM on that rank during KV cache writes.  # 
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")  # 打印本 rank 本地算得的块数
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )  # 把本地块数放进 GPU 张量以便 all_reduce
            # all_reduce with MIN: every rank learns the most conservative limit,  # 用 MIN 归约：所有 rank 都得到最保守的上限，
            # i.e. the block count that even the most memory-constrained rank can serve.  # 即显存最紧张的 rank 也能承载的块数。
            # This single agreed-upon value is then stored in config so the Scheduler  # 该统一值写回 config，之后的 Scheduler（在 rank 0 初始化）
            # (initialized afterwards on rank-0) never allocates more blocks than any  # 就不会分配超过任何 rank 物理可承载的块数。
            # rank can physically hold.  # 
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()  # 取各 rank 最小值作为全局块数
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.  # 单卡：无需跨卡同步，直接用本地值
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:  # rank 0 打印最终统一值
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # allocate max possible kv cache for the model, instead for each sequence  # 一次性按最大容量分配 KV cache（而非每条序列各自分配）：
        # this is the key for paged attention: one giant KV cache pool, divided into blocks  # 这正是 PagedAttention 的关键：一个巨大的 cache 池，按块切分、按需映射
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values  # 必须用 zeros 而非 empty，避免垃圾值污染未写入位置
        allocated_kv_cache = torch.zeros(2, self.config['num_layers'], self.config['max_cached_blocks'], self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')  # 形状 [2(K/V), 层数, 块数, 块大小, KV 头数, 每头维度]
        layer_id = 0  # 逐层把块池切片挂到注意力模块上
        for module in self.model.modules():  # 遍历所有子模块
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):  # 找到带 KV cache 槽位的注意力层
                module.k_cache = allocated_kv_cache[0, layer_id]  # 该层的 K cache = 池中第 layer_id 层的 K 切片（共享存储，零拷贝）
                module.v_cache = allocated_kv_cache[1, layer_id]  # V cache 同理
                layer_id += 1  # 下一层

    # given seqs  # 为 prefill 前向准备输入数据：
    # prepare the data needed for a prefill forward pass  # 拼接各序列的 input_ids、位置信息与 cu_seqlens（varlen 注意力），
    # taking prefix cache into consideration:  # 并考虑 prefix cache（已缓存部分跳过）：
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)  # slot_mapping 指明新 KV 写入哪些槽位，block_tables 指明从哪些块读历史 KV
    # cu_seqlens_q = [0, 3, 5, 9]  # cu_seqlens_q 示意：各序列 query 的累积边界
    #               │  │  │  │  # 
    #               │  │  │  └─ end of seq3 (position 9)  # 序列 3 结束于位置 9
    #               │  │  └──── end of seq2 (position 5)  # 序列 2 结束于位置 5
    #               │  └─────── end of seq1 (position 3)  # 序列 1 结束于位置 3
    #               └────────── start (position 0)  # 起点 0
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all input_ids after prefix cache  # 输入 token：各序列扣除 prefix cache 命中部分后的拼接
        input_ids = []
        # length: sum of all input_ids after prefix cache  # 槽位映射：与新 token 一一对应的 KV 写入位置（物理槽位号）
        slot_mappings = []
        # length: num_seqs  # 每条序列本次需计算的 query 长度
        seqlens_q = []
        # length: num_seqs  # 每条序列可读取的全部上下文长度（含 prefix cache）
        seqlens_k = []
        # length: num_seqs + 1  # query 累积长度前缀和（varlen flash attention 的分段边界）
        cu_seqlens_q = [0]
        # length: num_seqs + 1  # key 累积长度前缀和
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)  # 各序列的块表（填充到同长）
        block_tables = []
        for seq in seqs:  # 逐序列收集元数据
            token_ids = seq.token_ids  # 该序列全部 token
            num_cached_tokens = seq.num_cached_tokens  # prefix cache 命中的 token 数（本次无需重算）
            input_ids.extend(token_ids[num_cached_tokens:])  # 只拼接未命中部分作为本次输入
            seqlens_q.append(len(token_ids) - num_cached_tokens)  # query 数 = 未命中 token 数
            seqlens_k.append(len(token_ids))  # key 数 = 全部长度（KV cache 中已有历史部分）
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])  # 累积 query 边界
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])  # 累积 key 边界
            if seq.block_table:  # 有历史块时需要为新 token 计算写入槽位
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):  # 遍历未命中前缀的块
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:  # 非末块：整块都会写入，槽位为整块区间
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:  # 末块（未满）：只写到最后一个 token 的槽位
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:  # 存在 prefix cache（query 少于 key）时才需要块表读历史 KV
            # pad block_tables  # 把各序列块表填充到统一长度
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)  # 最长块表长度
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))  # 短的用 -1 填充
                block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)  # 输入 token 上卡（锁页内存 + 异步传输）
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)  # 槽位映射上卡

        set_context(  # 把本轮注意力上下文写入全局（供 attention 层读取）
            is_prefill=True,  # 标记 prefill 模式
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),  # query 累积边界
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),  # key 累积边界
            max_seqlen_q=max(seqlens_q),  # 批内最长 query（varlen kernel 需要）
            max_seqlen_k=max(seqlens_k),  # 批内最长 key
            slot_mapping=slot_mapping_tensor,  # KV 写入槽位
            context_lens=None,  # prefill 不需要逐序列上下文长度
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,  # 读历史 KV 的块表（无 prefix cache 时为 None）
        )
        return input_ids  # 返回拼好的输入 token 张量


    # prepare input data for decoding  # 为 decode 前向准备输入：每序列只有刚采出的一个 token
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []  # 每序列的最后一个 token
        context_lens = []    # 每序列已处理的上下文长度（决定读多少历史 KV）
        slot_mappings = []   # 新 token 的 KV 写入槽位
        block_tables = []    # 每序列的块表（读历史 KV 的块号列表）
        for seq in seqs:  # 逐序列收集
            input_ids.append(seq.last_token)  # 输入是刚采出的最后一个 token
            context_lens.append(len(seq))  # 上下文长度 = 当前 token 总数
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)  # 写入槽位 = 末块起始槽位 + 末块已有 token 数 - 1
        all_block_tables = [seq.block_table for seq in seqs]  # 收集全部块表
        max_num_blocks = max(len(bt) for bt in all_block_tables)  # 最长块表长度
        for i, seq in enumerate(seqs):  # 填充到统一长度
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))  # 用 -1 补齐
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)  # 输入 token 上卡
        set_context(  # 写入 decode 模式的注意力上下文
            is_prefill=False,  # 标记 decode 模式
            cu_seqlens_q=None,  # decode 每序列只有 1 个 query，不需要 varlen 边界
            cu_seqlens_k=None,  # 同上
            max_seqlen_q=0,  # 不适用
            max_seqlen_k=0,  # 不适用
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),  # 新 KV 的写入槽位
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),  # 各序列上下文长度
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,  # 块表（paged attention 读历史 KV）
        )
        return input_ids    

    # prepare the temperature  # 准备采样用的 temperature 张量（上卡）
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits  # 执行模型前向：
    # when decoding, use cuda graph execution to speed up  # prefill 走 eager（varlen 输入形状不定）；decode 重放 CUDA graph 提速
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs  # graph 变量里预分配了 input_ids/slot_mapping/context_lens/block_tables/outputs
    # into graph_variable, and then replay the graph  # 把本步输入拷进 graph 变量后重放
    @torch.inference_mode()  # 推理模式：禁用 autograd 追踪
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:  # prefill 或强制 eager：普通前向
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)  # varlen prefill 的 input_ids 保持 1D（各序列拼接）：
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens  # 不要升维，flash_attn_varlen_func 要求 1D 输入配 cu_seqlens
            hidden_states = self.model(input_ids)  # 前向得到各位置的隐藏状态
            logits = self.model.compute_logits(hidden_states)  # 只取最后一个 token 位置算 logits
        else:  # decode：重放 CUDA graph
            bs = input_ids.size(0)  # 本步批大小
            context = get_context()  # 读取 prepare_decode 写入的注意力上下文

            # finds smallest captured graph that fits the batch size  # 找到能容纳本批大小的最小已捕获 graph
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars  # graph 的预分配输入/输出缓冲
            # copy input data into graph variables  # 把本步输入拷进 graph 变量
            vars['input_ids'][:bs].copy_(input_ids)  # 输入 token
            vars['slot_mapping'][:bs].fill_(-1)  # 槽位先填 -1（多出的批位置不写 KV）
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)  # 再拷入真实槽位
            vars["context_lens"].zero_()  # 上下文长度清零（未用到的批位置为 0）
            vars['context_lens'][:bs].copy_(context.context_lens)  # 拷入真实上下文长度
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables  # 拷入块表（宽度按本步实际块数）
            # replay the graph  # 重放捕获的 kernel 序列
            graph.replay()
            logits = self.model.compute_logits(vars['outputs'][:bs])  # 用 graph 输出计算 logits

        return logits


    # prepare prefill  # 引擎单步的执行入口：
    # prepare sample  # 准备 prefill/decode 输入与采样参数
    # run model  # 执行前向
    # sample logits  # 采样出下一个 token
    # reset context  # 清理注意力上下文
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:  # 按批次类型准备输入
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)  # 前向得到 logits
        # only sample when rank == 0  # 只有 rank 0 负责采样（结果随后同步到各 worker）
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))  # 按 temperature 采样每个序列的下一个 token
        reset_context()  # 清理全局注意力上下文
        return token_ids

    # capture the CUDA graph:  # 捕获 decode 用的 CUDA graph：
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs  # 缓冲按最大尺寸一次性预分配，所有 graph 复用
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))  # 对常用批大小档位各捕获一份
    # with torch.cuda.graph(graph, self.graph_pool):  # 在 graph 捕获上下文中：
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured  # 跑一遍 model()，其 CUDA kernel 序列即被录制成 graph
    # (later use graph.replay() to run the captured graph)  # 之后用 graph.replay() 重放
    @torch.inference_mode()  # 推理模式捕获
    def capture_cudagraph(self) -> None:
        max_bs = self.config['max_num_seqs']  # 最大批大小
        max_len = self.config['max_model_length']  # 单序列最大长度
        max_num_blocks = math.ceil(max_len / self.block_size)  # 单序列最多占用的块数
        # for decoding, input is always of shape (batch_size, 1)  # decode 输入张量（每序列 1 个 token）
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention  # 为 paged attention 准备的缓冲：
        # where to write new KV values in the cache  # 新 KV 写入的槽位
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed  # 各序列已处理的 token 数
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # where to read KV values in the cache  # 读历史 KV 的块表
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # output logits  # 输出缓冲（各序列最后一位置的隐藏状态对应的 logits 空间）
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=f'cuda:{self.rank}')

        # graphs to be captured for different batch sizes  # 要捕获的批大小档位
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}  # 批大小 -> graph 的映射
        graph_pool = None  # 各 graph 共享的内存池（首次捕获后建立）

        for batch_size in reversed(batch_sizes):  # 从最大批开始捕获：先建好最大的内存池供后续共享
            graph = torch.cuda.CUDAGraph()  # 新建一个 CUDA graph
            set_context(
                is_prefill=False,  # decode 模式
                cu_seqlens_q=None,  # decode 不需要 varlen 边界
                cu_seqlens_k=None,  # 同上
                max_seqlen_q=0,  # 不适用
                max_seqlen_k=0,  # 不适用
                slot_mapping=slot_mapping[:batch_size],  # 本档位的槽位切片
                context_lens=context_lens[:batch_size],  # 本档位的上下文长度切片
                block_tables=block_tables[:batch_size],  # 本档位的块表切片
            )  # 捕获时固化 graph 引用的上下文张量
            outputs[:batch_size] = self.model(input_ids[:batch_size])  # 捕获前先热身跑一遍（分配捕获期间的临时内存）

            with torch.cuda.graph(graph, graph_pool):  # 进入捕获模式：期间所有 CUDA kernel 被录制
                outputs[:batch_size] = self.model(input_ids[:batch_size])  # 录制本档位批大小的 decode 前向
                if graph_pool is None:  # 第一个（最大的）graph 建立内存池
                    graph_pool = graph.pool()
            # store the captured graph  # 登记本档位的 graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture  # 同步确保捕获完成后再清理并捕获下一档
            torch.cuda.synchronize()
            reset_context()  # 清理本次捕获写入的注意力上下文

        self.graph_vars = dict(  # 保存预分配的 graph 输入/输出缓冲，供重放时拷入数据
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
