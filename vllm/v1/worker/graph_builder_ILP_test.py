import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple, Dict, Set
import json
import logging
import pprint

# ILP 相关
try:
    from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, PULP_CBC_CMD
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    logging.warning("PuLP not found. ILP optimization will not be available.")

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# PART 1: YOUR CUSTOM GRAPH FRAMEWORK (支持内存别名)
# ==============================================================================

class TensorSpec:
    """一个简单的数据类，用于存储Tensor的元数据。"""
    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype
        
        if dtype.is_floating_point:
            bytes_per_element = torch.finfo(dtype).bits // 8
        else:
            bytes_per_element = torch.iinfo(dtype).bits // 8
        self.size_bytes = torch.prod(torch.tensor(shape)).item() * bytes_per_element

    def __repr__(self) -> str:
        return f"TensorSpec(shape={list(self.shape)}, dtype={self.dtype}, size_bytes={self.size_bytes})"

class OpNode:
    """代表一个操作节点。"""
    def __init__(self, op_type: str, output_name: str, output_spec: TensorSpec, input_names: List[str]):
        self.op_type = op_type
        self.output_name = output_name
        self.output_spec = output_spec
        self.input_names = input_names
        
    def __repr__(self) -> str:
        return f"{self.output_name} = {self.op_type}({', '.join(self.input_names)})  |  {self.output_spec}"

class CustomGraph:
    """我们自己的、极简的计算图，支持别名。"""
    def __init__(self):
        self.nodes: Dict[str, OpNode] = {}
        self.execution_order: List[OpNode] = []
        # 用于存储别名关系 {source: target}
        self.aliases: Dict[str, str] = {}

    def add_op(self, output_name: str, op_type: str, input_names: List[str], shape: Tuple[int, ...], dtype: torch.dtype):
        if output_name in self.nodes:
            raise ValueError(f"Tensor with name '{output_name}' already exists in the graph.")
        spec = TensorSpec(shape, dtype)
        node = OpNode(op_type, output_name, spec, input_names)
        self.nodes[output_name] = node
        self.execution_order.append(node)
        return node
    
    def add_alias(self, source_name: str, target_name: str):
        """定义内存别名，强制 source 复用 target 的内存。"""
        if source_name not in self.nodes or target_name not in self.nodes:
            raise ValueError("Both source and target tensors for alias must exist in the graph.")
        
        source_spec = self.nodes[source_name].output_spec
        target_spec = self.nodes[target_name].output_spec
        
        if source_spec.size_bytes > target_spec.size_bytes:
            raise ValueError(
                f"Alias failed: source '{source_name}' ({source_spec.size_bytes} bytes) "
                f"is larger than target '{target_name}' ({target_spec.size_bytes} bytes)."
            )
        logging.info(f"Memory alias added: '{source_name}' will reuse memory of '{target_name}'.")
        self.aliases[source_name] = target_name

    def print_graph(self):
        print("--- Custom Computation Graph ---")
        for node in self.execution_order:
            print(node)
        
        if self.aliases:
            print("\n--- Aliases (for In-Place Optimization) ---")
            for source, target in self.aliases.items():
                print(f"  '{source}' -> reuses memory of -> '{target}'")
        print("---------------------------------------------")

# ==============================================================================
# PART 2: THE ADAPTED MEMORY ANALYZER (最终版)
# ==============================================================================

class MemoryAnalyzer:
    """接收 CustomGraph 对象，生成包含别名信息的最终内存计划。"""
    def __init__(self, graph: CustomGraph):
        self.graph = graph
        self.node_indices = {node.output_name: i for i, node in enumerate(self.graph.execution_order)}

    def analyze(self) -> Dict[str, Any]:
        logging.info("Starting memory analysis on CustomGraph...")
        
        liveness_map = self._analyze_liveness()
        peak_info = self._calculate_peak_memory(liveness_map)
        memory_timeline = self._calculate_memory_timeline(liveness_map, peak_info["peak_memory_bytes"])
        reuse_plan = self._generate_optimized_reuse_plan()
        
        logging.info("Analysis complete.")

        return {
            "peak_memory_mb": peak_info["peak_memory_bytes"] / (1024**2),
            "peak_node_name": peak_info["peak_node"].output_name,
            "peak_live_tensors": {
                name: f"{self.graph.nodes[name].output_spec.size_bytes / (1024**2):.4f} MB"
                for name in peak_info["peak_live_set_names"]
            },
            "memory_timeline": memory_timeline,
            "reuse_plan": reuse_plan
        }
        
    def _analyze_liveness(self) -> Dict[str, Set[str]]:
        liveness_map: Dict[str, Set[str]] = {}
        live_tensor_names: Set[str] = set()
        
        for node in reversed(self.graph.execution_order):
            liveness_map[node.output_name] = live_tensor_names.copy()
            if node.output_name in live_tensor_names:
                live_tensor_names.remove(node.output_name)
            for input_name in node.input_names:
                live_tensor_names.add(input_name)
        return liveness_map

    def _calculate_peak_memory(self, liveness_map: Dict[str, Set[str]]) -> Dict:
        peak_memory_bytes, peak_node, peak_live_set_names = 0, None, set()
        
        for node in self.graph.execution_order:
            current_node_size = node.output_spec.size_bytes
            live_after_node_names = liveness_map.get(node.output_name, set())
            live_set_for_peak_calc = live_after_node_names.union({node.output_name})
            
            current_total_memory = sum(self.graph.nodes[name].output_spec.size_bytes for name in live_set_for_peak_calc)
            
            if current_total_memory > peak_memory_bytes:
                peak_memory_bytes = current_total_memory
                peak_node = node
                peak_live_set_names = live_set_for_peak_calc
                
        return {"peak_memory_bytes": peak_memory_bytes, "peak_node": peak_node, "peak_live_set_names": peak_live_set_names}
    
    def _calculate_memory_timeline(self, liveness_map: Dict[str, Set[str]], peak_memory_bytes: int) -> List[Dict[str, Any]]:
        """计算每个时刻的显存占用和占有率"""
        timeline = []
        
        for i, node in enumerate(self.graph.execution_order):
            # 计算当前时刻存活的所有张量
            live_after_node_names = liveness_map.get(node.output_name, set())
            live_set_for_calc = live_after_node_names.union({node.output_name})
            
            # 计算当前时刻的总内存使用量
            current_total_memory = sum(self.graph.nodes[name].output_spec.size_bytes for name in live_set_for_calc)
            current_total_memory_mb = current_total_memory / (1024**2)
            
            # 计算占有率 (当前显存占用 / 峰值显存)
            occupancy_ratio = current_total_memory / peak_memory_bytes if peak_memory_bytes > 0 else 0.0
            
            # 计算存活张量数量
            live_tensor_count = len(live_set_for_calc)
            
            # 获取存活张量的详细信息
            live_tensors_info = {
                name: {
                    "size_mb": self.graph.nodes[name].output_spec.size_bytes / (1024**2),
                    "shape": list(self.graph.nodes[name].output_spec.shape),
                    "dtype": str(self.graph.nodes[name].output_spec.dtype)
                }
                for name in live_set_for_calc
            }
            
            timeline.append({
                "timestep": i,
                "node_name": node.output_name,
                "op_type": node.op_type,
                "total_memory_mb": current_total_memory_mb,
                "occupancy_ratio": occupancy_ratio,
                "occupancy_percentage": occupancy_ratio * 100,
                "live_tensor_count": live_tensor_count,
                "live_tensors": live_tensors_info
            })
        
        return timeline
    
    def _generate_optimized_reuse_plan(self) -> Dict[str, Dict[str, Any]]:
        """使用 ILP 优化内存分配"""
        if not HAS_PULP:
            raise ImportError("PuLP library is required. Install with: pip install pulp")
        
        # 1. 计算生命周期
        births = {node.output_name: idx for idx, node in enumerate(self.graph.execution_order)}
        deaths = {name: -1 for name in self.graph.nodes}
        for idx, node in enumerate(self.graph.execution_order):
            for input_name in node.input_names:
                if input_name in deaths:
                    deaths[input_name] = idx
        
        # 2. 收集所有非零大小的 tensor
        tensors = [(node.output_name, node.output_spec.size_bytes) 
                   for node in self.graph.execution_order 
                   if node.output_spec.size_bytes > 0]
        
        if not tensors:
            return {'_total_buffer_size_bytes': 0}
        
        tensor_names = [name for name, _ in tensors]
        tensor_sizes = {name: size for name, size in tensors}
        
        # 3. 找出所有生命周期冲突的 tensor 对
        conflicts = []
        for i, name_i in enumerate(tensor_names):
            birth_i = births[name_i]
            death_i = deaths.get(name_i, len(self.graph.execution_order))
            
            for j in range(i + 1, len(tensor_names)):
                name_j = tensor_names[j]
                birth_j = births[name_j]
                death_j = deaths.get(name_j, len(self.graph.execution_order))
                
                # 生命周期重叠判断
                if not (death_i < birth_j or death_j < birth_i):
                    conflicts.append((name_i, name_j))
        
        logging.info(f"Using ILP optimization for {len(tensor_names)} tensors...")
        logging.info(f"Building ILP model: {len(tensor_names)} tensors, {len(conflicts)} conflicts")
        
        # 4. 创建 ILP 问题
        prob = LpProblem("MemoryAllocation", LpMinimize)
        
        # 变量: 每个 tensor 的起始 offset
        # 使用 Continuous 而不是 Integer 以避免 Big-M 溢出问题
        offsets = {name: LpVariable(f"offset_{name}", lowBound=0, cat='Continuous') 
                   for name in tensor_names}
        
        # 变量: 总内存大小
        total_memory = LpVariable("total_memory", lowBound=0, cat='Continuous')
        
        # 目标函数: 最小化总内存
        prob += total_memory, "Minimize_Total_Memory"
        
        # 约束1: 每个 tensor 的结束位置不超过总内存
        for name in tensor_names:
            prob += offsets[name] + tensor_sizes[name] <= total_memory, f"bound_{name}"
        
        # 约束2: 处理别名约束（强制复用）
        for source_name, target_name in self.graph.aliases.items():
            if source_name in offsets and target_name in offsets:
                prob += offsets[source_name] == offsets[target_name], f"alias_{source_name}_to_{target_name}"
                logging.info(f"ILP: Adding alias constraint {source_name} == {target_name}")
        
        # 约束3: 冲突 tensor 不能内存重叠（使用 Big-M 方法）
        big_m = sum(tensor_sizes.values())
        
        for idx, (name_i, name_j) in enumerate(conflicts):
            # 二元变量: b=1 表示 i 在 j 之前，b=0 表示 j 在 i 之前
            b = LpVariable(f"b_{idx}", cat='Binary')
            
            # 如果 b=1: offset_i + size_i <= offset_j
            prob += offsets[name_i] + tensor_sizes[name_i] <= offsets[name_j] + big_m * (1 - b), \
                    f"conflict_{idx}_a"
            # 如果 b=0: offset_j + size_j <= offset_i
            prob += offsets[name_j] + tensor_sizes[name_j] <= offsets[name_i] + big_m * b, \
                    f"conflict_{idx}_b"
        
        # 5. 求解
        logging.info("Solving ILP problem...")
        solver = PULP_CBC_CMD(
            msg=0, 
            timeLimit=60,
            options=[
                'randomSeed 42',
                'randomCbcSeed 42',
                'perturbation off',
                'threads 1'
            ]
        )
        prob.solve(solver)
        
        # 6. 检查求解状态
        status = LpStatus[prob.status]
        if status != 'Optimal':
            logging.warning(f"ILP solver status: {status}")
        
        if status == 'Infeasible':
            raise RuntimeError("ILP problem is infeasible")
        
        if status not in ['Optimal', 'Not Solved']:
            logging.warning(f"ILP did not find optimal solution, status: {status}")
        
        # 7. 提取结果
        reuse_plan: Dict[str, Dict[str, Any]] = {}
        
        for name in tensor_names:
            # Round floating point offset to nearest integer
            offset_value = round(offsets[name].varValue) if offsets[name].varValue is not None else 0
            node = self.graph.nodes[name]
            reuse_plan[name] = {
                'offset': offset_value,
                'size': node.output_spec.size_bytes,
                'shape': node.output_spec.shape,
                'dtype': node.output_spec.dtype
            }
        
        # Round total memory to nearest integer (ceil for safety)
        import math
        total_mem = math.ceil(total_memory.varValue) if total_memory.varValue is not None else 0
        reuse_plan['_total_buffer_size_bytes'] = total_mem
        
        logging.info(f"ILP solution found. Total memory: {total_mem / (1024**2):.2f} MB (status: {status})")
        
        return reuse_plan

# ==============================================================================
# PART 3: THE APPLICATION (使用你自己的计算图)
# ==============================================================================

def define_llada_block_graph(N: int) -> CustomGraph:
    """
    使用您的 CustomGraph，手动、声明式地构建 LLaDALlamaBlock 的计算图。
    基于当前的计算流程创建节点
    """
    logging.info("Starting manual graph construction with CustomGraph...")
    graph = CustomGraph()

    # --- 定义模型配置 ---
    D_MODEL = 4096
    INTERMEDIATE_SIZE = 12288
    NUM_HEAD_Q = 32
    NUM_HEAD_KV = 32
    HEAD_DIM = D_MODEL // NUM_HEAD_Q
    VOCAB_SIZE = 126464
    DTYPE = torch.bfloat16

    # --- 开始构建图，严格按照你的计算顺序 ---
    
    # 输入张量
    graph.add_op('x', 'placeholder', [], shape=(N, D_MODEL), dtype=DTYPE)
    
    # 1. Attention RMS norm
    graph.add_op('x_normed', 'attn_norm', ['x'], shape=(N, D_MODEL), dtype=DTYPE)
    
    # 2. QKV projections
    graph.add_op('q', 'q_proj', ['x_normed'], shape=(N, D_MODEL), dtype=DTYPE)
    graph.add_op('k', 'k_proj', ['x_normed'], shape=(N, NUM_HEAD_KV * HEAD_DIM), dtype=DTYPE)
    graph.add_op('v', 'v_proj', ['x_normed'], shape=(N, NUM_HEAD_KV * HEAD_DIM), dtype=DTYPE)
    
    # 3. Attention operation
    graph.add_op('att_tmp', 'attention_op', ['q', 'k', 'v'], shape=(N, NUM_HEAD_Q * HEAD_DIM), dtype=DTYPE)
    
    # 4. Attention output (after o_proj)
    graph.add_op('att_out', 'o_proj', ['att_tmp'], shape=(N, D_MODEL), dtype=DTYPE)
    
    # 5. Residual connection + MLP RMS norm
    graph.add_op('x_normed_2', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, D_MODEL), dtype=DTYPE)
    graph.add_op('residual', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, D_MODEL), dtype=DTYPE)
    
    # 6. MLP projections
    graph.add_op('mlp_x', 'ff_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    graph.add_op('mlp_x_up', 'up_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    
    # 7. MLP activation and gating
    graph.add_op('mlp_gated', 'mul_op', ['mlp_x', 'mlp_x_up'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    
    # 8. Final projection and residual
    graph.add_op('mlp_output', 'ff_out', ['mlp_gated'], shape=(N, D_MODEL), dtype=DTYPE)
    
    
    #graph.add_op('xnorm2_barrier', 'barrier', ['x_normed_2'], shape=(N, 0), dtype=DTYPE)
    
    graph.add_op('final_output', 'add_residual_2', ['residual', 'mlp_output'], shape=(N, D_MODEL), dtype=DTYPE)
    graph.add_alias(source_name='final_output', target_name='x')

    graph.add_op('logits','get_logits',['final_output'],shape=(N,VOCAB_SIZE),dtype=DTYPE)

    logging.info("Manual graph construction complete.")
    return graph


def define_dream_block_graph(N: int) -> CustomGraph:
    """
    定义 Dream 模型单个 block 的计算图（用于显存复用分析）
    基于 DreamDecoderLayer 的计算流程创建节点
    
    Dream 的计算流程（view/reshape/silu 等原地操作不创建节点）：
    1. x -> input_layernorm -> x_normed
    2. x_normed -> q_proj/k_proj/v_proj -> q/k/v
    3. q/k/v -> RoPE + FlashAttention -> att_out_var
    4. att_out_var -> o_proj -> att_out
    5. x + att_out -> fused_add_rms_norm -> residual + x_normed_2
    6. x_normed_2 -> gate_proj/up_proj -> x_gate/x_up
    7. x_gate * x_up -> x_mlp (SiLU 是原地操作)
    8. x_mlp -> down_proj -> mlp_output
    9. residual + mlp_output -> final_output
    10. final_output -> lm_head -> logits
    """
    logging.info("Starting Dream block graph construction...")
    graph = CustomGraph()
    
    # --- Dream 模型配置（从 config.json） ---
    HIDDEN_SIZE = 3584
    INTERMEDIATE_SIZE = 18944
    NUM_HEADS = 28
    NUM_KV_HEADS = 4
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 128
    VOCAB_SIZE = 152064
    DTYPE = torch.bfloat16
    
    # --- 构建计算图，严格按照 Dream 的计算顺序 ---
    
    # 输入张量
    graph.add_op('x', 'placeholder', [], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 1. Attention 路径：input_layernorm
    graph.add_op('x_normed', 'input_layernorm', ['x'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 2. QKV projections
    graph.add_op('q', 'q_proj', ['x_normed'], shape=(N, NUM_HEADS * HEAD_DIM), dtype=DTYPE)
    graph.add_op('k', 'k_proj', ['x_normed'], shape=(N, NUM_KV_HEADS * HEAD_DIM), dtype=DTYPE)
    graph.add_op('v', 'v_proj', ['x_normed'], shape=(N, NUM_KV_HEADS * HEAD_DIM), dtype=DTYPE)
    
    # 3. RoPE + FlashAttention
    # 注意：q/k/v 的 reshape 是 view 操作（原地），不需要额外节点
    # RoPE 是 in-place 的，attention 直接依赖 q/k/v
    graph.add_op('att_out_var', 'flash_attention', ['q', 'k', 'v'], 
                 shape=(N, NUM_HEADS, HEAD_DIM), dtype=DTYPE)
    
    # 4. O projection
    # 注意：att_out_var 的 reshape 也是 view 操作（原地），不需要额外节点
    graph.add_op('att_out', 'o_proj', ['att_out_var'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 5. 第一个残差连接 + MLP RMS norm (fused)
    graph.add_op('x_normed_2', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    graph.add_op('residual', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 6. MLP projections
    graph.add_op('x_gate', 'gate_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    graph.add_op('x_up', 'up_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    
    # 7. Gating (element-wise multiply, SiLU 是原地操作，不需要单独节点)
    graph.add_op('x_mlp', 'mul_gate', ['x_gate', 'x_up'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    
    # 8. Down projection
    graph.add_op('mlp_output', 'down_proj', ['x_mlp'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 9. 第二个残差连接
    graph.add_op('final_output', 'add_residual_2', ['residual', 'mlp_output'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 标记 final_output 复用 x 的内存（用于下一层）
    graph.add_alias(source_name='final_output', target_name='x')
    
    # 10. Logits 计算（lm_head）
    graph.add_op('logits', 'lm_head', ['final_output'], shape=(N, VOCAB_SIZE), dtype=DTYPE)
    
    logging.info("Dream block graph construction complete.")
    return graph


def define_dream_block_graph_chunkwise(
    N: int,
    chunk_logits: bool = True,
    chunk_mlp: bool = True,
    num_chunks_logits: int = 7,
    num_chunks_mlp: int = 5
) -> CustomGraph:
    """
    定义 Dream 模型单个 block 的 chunkwise 计算图（用于显存复用分析）
    在 MLP 和 Logits 计算时使用分块策略，减少峰值显存占用
    
    Args:
        N: token 数量
        chunk_logits: 是否 chunk Logits
        chunk_mlp: 是否 chunk MLP
        num_chunks_logits: Logits 分块数
        num_chunks_mlp: MLP 分块数
    """
    logging.info("Starting Dream block chunkwise graph construction...")
    graph = CustomGraph()
    
    # --- Dream 模型配置（从 config.json） ---
    HIDDEN_SIZE = 3584
    INTERMEDIATE_SIZE = 18944
    NUM_HEADS = 28
    NUM_KV_HEADS = 4
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 128
    VOCAB_SIZE = 152064
    DTYPE = torch.bfloat16
    
    # --- Chunk 配置 ---
    max_chunk_size_mlp = (N + num_chunks_mlp - 1) // num_chunks_mlp if chunk_mlp else N
    max_chunk_size_logits = (N + num_chunks_logits - 1) // num_chunks_logits if chunk_logits else N
    
    # --- 构建计算图，严格按照 Dream 的计算顺序 ---
    
    # 输入张量
    graph.add_op('x', 'placeholder', [], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 1. Attention 路径：input_layernorm
    graph.add_op('x_normed', 'input_layernorm', ['x'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 2. QKV projections
    graph.add_op('q', 'q_proj', ['x_normed'], shape=(N, NUM_HEADS * HEAD_DIM), dtype=DTYPE)
    graph.add_op('k', 'k_proj', ['x_normed'], shape=(N, NUM_KV_HEADS * HEAD_DIM), dtype=DTYPE)
    graph.add_op('v', 'v_proj', ['x_normed'], shape=(N, NUM_KV_HEADS * HEAD_DIM), dtype=DTYPE)
    
    # 3. RoPE + FlashAttention
    graph.add_op('att_out_var', 'flash_attention', ['q', 'k', 'v'], 
                 shape=(N, NUM_HEADS, HEAD_DIM), dtype=DTYPE)
    
    # 4. O projection
    graph.add_op('att_out', 'o_proj', ['att_out_var'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 5. 第一个残差连接 + MLP RMS norm (fused)
    graph.add_op('x_normed_2', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    graph.add_op('residual', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 6-8. MLP 部分
    if chunk_mlp:
        # Chunk MLP
        graph.add_op('x_gate', 'gate_proj', ['x_normed_2'], shape=(max_chunk_size_mlp, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('x_up', 'up_proj', ['x_normed_2'], shape=(max_chunk_size_mlp, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('x_mlp', 'mul_gate', ['x_gate', 'x_up'], shape=(max_chunk_size_mlp, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('mlp_output', 'down_proj', ['x_mlp'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
        graph.add_op('mlp_loop_hazard_barrier', 'barrier', 
                     ['x_gate', 'x_up', 'x_mlp', 'mlp_output'], 
                     shape=(1,), dtype=torch.int8)
        graph.add_op('xnorm2_barrier', 'barrier', ['x_normed_2'], shape=(1,), dtype=torch.int8)
    else:
        # 不 chunk MLP
        graph.add_op('x_gate', 'gate_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('x_up', 'up_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('x_mlp', 'mul_gate', ['x_gate', 'x_up'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('mlp_output', 'down_proj', ['x_mlp'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 11. 第二个残差连接
    graph.add_op('final_output', 'add_residual_2', ['residual', 'mlp_output'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 标记 final_output 复用 x 的内存（用于下一层）
    graph.add_alias(source_name='final_output', target_name='x')
    
    # 12. Logits 计算（使用 chunk size）
    graph.add_op('logits', 'lm_head', ['final_output'], shape=(max_chunk_size_logits, VOCAB_SIZE), dtype=DTYPE)
    
    logging.info("Dream block chunkwise graph construction complete.")
    return graph


def define_llada_moe_block_graph(N: int) -> CustomGraph:
    """
    定义 LLaDA-MoE 单个 block 的计算图（用于显存复用分析）
    基于 LLaDAMoEDecoderLayer 的计算流程创建节点
    
    LLaDA-MoE 的计算流程（仿照 llada/dream 风格，只包含大tensor）：
    1. x -> input_layernorm -> x_normed
    2. x_normed -> q_proj/k_proj/v_proj -> q/k/v
    3. q/k/v -> RoPE + FlashAttention -> att_out_var
    4. att_out_var -> o_proj -> att_out
    5. x + att_out -> fused_add_rms_norm -> residual + x_normed_2
    6. x_normed_2 -> [MoE部分]:
       - moe_cache1 = GEMM1 (w1: gate+up, [N*8, 2048])
       - moe_cache2 = silu_and_mul ([N*8, 1024])
       - moe_cache3 = GEMM2 (w2, [N*8, 2048])
       - moe_output = weighted_sum ([N, 2048])
    7. residual + moe_output -> final_output
    8. final_output -> lm_head -> logits
    
    注意：小tensor（router_logits, topk_weights, topk_ids）不纳入图
    """
    logging.info("Starting LLaDA-MoE block graph construction...")
    graph = CustomGraph()
    
    # --- LLaDA-MoE 配置（llada-moe-7b-a1b）---
    HIDDEN_SIZE = 2048
    NUM_HEADS = 16
    NUM_KV_HEADS = 16
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 128
    EXPERT_INTERMEDIATE = 1024
    TOPK = 8
    VOCAB_SIZE = 157184
    DTYPE = torch.bfloat16
    
    # --- 开始构建图，严格按照计算顺序 ---
    
    # 输入张量
    graph.add_op('x', 'placeholder', [], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 1. Attention RMS norm
    graph.add_op('x_normed', 'input_layernorm', ['x'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 2. QKV projections
    graph.add_op('q', 'q_proj', ['x_normed'], shape=(N, NUM_HEADS * HEAD_DIM), dtype=DTYPE)
    graph.add_op('k', 'k_proj', ['x_normed'], shape=(N, NUM_KV_HEADS * HEAD_DIM), dtype=DTYPE)
    graph.add_op('v', 'v_proj', ['x_normed'], shape=(N, NUM_KV_HEADS * HEAD_DIM), dtype=DTYPE)
    
    # 3. RoPE + FlashAttention
    # 注意：q/k/v 的 reshape 是 view 操作（原地），不需要额外节点
    # RoPE 是 in-place 的，attention 直接依赖 q/k/v
    graph.add_op('att_out_var', 'flash_attention', ['q', 'k', 'v'], 
                 shape=(N, NUM_HEADS * HEAD_DIM), dtype=DTYPE)
    
    # 4. O projection
    graph.add_op('att_out', 'o_proj', ['att_out_var'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 5. 第一个残差连接 + MLP RMS norm (fused)
    graph.add_op('x_normed_2', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    graph.add_op('residual', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # ========== MoE 部分（大tensor）==========
    # 注意：router_logits [N, 64], topk_weights [N, 8], topk_ids [N, 8] 都很小，不纳入图
    
    # 6. 第一次 GEMM: hidden @ w1 -> moe_cache1
    #    直接用最终形状 [N, topk, 2*intermediate]，避免运行时 view
    graph.add_op('moe_cache1', 'moe_gemm1', ['x_normed_2'], 
                 shape=(N, TOPK, EXPERT_INTERMEDIATE * 2), dtype=DTYPE)
    
    # 7. SiLU 激活 + element-wise multiply (原地操作，但输出到新tensor)
    #    silu(gate) * up -> moe_cache2，保持 2D 形状
    graph.add_op('moe_cache2', 'silu_and_mul', ['moe_cache1'], 
                 shape=(N * TOPK, EXPERT_INTERMEDIATE), dtype=DTYPE)
    
    # 8. 第二次 GEMM: moe_cache2 @ w2 -> moe_cache3
    #    直接用最终形状 [N, topk, hidden_size]，避免运行时 view
    graph.add_op('moe_cache3', 'moe_gemm2', ['moe_cache2'], 
                 shape=(N, TOPK, HIDDEN_SIZE), dtype=DTYPE)
    
    # 9. 加权求和并聚合回 [N, hidden_size]
    #    （topk_weights 的加权在此步，但权重tensor本身很小，不纳入图）
    graph.add_op('moe_output', 'moe_weighted_sum', ['moe_cache3'], 
                 shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 10. 第二个残差连接
    graph.add_op('final_output', 'add_residual_2', ['residual', 'moe_output'], 
                 shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 标记 final_output 复用 x 的内存（用于下一层）
    graph.add_alias(source_name='final_output', target_name='x')
    
    # 11. Logits 计算（lm_head）
    graph.add_op('logits', 'lm_head', ['final_output'], 
                 shape=(N, VOCAB_SIZE), dtype=DTYPE)
    
    logging.info("LLaDA-MoE block graph construction complete.")
    return graph


def define_llada_moe_block_graph_chunkwise(
    N: int,
    chunk_logits: bool = True,
    chunk_moe: bool = True,
    num_chunks_logits: int = 7,
    num_chunks_moe: int = 5
) -> CustomGraph:
    """
    定义 LLaDA-MoE 单个 block 的 chunkwise 计算图（用于显存复用分析）
    在 MoE 和 Logits 计算时使用分块策略，减少峰值显存占用
    
    Args:
        N: token 数量
        chunk_logits: 是否 chunk Logits
        chunk_moe: 是否 chunk MoE
        num_chunks_logits: Logits 分块数
        num_chunks_moe: MoE 分块数
    """
    logging.info("Starting LLaDA-MoE block chunkwise graph construction...")
    graph = CustomGraph()
    
    # --- LLaDA-MoE 配置（llada-moe-7b-a1b）---
    HIDDEN_SIZE = 2048
    NUM_HEADS = 16
    NUM_KV_HEADS = 16
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 128
    EXPERT_INTERMEDIATE = 1024
    TOPK = 8
    VOCAB_SIZE = 157184
    DTYPE = torch.bfloat16
    
    # --- Chunk 配置 ---
    max_chunk_size_moe = (N + num_chunks_moe - 1) // num_chunks_moe if chunk_moe else N
    max_chunk_size_logits = (N + num_chunks_logits - 1) // num_chunks_logits if chunk_logits else N
    
    # --- 开始构建图，严格按照计算顺序 ---
    
    # 输入张量
    graph.add_op('x', 'placeholder', [], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 1. Attention RMS norm
    graph.add_op('x_normed', 'input_layernorm', ['x'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 2. QKV projections
    graph.add_op('q', 'q_proj', ['x_normed'], shape=(N, NUM_HEADS * HEAD_DIM), dtype=DTYPE)
    graph.add_op('k', 'k_proj', ['x_normed'], shape=(N, NUM_KV_HEADS * HEAD_DIM), dtype=DTYPE)
    graph.add_op('v', 'v_proj', ['x_normed'], shape=(N, NUM_KV_HEADS * HEAD_DIM), dtype=DTYPE)
    
    # 3. RoPE + FlashAttention
    graph.add_op('att_out_var', 'flash_attention', ['q', 'k', 'v'], 
                 shape=(N, NUM_HEADS * HEAD_DIM), dtype=DTYPE)
    
    # 4. O projection
    graph.add_op('att_out', 'o_proj', ['att_out_var'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 5. 第一个残差连接 + MLP RMS norm (fused)
    graph.add_op('x_normed_2', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    graph.add_op('residual', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # ========== MoE 部分 ==========
    # 注意：router_logits, topk_weights, topk_ids 都很小，不纳入图
    
    if chunk_moe:
        # Chunk MoE
        graph.add_op('moe_cache1', 'moe_gemm1', ['x_normed_2'], 
                     shape=(max_chunk_size_moe, TOPK, EXPERT_INTERMEDIATE * 2), dtype=DTYPE)
        graph.add_op('moe_cache2', 'silu_and_mul', ['moe_cache1'], 
                     shape=(max_chunk_size_moe * TOPK, EXPERT_INTERMEDIATE), dtype=DTYPE)
        graph.add_op('moe_cache3', 'moe_gemm2', ['moe_cache2'], 
                     shape=(max_chunk_size_moe, TOPK, HIDDEN_SIZE), dtype=DTYPE)
        graph.add_op('moe_output', 'moe_weighted_sum', ['moe_cache3'], 
                     shape=(N, HIDDEN_SIZE), dtype=DTYPE)
        graph.add_op('moe_loop_hazard_barrier', 'barrier', 
                     ['moe_cache1', 'moe_cache2', 'moe_cache3', 'moe_output'], 
                     shape=(1,), dtype=torch.int8)
        graph.add_op('xnorm2_barrier', 'barrier', ['x_normed_2'], shape=(1,), dtype=torch.int8)
    else:
        # 不 chunk MoE
        graph.add_op('moe_cache1', 'moe_gemm1', ['x_normed_2'], 
                     shape=(N, TOPK, EXPERT_INTERMEDIATE * 2), dtype=DTYPE)
        graph.add_op('moe_cache2', 'silu_and_mul', ['moe_cache1'], 
                     shape=(N * TOPK, EXPERT_INTERMEDIATE), dtype=DTYPE)
        graph.add_op('moe_cache3', 'moe_gemm2', ['moe_cache2'], 
                     shape=(N, TOPK, HIDDEN_SIZE), dtype=DTYPE)
        graph.add_op('moe_output', 'moe_weighted_sum', ['moe_cache3'], 
                     shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 12. 第二个残差连接
    graph.add_op('final_output', 'add_residual_2', ['residual', 'moe_output'], 
                 shape=(N, HIDDEN_SIZE), dtype=DTYPE)
    
    # 标记 final_output 复用 x 的内存（用于下一层）
    graph.add_alias(source_name='final_output', target_name='x')
    
    # 13. Logits 计算（使用 chunk size）
    graph.add_op('logits', 'lm_head', ['final_output'], 
                 shape=(max_chunk_size_logits, VOCAB_SIZE), dtype=DTYPE)
    
    logging.info("LLaDA-MoE block chunkwise graph construction complete.")
    return graph


def define_llada_block_graph_chunkwise(
    N: int,
    chunk_logits: bool = True,
    chunk_mlp: bool = True,
    num_chunks_logits: int = 7,
    num_chunks_mlp: int = 5
) -> CustomGraph:
    """
    使用您的 CustomGraph，手动、声明式地构建 LLaDALlamaBlock 的计算图。
    基于当前的计算流程创建节点
    
    Args:
        N: token 数量
        chunk_logits: 是否 chunk Logits
        chunk_mlp: 是否 chunk MLP
        num_chunks_logits: Logits 分块数
        num_chunks_mlp: MLP 分块数
    """
    logging.info("Starting manual graph construction with CustomGraph...")
    graph = CustomGraph()
    
    # --- 定义模型配置 ---
    D_MODEL = 4096
    INTERMEDIATE_SIZE = 12288
    NUM_HEAD_Q = 32
    NUM_HEAD_KV = 32
    HEAD_DIM = D_MODEL // NUM_HEAD_Q
    VOCAB_SIZE = 126464
    DTYPE = torch.bfloat16

    max_chunk_size_mlp = (N + num_chunks_mlp - 1) // num_chunks_mlp if chunk_mlp else N
    max_chunk_size_logits = (N + num_chunks_logits - 1) // num_chunks_logits if chunk_logits else N


    # --- 开始构建图，严格按照你的计算顺序 ---
    
    # 输入张量
    graph.add_op('x', 'placeholder', [], shape=(N, D_MODEL), dtype=DTYPE)
    
    # 1. Attention RMS norm
    graph.add_op('x_normed', 'attn_norm', ['x'], shape=(N, D_MODEL), dtype=DTYPE)
    
    # 2. QKV projections
    graph.add_op('q', 'q_proj', ['x_normed'], shape=(N, D_MODEL), dtype=DTYPE)
    graph.add_op('k', 'k_proj', ['x_normed'], shape=(N, NUM_HEAD_KV * HEAD_DIM), dtype=DTYPE)
    graph.add_op('v', 'v_proj', ['x_normed'], shape=(N, NUM_HEAD_KV * HEAD_DIM), dtype=DTYPE)
    
    # 3. Attention operation
    graph.add_op('att_tmp', 'attention_op', ['q', 'k', 'v'], shape=(N, NUM_HEAD_Q * HEAD_DIM), dtype=DTYPE)
    
    # 4. Attention output (after o_proj)
    graph.add_op('att_out', 'o_proj', ['att_tmp'], shape=(N, D_MODEL), dtype=DTYPE)
    
    # 5. Residual connection + MLP RMS norm
    graph.add_op('x_normed_2', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, D_MODEL), dtype=DTYPE)
    graph.add_op('residual', 'fused_add_rms_norm_out', ['x', 'att_out'], shape=(N, D_MODEL), dtype=DTYPE)
    

    # 6-8. MLP 部分
    if chunk_mlp:
        # Chunk MLP
        graph.add_op('mlp_x', 'ff_proj', ['x_normed_2'], shape=(max_chunk_size_mlp, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('mlp_x_up', 'up_proj', ['x_normed_2'], shape=(max_chunk_size_mlp, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('mlp_gated', 'mul_op', ['mlp_x', 'mlp_x_up'], shape=(max_chunk_size_mlp, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('mlp_output', 'ff_out', ['mlp_gated'], shape=(N, D_MODEL), dtype=DTYPE)
        graph.add_op('mlp_loop_hazard_barrier', 'barrier', 
                     ['mlp_x', 'mlp_x_up', 'mlp_gated', 'mlp_output'], 
                     shape=(1,), dtype=torch.int8)
        graph.add_op('xnorm2_barrier', 'barrier', ['x_normed_2'], shape=(1,), dtype=torch.int8)
    else:
        # 不 chunk MLP
        graph.add_op('mlp_x', 'ff_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('mlp_x_up', 'up_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('mlp_gated', 'mul_op', ['mlp_x', 'mlp_x_up'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
        graph.add_op('mlp_output', 'ff_out', ['mlp_gated'], shape=(N, D_MODEL), dtype=DTYPE)
    
    graph.add_op('final_output', 'add_residual_2', ['residual', 'mlp_output'], shape=(N, D_MODEL), dtype=DTYPE)
    graph.add_alias(source_name='final_output', target_name='x')

    graph.add_op('logits','get_logits',['final_output'],shape=(max_chunk_size_logits,VOCAB_SIZE),dtype=DTYPE)

    logging.info("Manual graph construction complete.")
    return graph





if __name__ == "__main__":
    N = 110000  # token 数量
    
    # 1. 使用你的函数定义图
    graph_blueprint = define_llada_block_graph(N=N)
    #graph_blueprint = define_llada_block_graph_chunkwise(N=100)
    
    # 2. 打印图的结构
    graph_blueprint.print_graph()

    # 3. 实例化并执行分析（池内显存）
    analyzer = MemoryAnalyzer(graph_blueprint)
    final_analysis = analyzer.analyze()
    
    # 4. 计算池外 activation 显存（forward 返回值 + execute_diffusion_once 后处理，随 N scale）
    print("\n" + "="*25 + " 池外 ACTIVATION 显存分析 " + "="*25)
    
    # === LLaDAOutput 返回字段 ===
    # index (x0_2d): [N] int64，采样后的 token ID
    index_bytes = N * 8
    index_mb = index_bytes / (1024**2)
    print(f"index (x0_2d): {index_mb:.4f} MB  # [N] int64, LLaDAOutput.index")
    
    # confidence (x0_p_2d): [N] bfloat16，采样 token 的概率
    confidence_bytes = N * 2
    confidence_mb = confidence_bytes / (1024**2)
    print(f"confidence (x0_p_2d): {confidence_mb:.4f} MB  # [N] bfloat16, LLaDAOutput.confidence")
    
    # === execute_diffusion_once 后处理中的 mask/constraint 张量 ===
    # flat_x: [N] int64，扁平化的 x
    flat_x_bytes = N * 8
    flat_x_mb = flat_x_bytes / (1024**2)
    print(f"flat_x: {flat_x_mb:.4f} MB  # [N] int64, 扁平化输入")
    
    # mask_2d: [N] bool，mask 位置标记
    mask_2d_bytes = N * 1  # bool = 1 byte
    mask_2d_mb = mask_2d_bytes / (1024**2)
    print(f"mask_2d: {mask_2d_mb:.4f} MB  # [N] bool, mask 位置")
    
    # block_2d: [N] bool，block 约束标记
    block_2d_bytes = N * 1
    block_2d_mb = block_2d_bytes / (1024**2)
    print(f"block_2d: {block_2d_mb:.4f} MB  # [N] bool, block 约束")
    
    # confidence_2d: [N] bfloat16，置信度（用于 TopK 选择）
    confidence_2d_bytes = N * 2
    confidence_2d_mb = confidence_2d_bytes / (1024**2)
    print(f"confidence_2d: {confidence_2d_mb:.4f} MB  # [N] bfloat16, 置信度缓存")
    
    # transfer_2d: [N] bool，传输标记（TopK 选中的位置）
    transfer_2d_bytes = N * 1
    transfer_2d_mb = transfer_2d_bytes / (1024**2)
    print(f"transfer_2d: {transfer_2d_mb:.4f} MB  # [N] bool, TopK 选中标记")
    
    # final_x_2d: [N] int64，最终结果（应用 transfer 后）
    final_x_2d_bytes = N * 8
    final_x_2d_mb = final_x_2d_bytes / (1024**2)
    print(f"final_x_2d: {final_x_2d_mb:.4f} MB  # [N] int64, 最终结果")
    
    # 池外总计
    out_of_pool_bytes = (index_bytes + confidence_bytes + flat_x_bytes + 
                         mask_2d_bytes + block_2d_bytes + confidence_2d_bytes + 
                         transfer_2d_bytes + final_x_2d_bytes)
    out_of_pool_mb = out_of_pool_bytes / (1024**2)
    print(f"\n池外 activation 总计: {out_of_pool_mb:.4f} MB")
    
    # 5. 总峰值显存 = 池内 + 池外
    pool_peak_mb = final_analysis['peak_memory_mb']
    total_peak_mb = pool_peak_mb + out_of_pool_mb
    
    print("\n" + "="*25 + " 总峰值显存 (ACTIVATION ONLY) " + "="*25)
    print(f"池内峰值 (activation pool): {pool_peak_mb:.4f} MB")
    print(f"池外 activation: {out_of_pool_mb:.4f} MB")
    print(f"总峰值 activation: {total_peak_mb:.4f} MB")
    print(f"池外占比: {(out_of_pool_mb/total_peak_mb)*100:.2f}%")
    print("="*75)
    
    # 6. 打印池内分析报告
    print("\n" + "="*25 + " 池内分析详情 " + "="*25)
    pprint.pprint(final_analysis)
    print("="*75)
    
    # 6. 打印显存占有率时间线
    print("\n" + "="*30 + " MEMORY OCCUPANCY TIMELINE " + "="*30)
    timeline = final_analysis['memory_timeline']
    peak_memory_mb = final_analysis['peak_memory_mb']
    
    print(f"峰值显存: {peak_memory_mb:.4f} MB")
    print(f"{'时刻':<4} {'操作':<15} {'显存(MB)':<12} {'占有率(%)':<10} {'存活张量数':<8}")
    print("-" * 60)
    
    for step in timeline:
        print(f"{step['timestep']:<4} {step['node_name']:<15} {step['total_memory_mb']:<12.4f} "
              f"{step['occupancy_percentage']:<10.2f} {step['live_tensor_count']:<8}")
    
    print("="*90)
    
    # 7. 自动验证计划是否符合我们的指令
    print("\n--- Verification ---")
    plan = final_analysis['reuse_plan']
    
    if plan.get('final_output') and plan.get('x') and plan['final_output']['offset'] == plan['x']['offset']:
        print(f"✅ Alias Verification PASSED: 'final_output' and 'x' share the same memory offset ({plan['x']['offset']}).")
    else:
        print(f"❌ Alias Verification FAILED: 'final_output' did not correctly reuse the memory of 'x'.")