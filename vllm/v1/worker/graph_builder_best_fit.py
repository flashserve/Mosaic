import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple, Dict, Set
import json
import logging
import pprint

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
        """
        Hybrid Best-Fit + Greedy strategy:
        1. Use conflict-aware ordering (inspired by Greedy-Coloring)
        2. Within each "batch" of tensors, prioritize by: size (large first) and conflicts (high first)
        3. Use Best-Fit for gap selection
        
        This combines:
        - Execution order locality (from First-Fit)
        - Conflict awareness (from Greedy-Coloring)
        - Best-Fit gap selection (reduces fragmentation)
        """
        births = {node.output_name: idx for idx, node in enumerate(self.graph.execution_order)}
        deaths = {name: -1 for name in self.graph.nodes}
        for idx, node in enumerate(self.graph.execution_order):
            for input_name in node.input_names:
                deaths[input_name] = idx

        # Build conflict graph
        tensor_names = [node.output_name for node in self.graph.execution_order 
                       if node.output_spec.size_bytes > 0 and node.output_name not in self.graph.aliases]
        
        conflicts = {name: set() for name in tensor_names}
        for i, name_i in enumerate(tensor_names):
            birth_i = births[name_i]
            death_i = deaths.get(name_i, len(self.graph.execution_order))
            for name_j in tensor_names[i+1:]:
                birth_j = births[name_j]
                death_j = deaths.get(name_j, len(self.graph.execution_order))
                if not (death_i < birth_j or death_j < birth_i):
                    conflicts[name_i].add(name_j)
                    conflicts[name_j].add(name_i)
        
        # Group tensors by birth time (window size = 5 for local optimization)
        window_size = 5
        ordered_tensors = []
        for i in range(0, len(self.graph.execution_order), window_size):
            window = self.graph.execution_order[i:i+window_size]
            # Within each window, sort by: 1) conflict count (high first), 2) size (large first)
            window_sorted = sorted(
                window,
                key=lambda n: (
                    -len(conflicts.get(n.output_name, set())),  # More conflicts = higher priority
                    -n.output_spec.size_bytes  # Larger size = higher priority
                )
            )
            ordered_tensors.extend(window_sorted)
        
        allocated_blocks: List[Dict[str, Any]] = []
        reuse_plan: Dict[str, Dict[str, Any]] = {}
        max_offset = 0

        # Process tensors in optimized order
        for node in ordered_tensors:
            tensor_name = node.output_name
            tensor_spec = node.output_spec
            tensor_size = tensor_spec.size_bytes
            if not tensor_size > 0: continue
            
            current_birth_time = births[tensor_name]
            
            # Find all blocks that are live when this tensor is born
            live_blocks_at_birth = []
            for block in allocated_blocks:
                block_name = block['tensor_name']
                if deaths.get(block_name, len(births)) >= current_birth_time:
                    live_blocks_at_birth.append(block)

            live_blocks_at_birth.sort(key=lambda b: b['start'])

            if tensor_name in self.graph.aliases:
                target_name = self.graph.aliases[tensor_name]
                if target_name not in reuse_plan:
                    raise RuntimeError(f"Alias error: Target '{target_name}' has not been allocated yet.")
                
                found_offset = reuse_plan[target_name]['offset']
                logging.info(f"Applying alias for '{tensor_name}': reusing offset {found_offset} from '{target_name}'.")
            else:
                # Best-Fit: find the SMALLEST gap that fits
                best_offset = -1
                best_gap_size = float('inf')
                
                last_end = 0
                for block in live_blocks_at_birth:
                    gap_size = block['start'] - last_end
                    if gap_size >= tensor_size and gap_size < best_gap_size:
                        best_offset = last_end
                        best_gap_size = gap_size
                    last_end = block['end']
                
                # Check the gap after the last block
                if best_offset == -1:
                    best_offset = last_end
                
                found_offset = best_offset
            
            reuse_plan[tensor_name] = {
                'offset': found_offset, 'size': tensor_size,
                'shape': tensor_spec.shape, 'dtype': tensor_spec.dtype
            }
            allocated_blocks.append({'tensor_name': tensor_name, 'start': found_offset, 'end': found_offset + tensor_size})
            max_offset = max(max_offset, max(b['end'] for b in allocated_blocks) if allocated_blocks else 0)

        # Handle aliases
        for node in self.graph.execution_order:
            tensor_name = node.output_name
            if tensor_name in self.graph.aliases and tensor_name not in reuse_plan:
                target_name = self.graph.aliases[tensor_name]
                if target_name not in reuse_plan:
                    raise RuntimeError(f"Alias error: Target '{target_name}' has not been allocated yet.")
                
                found_offset = reuse_plan[target_name]['offset']
                reuse_plan[tensor_name] = {
                    'offset': found_offset,
                    'size': node.output_spec.size_bytes,
                    'shape': node.output_spec.shape,
                    'dtype': node.output_spec.dtype
                }

        reuse_plan['_total_buffer_size_bytes'] = max_offset
        logging.info(f"Hybrid Best-Fit+Greedy allocation: total memory = {max_offset / (1024**2):.2f} MB")
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
    # 1. 使用你的函数定义图
    graph_blueprint = define_llada_block_graph(N=64000)
    #graph_blueprint = define_llada_block_graph_chunkwise(N=100)
    # 2. 添加核心指令：强制 final_output 复用 x 的内存
    # graph_blueprint.add_alias(source_name='final_output', target_name='x')
    
    # 3. 打印图的结构和我们添加的指令
    graph_blueprint.print_graph()

    # 4. 实例化并执行分析
    analyzer = MemoryAnalyzer(graph_blueprint)
    final_analysis = analyzer.analyze()

    # 5. 打印最终分析报告
    print("\n" + "="*25 + " FINAL ANALYSIS REPORT " + "="*25)
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