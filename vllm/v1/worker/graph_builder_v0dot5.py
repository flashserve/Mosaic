import torch
import torch.nn as nn
import torch.nn.functional as F
# 注意：我们不再需要 torch.fx 了！
from typing import Any, List, Optional, Tuple, Dict, Set
import json
import logging
import pprint

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# PART 1: YOUR CUSTOM GRAPH FRAMEWORK (您的自定义图框架)
# ==============================================================================

class TensorSpec:
    """一个简单的数据类，用于存储Tensor的元数据。"""
    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype
        
        # 计算并缓存大小（字节）
        if dtype.is_floating_point:
            bytes_per_element = torch.finfo(dtype).bits // 8
        else:
            bytes_per_element = torch.iinfo(dtype).bits // 8
        self.size_bytes = torch.prod(torch.tensor(shape)).item() * bytes_per_element

    def __repr__(self) -> str:
        return f"TensorSpec(shape={list(self.shape)}, dtype={self.dtype}, size_bytes={self.size_bytes})"

class OpNode:
    """代表一个操作节点，满足您的核心需求。"""
    def __init__(self, op_type: str, output_name: str, output_spec: TensorSpec, input_names: List[str]):
        self.op_type = op_type
        self.output_name = output_name
        self.output_spec = output_spec
        self.input_names = input_names
        
    def __repr__(self) -> str:
        return f"{self.output_name} = {self.op_type}({', '.join(self.input_names)})  |  {self.output_spec}"

class CustomGraph:
    """我们自己的、极简的计算图。"""
    def __init__(self):
        self.nodes: Dict[str, OpNode] = {}
        self.execution_order: List[OpNode] = []

    def add_op(self, output_name: str, op_type: str, input_names: List[str], shape: Tuple[int, ...], dtype: torch.dtype):
        if output_name in self.nodes:
            raise ValueError(f"Tensor with name '{output_name}' already exists in the graph.")
        spec = TensorSpec(shape, dtype)
        node = OpNode(op_type, output_name, spec, input_names)
        self.nodes[output_name] = node
        self.execution_order.append(node)
        return node
        
    def print_graph(self):
        print("--- Custom Computation Graph ---")
        for node in self.execution_order:
            print(node)
        print("------------------------------")

# ==============================================================================
# PART 2: THE ADAPTED MEMORY ANALYZER (适配后的内存分析器)
# ==============================================================================

class MemoryAnalyzer:
    """
    【适配版】
    接收一个 CustomGraph 对象，并对其进行内存分析，生成优化复用计划。
    """
    def __init__(self, graph: CustomGraph):
        self.graph = graph
        self.node_indices = {node.output_name: i for i, node in enumerate(self.graph.execution_order)}

    def analyze(self) -> Dict[str, Any]:
        logging.info("Starting memory analysis on CustomGraph...")
        
        liveness_map = self._analyze_liveness()
        logging.info("Liveness analysis complete.")
        
        peak_info = self._calculate_peak_memory(liveness_map)
        logging.info("Peak memory calculation complete.")
        
        reuse_plan = self._generate_optimized_reuse_plan()
        logging.info("Optimized reuse plan generation complete.")

        # 返回的计划中包含原生torch对象，方便直接使用
        return {
            "peak_memory_mb": peak_info["peak_memory_bytes"] / (1024**2),
            "peak_node_name": peak_info["peak_node"].output_name,
            "peak_live_tensors": {
                name: f"{self.graph.nodes[name].output_spec.size_bytes / (1024**2):.4f} MB"
                for name in peak_info["peak_live_set_names"]
            },
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
            memory_of_others = sum(self.graph.nodes[name].output_spec.size_bytes for name in live_after_node_names)
            
            current_total_memory = current_node_size + memory_of_others
            if current_total_memory > peak_memory_bytes:
                peak_memory_bytes = current_total_memory
                peak_node = node
                peak_live_set_names = live_after_node_names.union({node.output_name})
                
        return {"peak_memory_bytes": peak_memory_bytes, "peak_node": peak_node, "peak_live_set_names": peak_live_set_names}

    def _generate_optimized_reuse_plan(self) -> Dict[str, Dict[str, Any]]:
        births = {node.output_name: idx for idx, node in enumerate(self.graph.execution_order)}
        deaths = {name: -1 for name in self.graph.nodes}
        for idx, node in enumerate(self.graph.execution_order):
            for input_name in node.input_names:
                deaths[input_name] = idx

        allocated_blocks: List[Dict[str, Any]] = []
        reuse_plan: Dict[str, Dict[str, Any]] = {}
        max_offset = 0

        for idx, node in enumerate(self.graph.execution_order):
            tensor_size = node.output_spec.size_bytes
            if not tensor_size > 0: continue
            
            live_blocks_at_birth = [b for b in allocated_blocks if deaths.get(b['tensor_name'], -1) > births[node.output_name]]
            live_blocks_at_birth.sort(key=lambda b: b['start'])

            last_end = 0
            found_offset = -1
            for block in live_blocks_at_birth:
                gap = block['start'] - last_end
                if gap >= tensor_size:
                    found_offset = last_end
                    break
                last_end = block['end']

            if found_offset == -1:
                found_offset = last_end
            
            reuse_plan[node.output_name] = {
                'offset': found_offset, 
                'size': tensor_size,
                'shape': node.output_spec.shape,
                'dtype': node.output_spec.dtype
            }
            allocated_blocks.append({'tensor_name': node.output_name, 'start': found_offset, 'end': found_offset + tensor_size})
            max_offset = max(max_offset, max(b['end'] for b in allocated_blocks) if allocated_blocks else 0)

        reuse_plan['_total_buffer_size_bytes'] = max_offset
        return reuse_plan

# ==============================================================================
# PART 3: THE APPLICATION (主执行流程)
# ==============================================================================

# def define_llada_block_graph() -> CustomGraph:
#     """
#     使用您的 CustomGraph，手动、声明式地构建 LLaDALlamaBlock 的计算图。
#     """
#     logging.info("Starting manual graph construction with CustomGraph...")
#     graph = CustomGraph()

#     # --- 定义模型配置 ---
#     D_MODEL = 4096
#     INTERMEDIATE_SIZE = 12288
#     NUM_HEAD_Q=32
#     NUM_HEAD_KV=32
#     HEAD_DIM=D_MODEL//NUM_HEAD_Q
#     DTYPE = torch.bfloat16
#     N = 1  # 为“单位token”进行分析

#     # --- 开始构建图 ---
#     graph.add_op('x', 'placeholder', [], shape=(N, D_MODEL), dtype=DTYPE)
#     graph.add_op('x_normed', 'attn_norm', ['x'], shape=(N, D_MODEL), dtype=DTYPE)
#     graph.add_op('q', 'q_proj', ['x_normed'], shape=(N, D_MODEL), dtype=DTYPE)
#     graph.add_op('k', 'k_proj', ['x_normed'], shape=(N, D_MODEL), dtype=DTYPE)
#     graph.add_op('v', 'v_proj', ['x_normed'], shape=(N, D_MODEL), dtype=DTYPE)
#     graph.add_op('att_tmp', 'attention_op', ['q', 'k', 'v'], shape=(N, NUM_HEAD_Q, HEAD_DIM), dtype=DTYPE)
#     graph.add_op('att_out', 'o_proj', ['att_tmp'], shape=(N, D_MODEL), dtype=DTYPE)
#     #graph.add_op('x_after_attn', 'add_residual_1', ['x', 'att_out'], shape=(N, D_MODEL), dtype=DTYPE)
#     graph.add_op('x_normed_2', 'ff_norm', ['x_after_attn'], shape=(N, D_MODEL), dtype=DTYPE)
#     graph.add_op('x_after_attn', 'add_residual_1', ['x', 'att_out'], shape=(N, D_MODEL), dtype=DTYPE)
#     graph.add_op('gate', 'ff_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
#     graph.add_op('up', 'up_proj', ['x_normed_2'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
#     graph.add_op('activated_gate', 'silu_op', ['gate'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
#     graph.add_op('fused_mlp', 'mul_op', ['activated_gate', 'up'], shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
#     graph.add_op('mlp_output', 'ff_out', ['fused_mlp'], shape=(N, D_MODEL), dtype=DTYPE)
#     graph.add_op('final_output', 'add_residual_2', ['x_after_attn', 'mlp_output'], shape=(N, D_MODEL), dtype=DTYPE)
    
#     logging.info("Manual graph construction complete.")
#     return graph



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
    DTYPE = torch.bfloat16
    #N = 1  # 为"单位token"进行分析

    # --- 开始构建图，严格按照计算顺序 ---
    
    # 输入张量
    graph.add_op('x', 'placeholder', [], shape=(N, D_MODEL), dtype=DTYPE)
    
    # 1. Attention RMS norm
    graph.add_op('x_normed', 'attn_norm', ['x'], shape=(N, D_MODEL), dtype=DTYPE)
    
    # 2. QKV projections
    graph.add_op('q', 'q_proj', ['x_normed'], shape=(N, D_MODEL), dtype=DTYPE)
    graph.add_op('k', 'k_proj', ['x_normed'], shape=(N, NUM_HEAD_KV * HEAD_DIM), dtype=DTYPE)
    graph.add_op('v', 'v_proj', ['x_normed'], shape=(N, NUM_HEAD_KV * HEAD_DIM), dtype=DTYPE)
    
    # 3. Attention operation
    graph.add_op('att_tmp', 'attention_op', ['q', 'k', 'v'], shape=(N, NUM_HEAD_Q, HEAD_DIM), dtype=DTYPE)
    
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
    #graph.add_op('final_output', 'add_residual_2', ['residual', 'mlp_output'], shape=(N, D_MODEL), dtype=DTYPE)
    graph.add_op('final_output', 'add_residual_2', ['x', 'mlp_output'], shape=(N, D_MODEL), dtype=DTYPE)
    
    logging.info("Manual graph construction complete.")
    return graph






















if __name__ == "__main__":
    # 1. 手动定义并获取计算图“蓝图”
    graph_blueprint = define_llada_block_graph()
    graph_blueprint.print_graph()

    # 2. 实例化分析器，传入您的自定义图
    analyzer = MemoryAnalyzer(graph_blueprint)

    # 3. 执行分析，获取计划
    final_analysis = analyzer.analyze()

    # 4. 打印最终的、可用于运行时的静态复用计划
    print("\n" + "="*20 + " FINAL STATIC REUSE PLAN " + "="*20)
    pprint.pprint(final_analysis)
    print("="*65)
    
    print("\n解读:")
    print("1. 'peak_memory_mb': 在处理1个token时，理论上需要的最大激活内存。")
    print("2. 'peak_live_tensors': 在内存最紧张时，必须同时存在的张量列表。")
    print("3. 'reuse_plan': 您的静态复用模板！它详细说明了每个张量应有的偏移量、")
    print("   大小、形状和数据类型。这份计划可以直接用于运行时的内存分配。")