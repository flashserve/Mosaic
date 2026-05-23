import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from typing import Any, List, Optional, Tuple, Dict, Set
import json
import logging
import pprint # 使用pprint来优雅地打印复杂字典

# --- 配置日志记录 ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# PART 1: THE TOOLS (我们的工具箱)
# ==============================================================================

class GraphBuilder:
    """
    一个用于程序化、手动构建 torch.fx.Graph 的辅助类。
    它的产出是一张纯粹的、带有元数据的“建筑蓝图”（fx.Graph）。
    """
    def __init__(self):
        self.graph = fx.Graph()

    def add_node(self, op: str, target: Any, args: Tuple[Any, ...] = (), name: Optional[str] = None, 
                 shape: Optional[Tuple[int, ...]] = None, dtype: Optional[torch.dtype] = None) -> fx.Node:
        new_node = self.graph.create_node(op, target, args=args, name=name)
        if shape is not None and dtype is not None:
            new_node.meta['tensor_meta'] = TensorMetadata(
                shape=torch.Size(shape), dtype=dtype, requires_grad=False,
                stride=self._calculate_stride(shape), memory_format=torch.contiguous_format,
                is_quantized=False, qparams={}
            )
        return new_node

    def get_graph(self) -> fx.Graph:
        return self.graph

    @staticmethod
    def _calculate_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = stride[i+1] * shape[i+1]
        return tuple(stride)


class MemoryAnalyzer:
    """
    接收一个信息完备的 fx.Graph，并对其进行内存分析，生成包含原生torch对象的优化复用计划。
    """
    def __init__(self, graph: fx.Graph):
        self.graph = graph
        # 预计算每个节点的索引，方便查找
        self.node_indices = {node: i for i, node in enumerate(self.graph.nodes)}

    def analyze(self) -> Dict[str, Any]:
        logging.info("Starting memory analysis...")
        
        liveness_map = self._analyze_liveness()
        logging.info("Liveness analysis complete.")
        
        peak_info = self._calculate_peak_memory(liveness_map)
        logging.info("Peak memory calculation complete.")
        
        reuse_plan = self._generate_optimized_reuse_plan()
        logging.info("Optimized reuse plan generation complete.")

        return {
            "peak_memory_mb": peak_info["peak_memory_bytes"] / (1024**2),
            "peak_node": peak_info["peak_node"].name,
            "peak_live_tensors": {
                node.name: f"{self._get_tensor_size_bytes(node) / (1024**2):.4f} MB"
                for node in peak_info["peak_live_set"]
            },
            "reuse_plan": reuse_plan
        }
        
    def _get_tensor_size_bytes(self, node: fx.Node) -> int:
        meta = node.meta.get('tensor_meta')
        if not meta or not hasattr(meta, 'dtype'): return 0
        dtype, shape = meta.dtype, meta.shape
        bytes_per_element = (torch.finfo(dtype).bits // 8) if dtype.is_floating_point else (torch.iinfo(dtype).bits // 8)
        return torch.prod(torch.tensor(shape)).item() * bytes_per_element

    def _analyze_liveness(self) -> Dict[fx.Node, Set[fx.Node]]:
        liveness_map: Dict[fx.Node, Set[fx.Node]] = {}
        live_nodes: Set[fx.Node] = set()
        for node in reversed(self.graph.nodes):
            liveness_map[node] = live_nodes.copy()
            if node in live_nodes:
                live_nodes.remove(node)
            for input_node in node.all_input_nodes:
                live_nodes.add(input_node)
        return liveness_map

    def _calculate_peak_memory(self, liveness_map: Dict[fx.Node, Set[fx.Node]]) -> Dict:
        peak_memory_bytes, peak_node, peak_live_set = 0, None, set()
        for node in self.graph.nodes:
            current_node_size = self._get_tensor_size_bytes(node)
            live_after_node = liveness_map.get(node, set())
            memory_of_others = sum(self._get_tensor_size_bytes(n) for n in live_after_node)
            current_total_memory = current_node_size + memory_of_others
            if current_total_memory > peak_memory_bytes:
                peak_memory_bytes = current_total_memory
                peak_node = node
                peak_live_set = live_after_node.union({node})
        return {"peak_memory_bytes": peak_memory_bytes, "peak_node": peak_node, "peak_live_set": peak_live_set}

    def _generate_optimized_reuse_plan(self) -> Dict[str, Dict[str, Any]]:
        births = {node: idx for idx, node in enumerate(self.graph.nodes)}
        deaths = {node: -1 for node in self.graph.nodes}
        for idx, node in enumerate(self.graph.nodes):
            for input_node in node.all_input_nodes:
                deaths[input_node] = idx

        allocated_blocks: List[Dict[str, Any]] = []
        reuse_plan: Dict[str, Dict[str, Any]] = {}
        max_offset = 0

        for idx, node in enumerate(self.graph.nodes):
            tensor_size = self._get_tensor_size_bytes(node)
            if not tensor_size > 0:
                continue
            
            tensor_meta = node.meta['tensor_meta']
            
            live_blocks_at_birth = [b for b in allocated_blocks if deaths.get(b['tensor'], -1) > births[node]]
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
            
            reuse_plan[node.name] = {
                'offset': found_offset, 
                'size': tensor_size,
                'shape': tensor_meta.shape,
                'dtype': tensor_meta.dtype
            }
            allocated_blocks.append({'tensor': node, 'start': found_offset, 'end': found_offset + tensor_size})
            max_offset = max(max_offset, max(b['end'] for b in allocated_blocks) if allocated_blocks else 0)

        reuse_plan['_total_buffer_size_bytes'] = max_offset
        return reuse_plan

# ==============================================================================
# PART 2: THE APPLICATION (使用工具来构建和分析您的模型)
# ==============================================================================

def define_llada_block_graph() -> fx.Graph:
    """
    手动、声明式地构建 LLaDALlamaBlock 的计算图。
    """
    logging.info("Starting manual graph construction...")
    builder = GraphBuilder()

    # --- 定义模型配置（通常从config.json读取） ---
    D_MODEL = 4096
    INTERMEDIATE_SIZE = 12288  # 修正：根据您之前的日志，LLaDA 8B的尺寸是11008
    head_num_q=32
    head_num_kv=32
    DTYPE = torch.bfloat16
    head_num_q=32
    N = 1  # 我们为“单位token”进行分析

    # --- 开始构建图 ---

    # 1. 输入
    x = builder.add_node('placeholder', 'x', name='x', shape=(N, D_MODEL), dtype=DTYPE)
    
    # --- Attention部分 ---
    x_normed = builder.add_node('call_module', 'attn_norm', args=(x,), name='x_normed', shape=(N, D_MODEL), dtype=DTYPE)
    q = builder.add_node('call_module', 'q_proj', args=(x_normed,), name='q', shape=(N, D_MODEL), dtype=DTYPE)
    k = builder.add_node('call_module', 'k_proj', args=(x_normed,), name='k', shape=(N, D_MODEL), dtype=DTYPE)
    v = builder.add_node('call_module', 'v_proj', args=(x_normed,), name='v', shape=(N, D_MODEL), dtype=DTYPE)
    
    # 模拟Attention操作的输出投影
    att_out = builder.add_node('call_module', 'attn_out', args=(v,), name='att_out', shape=(N, D_MODEL), dtype=DTYPE)
    
    # 第一个残差连接
    x_after_attn = builder.add_node('call_function', torch.add, args=(x, att_out), name='x_after_attn', shape=(N, D_MODEL), dtype=DTYPE)

    # --- MLP部分 ---
    x_normed_2 = builder.add_node('call_module', 'ff_norm', args=(x_after_attn,), name='x_normed_2', shape=(N, D_MODEL), dtype=DTYPE)
    
    gate = builder.add_node('call_module', 'ff_proj', args=(x_normed_2,), name='gate', shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    up = builder.add_node('call_module', 'up_proj', args=(x_normed_2,), name='up', shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    
    activated_gate = builder.add_node('call_function', F.silu, args=(gate,), name='activated_gate', shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    fused_mlp = builder.add_node('call_function', torch.mul, args=(activated_gate, up), name='fused_mlp', shape=(N, INTERMEDIATE_SIZE), dtype=DTYPE)
    
    mlp_output = builder.add_node('call_module', 'ff_out', args=(fused_mlp,), name='mlp_output', shape=(N, D_MODEL), dtype=DTYPE)

    # 第二个残差连接
    final_output = builder.add_node('call_function', torch.add, args=(x_after_attn, mlp_output), name='final_output', shape=(N, D_MODEL), dtype=DTYPE)
    
    # 图的输出
    builder.add_node('output', 'output', args=(final_output,))

    logging.info("Manual graph construction complete.")
    return builder.get_graph()

# ==============================================================================
# PART 3: MAIN EXECUTION (主执行流程)
# ==============================================================================

if __name__ == "__main__":
    # 1. 手动定义并获取计算图“蓝图”
    graph_blueprint = define_llada_block_graph()

    print("\n--- Generated Graph Blueprint ---")
    graph_blueprint.print_tabular()

    # 2. 实例化分析器，传入蓝图
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