"""
在线 Chunk 配置搜索器（Bottleneck-Oriented）

根据实际的 (P, O) 实时搜索最优 chunk 配置
使用贪心策略：找到瓶颈 → 砍瓶颈 → 检查能否推 → 重复
"""

import logging
from typing import Dict, Any, Optional
import time
logger = logging.getLogger(__name__)


def analyze_components_from_timeline(timeline: list) -> dict:
    """
    从 MemoryAnalyzer 的 timeline 中分析各部分的峰值显存
    
    Args:
        timeline: MemoryAnalyzer.analyze()['memory_timeline']
    
    Returns:
        {
            'fixed_peak_mb': xxx,
            'mlp_peak_mb': xxx,
            'logits_peak_mb': xxx,
            'total_peak_mb': xxx
        }
    """
    fixed_peak_mb = 0
    mlp_peak_mb = 0
    logits_peak_mb = 0
    total_peak_mb = 0
    
    for step in timeline:
        node_name = step['node_name']
        total_mb = step['total_memory_mb']
        total_peak_mb = max(total_peak_mb, total_mb)
        
        # 判断节点属于哪个部分
        if 'logits' in node_name:
            logits_peak_mb = max(logits_peak_mb, total_mb)
        elif any(kw in node_name for kw in ['mlp', 'moe', 'cache', 'x_gate', 'x_up', 'x_mlp', 'barrier']):
            # MLP/MoE 相关节点
            mlp_peak_mb = max(mlp_peak_mb, total_mb)
        else:
            # 固定部分（Attention, RMSNorm 等）
            fixed_peak_mb = max(fixed_peak_mb, total_mb)
    
    return {
        'fixed_peak_mb': fixed_peak_mb,
        'mlp_peak_mb': mlp_peak_mb,
        'logits_peak_mb': logits_peak_mb,
        'total_peak_mb': total_peak_mb
    }


def search_optimal_config_online(
    P: int,
    O: int,
    pool_size: int,
    model_name: str = 'llada'
) -> Dict[str, Any]:
    """
    在线搜索最优 chunk 配置（Bottleneck-Oriented）
    
    算法：
    1. 从 no_chunk 开始
    2. 循环：
       a. 生成图并分析 → 计算 total_bytes 和各部分峰值
       b. 检查 total_bytes <= pool_size？
          - 能推 → 返回当前配置
          - 不能 → 找瓶颈（fixed/mlp/logits 中最大的）
       c. 砍瓶颈（增加对应的 chunks）
       d. 重复直到能推或无法再优化
    
    Args:
        P: Prompt token 数量
        O: Output token 数量
        pool_size: Activation pool 大小（字节）
        model_name: 模型名称 (llada, dream, llada-moe)
    
    Returns:
        {
            'chunk_logits': bool,
            'chunk_mlp': bool,
            'num_chunks_logits': int,
            'num_chunks_mlp': int,
            'total_bytes': int,
            'components': {...}  # 最终配置的内存组成
        }
    
    Raises:
        RuntimeError: 当 fixed 部分是瓶颈且无法通过 chunk 优化时
    """
 
    search_start_time = time.time()
    

    safety_margin_mb = 200   
   
    pool_size -= safety_margin_mb * 1024 * 1024
    
    from vllm.v1.worker.graph_builder import MemoryAnalyzer
    
    # 根据 model_name 选择对应的 graph builder
    if 'moe' in model_name.lower():
        from vllm.v1.worker.graph_builder import (
            define_llada_moe_block_graph,
            define_llada_moe_block_graph_chunkwise
        )
        define_graph = define_llada_moe_block_graph
        define_graph_chunkwise = define_llada_moe_block_graph_chunkwise
    elif 'dream' in model_name.lower():
        from vllm.v1.worker.graph_builder import (
            define_dream_block_graph,
            define_dream_block_graph_chunkwise
        )
        define_graph = define_dream_block_graph
        define_graph_chunkwise = define_dream_block_graph_chunkwise
    else:  # llada
        from vllm.v1.worker.graph_builder import (
            define_llada_block_graph,
            define_llada_block_graph_chunkwise
        )
        define_graph = define_llada_block_graph
        define_graph_chunkwise = define_llada_block_graph_chunkwise
    
    # 初始配置：不 chunk
    config = {
        'chunk_logits': False,
        'chunk_mlp': False,
        'num_chunks_logits': 1,
        'num_chunks_mlp': 1
    }
    
    iteration = 0
    
    logger.info(f"[OnlineSearcher] 开始搜索: P={P}, O={O}, pool={pool_size/1024**3:.2f}GB")
    
    while True:
        iteration += 1
        
        # 1. 生成图并分析
        if not config['chunk_logits'] and not config['chunk_mlp']:
            # no_chunk
            graph = define_graph(P, O)
        else:
            # chunkwise
            # 根据模型类型使用正确的参数名
            if 'moe' in model_name.lower():
                # LLaDA-MoE 使用 chunk_moe 参数
                graph = define_graph_chunkwise(
                    P, O,
                    chunk_logits=config['chunk_logits'],
                    chunk_moe=config['chunk_mlp'],  # 注意：config 中使用 chunk_mlp，但函数参数是 chunk_moe
                    num_chunks_logits=config['num_chunks_logits'],
                    num_chunks_moe=config['num_chunks_mlp']  # 注意：config 中使用 num_chunks_mlp，但函数参数是 num_chunks_moe
                )
            else:
                # Dream 和 LLaDA 使用 chunk_mlp 参数
                graph = define_graph_chunkwise(
                    P, O,
                    chunk_logits=config['chunk_logits'],
                    chunk_mlp=config['chunk_mlp'],
                    num_chunks_logits=config['num_chunks_logits'],
                    num_chunks_mlp=config['num_chunks_mlp']
                )
        
        analyzer = MemoryAnalyzer(graph)
        analysis = analyzer.analyze()
        total_bytes = analysis['reuse_plan']['_total_buffer_size_bytes']
        
        # 分析各部分峰值
        components = analyze_components_from_timeline(analysis['memory_timeline'])
        fixed_mb = components['fixed_peak_mb']
        mlp_mb = components['mlp_peak_mb']
        logits_mb = components['logits_peak_mb']
        
        logger.info(
            f"[OnlineSearcher] Iter {iteration}: "
            f"logits_chunks={config['num_chunks_logits']}, "
            f"mlp_chunks={config['num_chunks_mlp']}, "
            f"total={total_bytes/1024**3:.3f}GB, "
            f"fixed={fixed_mb:.2f}MB, mlp={mlp_mb:.2f}MB, logits={logits_mb:.2f}MB"
        )
        
        # 2. 检查能否推
        if total_bytes <= pool_size:
            # MoE 特判：余量太小时拒绝，避免边界 OOM
            if 'moe' in model_name.lower():
                margin_bytes = pool_size - total_bytes
                min_margin_mb = 700  # MoE 最小安全余量
                if margin_bytes < min_margin_mb * 1024 * 1024:
                    logger.info(
                        f"[OnlineSearcher]   ⚠️ MoE 特判: "
                        f"pool {pool_size/(1024**3):.3f}GB - need {total_bytes/(1024**3):.3f}GB "
                        f"= 余量 {margin_bytes/(1024**2):.0f}MB < 安全余量 {min_margin_mb}MB，"
                        f"拒绝此配置，继续搜索"
                    )
                    # 继续下一轮搜索（不 return，继续循环）
                else:
                    # MoE 余量足够，接受配置
                    search_end_time = time.time()
                    search_time = search_end_time - search_start_time
                    logger.info(
                        f"[OnlineSearcher] ✅ 找到可行配置 (MoE): "
                        f"chunk_logits={config['chunk_logits']}({config['num_chunks_logits']}), "
                        f"chunk_mlp={config['chunk_mlp']}({config['num_chunks_mlp']}), "
                        f"pool {pool_size/(1024**3):.3f}GB - need {total_bytes/(1024**3):.3f}GB "
                        f"= 余量 {margin_bytes/(1024**2):.0f}MB >= 安全余量 {min_margin_mb}MB, "
                        f"搜索耗时 {search_time*1000:.2f}ms"
                    )
                    config['total_bytes'] = total_bytes
                    config['components'] = components
                    return config
            else:
                # 非 MoE，接受配置
                search_end_time = time.time()
                search_time = search_end_time - search_start_time
                logger.info(
                    f"[OnlineSearcher] ✅ 找到可行配置: "
                    f"chunk_logits={config['chunk_logits']}({config['num_chunks_logits']}), "
                    f"chunk_mlp={config['chunk_mlp']}({config['num_chunks_mlp']}), "
                    f"需要 {total_bytes/1024**3:.3f}GB <= {pool_size/1024**3:.3f}GB, "
                    f"搜索耗时 {search_time*1000:.2f}ms"
                )
                config['total_bytes'] = total_bytes
                config['components'] = components
                return config
        
        # 3. 找瓶颈
        bottleneck = max(fixed_mb, mlp_mb, logits_mb)
        
        # 4. 砍瓶颈
        if bottleneck == logits_mb and logits_mb > fixed_mb:
            # Logits 是瓶颈 → 增加 logits chunks
            if not config['chunk_logits']:
                config['chunk_logits'] = True
                config['num_chunks_logits'] = 2
                logger.info(f"[OnlineSearcher]   → 砍 Logits 瓶颈: 启用 chunk_logits (chunks=2)")
            else:
                config['num_chunks_logits'] += 1
                logger.info(f"[OnlineSearcher]   → 砍 Logits 瓶颈: chunks={config['num_chunks_logits']}")
        
        elif bottleneck == mlp_mb and mlp_mb > fixed_mb:
            # MLP 是瓶颈 → 增加 mlp chunks
            if not config['chunk_mlp']:
                config['chunk_mlp'] = True
                config['num_chunks_mlp'] = 2
                logger.info(f"[OnlineSearcher]   → 砍 MLP 瓶颈: 启用 chunk_mlp (chunks=2)")
            else:
                config['num_chunks_mlp'] += 1
                logger.info(f"[OnlineSearcher]   → 砍 MLP 瓶颈: chunks={config['num_chunks_mlp']}")
        
        else:
            # Fixed 是瓶颈 → 无法优化
            search_end_time = time.time()
            search_time = search_end_time - search_start_time
            logger.error(
                f"[OnlineSearcher] ❌ Fixed 部分是瓶颈 ({fixed_mb:.2f} MB)，无法通过 chunk 优化，"
                f"搜索耗时 {search_time*1000:.2f}ms"
            )
            raise RuntimeError(
                f"CUDA out of memory: Fixed 部分是瓶颈 ({fixed_mb:.2f} MB)，无法通过 chunk 优化。\n"
                f"请求: P={P}, O={O}, 需要 {total_bytes/1024**3:.3f} GB，"
                f"但 pool 只有 {pool_size/1024**3:.3f} GB。\n"
                f"内存组成: fixed={fixed_mb:.2f}MB, mlp={mlp_mb:.2f}MB, logits={logits_mb:.2f}MB\n"
                f"建议：增加 activation pool 大小或减少请求长度。"
            )

