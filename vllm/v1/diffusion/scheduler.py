# vllm/v1/diffusion/scheduler.py
from __future__ import annotations
from typing import Dict, List, Optional
from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import create_request_queue, SchedulingPolicy
from vllm.v1.diffusion.types import (
    DiffusionRequest,
    DiffusionBatch,
    DiffusionRuntimeState,
    DiffusionStepRequest,
    DiffusionStepBatch,
    DiffusionStepUpdate,
    FinishedDiffusionItem,
)

logger = init_logger(__name__)

class DiffusionScheduler:
    """扩散模式下的最简调度器（迭代级）。

    waiting：尚未进入运行的请求队列（FCFS）。
    running：正在迭代的请求（req_id -> 运行时状态）。
    finished_req_ids：本 step 完成的请求 id 集合（供 EngineCore 回收）。
    """
    def __init__(self, max_batch_size: int = 1):
        self.max_batch_size = max_batch_size
        self._model_runner = None  # 将在设置时由EngineCore传入
        # 直接复用 vLLM 的请求队列实现，后续可切换为 priority。
        self.waiting = create_request_queue(SchedulingPolicy.FCFS)
        # 所有请求的运行时状态：req_id -> state
        self.states: Dict[str, DiffusionRuntimeState] = {}
        # 正在执行迭代的请求 id 列表（保持 FCFS 公平性）
        self.running: List[str] = []
        self.finished_req_ids: set[str] = set()

    def set_model_runner(self, model_runner) -> None:
        """设置model runner引用，用于获取activation memory pool信息"""
        self._model_runner = model_runner

    def _can_add_more_requests(self) -> bool:
        """检查是否还能添加更多请求到running队列"""
        if self._model_runner is None or not hasattr(self._model_runner, 'activation_memory_pool'):
            logger.info(f"[DEBUG_CALL_CHAIN] _can_add_more_requests被调用: self._model_runner is None or not hasattr(self._model_runner, 'activation_memory_pool')")
            return len(self.running) < self.max_batch_size
            
        activation_pool = self._model_runner.activation_memory_pool
        if not activation_pool._initialized:
            logger.info(f"\n\n\n[DEBUG_CALL_CHAIN] activation_pool._initialized is False\n\n\n")
            return len(self.running) < self.max_batch_size

        # 检查waiting队列中下一个请求的token数量
        if not bool(self.waiting):
            return False
            
        # NOTE: 注释掉显存准入控制，因为 P/O 分离后无法提前准确估算显存占用
        # 现在只检查 batch_size 限制，如果实际 forward 时 OOM 会直接报错
        # 这适用于单请求测试场景，多请求场景需要重新设计准入控制逻辑
        
        # # 获取显存池支持的最大token数量
        # max_tokens_from_pool = activation_pool.calculate_max_num_tokens()
        
        # # 计算当前running请求的总token数量
        # current_tokens = sum(self.states[req_id].total_len for req_id in self.running if req_id in self.states)
        
        # # 获取waiting队列中的下一个请求（不pop，只是peek）
        # next_req = None
        
        # if bool(self.waiting):
        #     next_req = self.waiting[0] if hasattr(self.waiting[0], 'input_token_ids') else None
        #     logger.info(f"\n\n\n[DEBUG_CALL_CHAIN] waiting_queue[0] = {self.waiting[0]}\n\n\n")
        
        # if next_req is None:
        #     logger.info(f"\n\n\n[DEBUG_CALL_CHAIN] next_req is None\n\n\n")
        #     return len(self.running) < self.max_batch_size
            
        # next_req_tokens = len(next_req.input_token_ids) + next_req.params.gen_length
        # logger.info(f'len(self.running) = {len(self.running)}')
        # logger.info(f'max_tokens_from_pool = {max_tokens_from_pool}')
        # logger.info(f'current_tokens = {current_tokens}')
        # logger.info(f'next_req_tokens = {next_req_tokens}')
        
        # # 检测单个请求是否超过显存池总容量（防止死循环）
        # if next_req_tokens > max_tokens_from_pool:
        #     raise RuntimeError(
        #         f"CUDA out of memory: 请求 {next_req.req_id} 需要 {next_req_tokens} tokens "
        #         f"({next_req_tokens * activation_pool.memory_per_token / 1024**2:.2f} MB), "
        #         f"超过显存池总容量 {max_tokens_from_pool} tokens "
        #         f"({max_tokens_from_pool * activation_pool.memory_per_token / 1024**2:.2f} MB). "
        #         f"请求参数: input_length={len(next_req.input_token_ids)}, "
        #         f"gen_length={next_req.params.gen_length}"
        #     )
        
        # # 检查加入下一个请求后是否超过显存限制
        # would_exceed_memory = (current_tokens + next_req_tokens) > max_tokens_from_pool
        # would_exceed_batch_size = len(self.running) >= self.max_batch_size
        
        # can_add = not would_exceed_memory and not would_exceed_batch_size
        
        # logger.debug(f"[DiffusionScheduler] 检查是否能添加请求: "
        #             f"current_tokens={current_tokens}, "
        #             f"next_req_tokens={next_req_tokens}, "
        #             f"max_tokens={max_tokens_from_pool}, "
        #             f"can_add={can_add}")
        
        # 只检查 batch size 限制，显存占用在实际 forward 时检查
        return len(self.running) < self.max_batch_size

    def add_request(self, req: DiffusionRequest) -> None:
        """将新请求加入 waiting 队列，并初始化运行时状态。"""
        # 初始化 x：prompt + 生成区 mask_id
        prompt_len = len(req.input_token_ids)
        total_len = prompt_len + req.params.gen_length
        x = list(req.input_token_ids) + [req.params.mask_id] * req.params.gen_length
        state = DiffusionRuntimeState(
            req_id=req.req_id,
            x=x,
            prompt_len=prompt_len,
            total_len=total_len,
            params=req.params,
        )
        # 保存状态，等待迁移到 running 时使用
        self.states[req.req_id] = state
        # waiting 队列保存原始请求对象以维持到达顺序
        self.waiting.add_request(req)  # type: ignore[arg-type]

    def _ensure_running_capacity(self) -> None:
        """将 waiting 中的请求迁移到 running，直到达到显存或batch容量限制。"""
        while self._can_add_more_requests():
            req = self.waiting.pop_request()
            # 确保状态存在
            if req.req_id not in self.states:
                prompt_len = len(req.input_token_ids)
                total_len = prompt_len + req.params.gen_length
                x = list(req.input_token_ids) + [req.params.mask_id] * req.params.gen_length
                self.states[req.req_id] = DiffusionRuntimeState(
                    req_id=req.req_id,
                    x=x,
                    prompt_len=prompt_len,
                    total_len=total_len,
                    params=req.params,
                )
            # 加入 running 列表
            if req.req_id not in self.running:
                self.running.append(req.req_id)

    def schedule_step(self) -> Optional[DiffusionStepBatch]:
        """挑选至多 max_batch_size 个运行中的请求，组成一步批次。"""
        # 补充 running 容量
        self._ensure_running_capacity()
        # 选择可执行一步的请求
        step_reqs: List[DiffusionStepRequest] = []
        for req_id in list(self.running):
            state = self.states.get(req_id)
            if state is None:
                continue
            # 判断该请求是否已完成
            #-----------------------------修复，预检测不会触发--------------------------
            # if state.x[state.prompt_len:].count(state.params.mask_id) == 0:
            #     self.finished_req_ids.add(state.req_id)
            #     # 从 running 移除，等待被 collect
            #     try:
            #         self.running.remove(state.req_id)
            #     except ValueError:
            #         pass
            #     continue
            #-----------------------------修复，预检测不会触发--------------------------
            # 填充一步请求
            step_reqs.append(
                DiffusionStepRequest(
                    req_id=state.req_id,
                    x=list(state.x),  # 传副本，runner 可原地修改返回 new_x
                    prompt_len=state.prompt_len,
                    total_len=state.total_len,
                    params=state.params,
                    block_idx=state.block_idx,
                    step_idx_in_block=state.step_idx_in_block,
                ))
            # running队列中的请求已经通过_ensure_running_capacity进行了显存限制
            # 这里只需要确保不超过max_batch_size即可
            if len(step_reqs) >= self.max_batch_size:
                break
        if not step_reqs:
            return None

        # 打印本批每个请求的剩余步数（用于调试迭代式调度进度）
        debug_items: List[str] = []
        for r in step_reqs:
            num_blocks = max(1, r.params.gen_length // max(1, r.params.block_length))
            steps_per_block = max(1, r.params.steps // num_blocks)
            remain_in_block = max(0, steps_per_block - r.step_idx_in_block)
            remain_blocks_after = max(0, num_blocks - 1 - r.block_idx)
            total_remaining_steps = remain_in_block + remain_blocks_after * steps_per_block
            debug_items.append(
                f"{r.req_id}:remain={total_remaining_steps} (block {r.block_idx+1}/{num_blocks}, step {r.step_idx_in_block+1}/{steps_per_block}) seqlen {r.total_len}"
            )
        logger.info("[DiffusionScheduler] step_batch size=%d | %s",
                    len(step_reqs), ", ".join(debug_items))
        #------------------------------------------------------------------
        return DiffusionStepBatch(requests=step_reqs)

    def apply_updates_and_collect_finished(
        self, updates: List[DiffusionStepUpdate]
    ) -> List[FinishedDiffusionItem]:
        """合并 runner 返回的一步更新，推进运行时状态，并收集完成项。"""
        finished_items: List[FinishedDiffusionItem] = []
        for up in updates:
            st = self.states.get(up.req_id)
            if st is None:
                continue
            # 覆盖 x（首版简单：整段替换）
            st.x = list(up.new_x)
            # 推进步进度
            st.step_idx_in_block += 1
            # 计算每块步数与总块数
            num_blocks = max(1, st.params.gen_length // max(1, st.params.block_length))
            steps_per_block = max(1, st.params.steps // num_blocks)
            if st.step_idx_in_block >= steps_per_block:
                st.step_idx_in_block = 0
                st.block_idx += 1

            # 完成判定：生成区无 mask_id
            finished_by_mask = (st.x[st.prompt_len:].count(st.params.mask_id) == 0)
            finished_by_budget = (st.block_idx >= num_blocks)
            if finished_by_mask or finished_by_budget:
                tail = st.x[st.prompt_len: st.prompt_len + st.params.gen_length]
                finished_items.append(
                    FinishedDiffusionItem(req_id=st.req_id, tail_token_ids=tail))
                self.finished_req_ids.add(st.req_id)
                # 从 running 移除
                try:
                    self.running.remove(st.req_id)
                except ValueError:
                    pass

        return finished_items

    # 保留旧接口以兼容一次性路径（不再使用）
    def schedule(self) -> Optional[DiffusionBatch]:
        if not bool(self.waiting):
            return None
        batch_list = []
        while bool(self.waiting) and len(batch_list) < self.max_batch_size:
            batch_list.append(self.waiting.pop_request())
        return DiffusionBatch(requests=batch_list)

    def has_active_requests(self) -> bool:
        """是否仍有需要处理的请求（waiting 或 running）。"""
        return bool(self.waiting) or len(self.running) > 0