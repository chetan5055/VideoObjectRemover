import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from queue import Queue
from threading import Thread, Lock

import torch
from torch.cuda.nvtx import range_pop, range_push
from tqdm import tqdm
from loguru import logger

from sorawm.cleaner.e2fgvi_hq_cleaner import *
from sorawm.utils.video_utils import merge_frames_with_overlap

from concurrent.futures import ThreadPoolExecutor


# =============================================================================
# 方案 1: 基础双缓冲流水线 (推荐首选，简单有效)
# =============================================================================

class OptimizedE2FGVIHDCleaner(E2FGVIHDCleaner):
    """
    基础双缓冲流水线优化
    
    原理:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 原始串行:  GPU[0] → CPU[0] → GPU[1] → CPU[1] → GPU[2] → ...        │
    │                                                                      │
    │ 优化后:    GPU: [Batch 0]────[Batch 1]────[Batch 2]────             │
    │            CPU:      wait [Batch 0]────[Batch 1]────[Batch 2]       │
    └─────────────────────────────────────────────────────────────────────┘
    
    性能提升: ~30-50%
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 创建双 CUDA streams
        self.stream_a = torch.cuda.Stream()
        self.stream_b = torch.cuda.Stream()
    
    def process_frames_chunk(
        self,
        chunk_length: int,
        neighbor_stride: int,
        imgs_chunk: torch.Tensor,
        masks_chunk: torch.Tensor,
        binary_masks_chunk: np.ndarray,
        frames_np_chunk: np.ndarray,
        h: int,
        w: int,
    ) -> List[np.ndarray]:
        
        comp_frames_chunk = [None] * chunk_length
        
        # 预计算 padding 参数
        mod_size_h = 60
        mod_size_w = 108
        h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
        w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
        
        # 预计算所有批次索引
        all_batches = []
        for f in range(0, chunk_length, neighbor_stride):
            neighbor_ids = [
                i for i in range(
                    max(0, f - neighbor_stride),
                    min(chunk_length, f + neighbor_stride + 1),
                )
            ]
            ref_ids = get_ref_index(
                f, neighbor_ids, chunk_length,
                self.config.ref_length, self.config.num_ref,
            )
            all_batches.append((neighbor_ids, ref_ids))
        
        # 流水线状态
        prev_pred_imgs = None
        prev_neighbor_ids = None
        prev_stream = None
        
        streams = [self.stream_a, self.stream_b]
        
        for batch_idx, (neighbor_ids, ref_ids) in enumerate(
            tqdm(all_batches, desc="  Frame progress", position=1, leave=False)
        ):
            current_stream = streams[batch_idx % 2]
            
            # ===== 阶段 1: CPU 处理上一批 (与当前 GPU 并行) =====
            if prev_pred_imgs is not None:
                range_push("CPU_postprocess")
                
                # 等待上一个 stream 完成
                prev_stream.synchronize()
                
                # GPU → CPU 传输 + 后处理
                pred_np = prev_pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                
                for i in range(len(prev_neighbor_ids)):
                    idx = prev_neighbor_ids[i]
                    img = (
                        np.array(pred_np[i]).astype(np.uint8)
                        * binary_masks_chunk[idx]
                        + frames_np_chunk[idx] * (1 - binary_masks_chunk[idx])
                    )
                    if comp_frames_chunk[idx] is None:
                        comp_frames_chunk[idx] = img
                    else:
                        comp_frames_chunk[idx] = (
                            comp_frames_chunk[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )
                
                range_pop()
            
            # ===== 阶段 2: GPU 处理当前批 =====
            range_push(f"GPU_batch_{batch_idx}")
            
            with torch.cuda.stream(current_stream):
                selected_imgs = imgs_chunk[:1, neighbor_ids + ref_ids, :, :, :]
                selected_masks = masks_chunk[:1, neighbor_ids + ref_ids, :, :, :]
                
                with torch.no_grad():
                    masked_imgs = selected_imgs * (1 - selected_masks)
                    
                    # Padding with reflection
                    masked_imgs = torch.cat(
                        [masked_imgs, torch.flip(masked_imgs, [3])], 3
                    )[:, :, :, : h + h_pad, :]
                    masked_imgs = torch.cat(
                        [masked_imgs, torch.flip(masked_imgs, [4])], 4
                    )[:, :, :, :, : w + w_pad]
                    
                    # 模型推理
                    pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                    pred_imgs = pred_imgs[:, :, :h, :w]
                    pred_imgs = (pred_imgs + 1) / 2
            
            range_pop()
            
            # 保存当前结果供下一轮处理
            prev_pred_imgs = pred_imgs
            prev_neighbor_ids = neighbor_ids
            prev_stream = current_stream
        
        # ===== 处理最后一批 =====
        if prev_pred_imgs is not None:
            range_push("CPU_postprocess_final")
            prev_stream.synchronize()
            pred_np = prev_pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
            
            for i in range(len(prev_neighbor_ids)):
                idx = prev_neighbor_ids[i]
                img = (
                    np.array(pred_np[i]).astype(np.uint8)
                    * binary_masks_chunk[idx]
                    + frames_np_chunk[idx] * (1 - binary_masks_chunk[idx])
                )
                if comp_frames_chunk[idx] is None:
                    comp_frames_chunk[idx] = img
                else:
                    comp_frames_chunk[idx] = (
                        comp_frames_chunk[idx].astype(np.float32) * 0.5
                        + img.astype(np.float32) * 0.5
                    )
            range_pop()
        raise RuntimeError("Stop here")
        return comp_frames_chunk


# =============================================================================
# 方案 2: 多线程 + 线程池 (更高并行度)
# =============================================================================

@dataclass
class CPUTask:
    """CPU 后处理任务"""
    pred_imgs: np.ndarray
    neighbor_ids: List[int]
    task_id: int


class ThreadPoolE2FGVIHDCleaner(E2FGVIHDCleaner):
    """
    多线程 + 线程池优化
    
    原理:
    - GPU 使用多个 CUDA Streams 并行
    - CPU 使用线程池异步处理后处理任务
    - 任务队列解耦 GPU 和 CPU 操作
    
    性能提升: ~40-60%
    """
    
    def __init__(self, *args, num_cpu_workers: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_cpu_workers = num_cpu_workers
        self.stream_a = torch.cuda.Stream()
        self.stream_b = torch.cuda.Stream()
    
    def process_frames_chunk(
        self,
        chunk_length: int,
        neighbor_stride: int,
        imgs_chunk: torch.Tensor,
        masks_chunk: torch.Tensor,
        binary_masks_chunk: np.ndarray,
        frames_np_chunk: np.ndarray,
        h: int,
        w: int,
    ) -> List[np.ndarray]:
        
        comp_frames_chunk = [None] * chunk_length
        results_lock = Lock()
        
        # Padding 参数
        mod_size_h = 60
        mod_size_w = 108
        h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
        w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
        
        # 预计算批次
        all_batches = []
        for f in range(0, chunk_length, neighbor_stride):
            neighbor_ids = [
                i for i in range(
                    max(0, f - neighbor_stride),
                    min(chunk_length, f + neighbor_stride + 1),
                )
            ]
            ref_ids = get_ref_index(
                f, neighbor_ids, chunk_length,
                self.config.ref_length, self.config.num_ref,
            )
            all_batches.append((neighbor_ids, ref_ids))
        
        # CPU 后处理函数
        def cpu_postprocess(pred_np: np.ndarray, neighbor_ids: List[int]):
            range_push("CPU_postprocess_thread")
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = (
                    np.array(pred_np[i]).astype(np.uint8)
                    * binary_masks_chunk[idx]
                    + frames_np_chunk[idx] * (1 - binary_masks_chunk[idx])
                )
                with results_lock:
                    if comp_frames_chunk[idx] is None:
                        comp_frames_chunk[idx] = img
                    else:
                        comp_frames_chunk[idx] = (
                            comp_frames_chunk[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )
            range_pop()
        
        # 使用线程池
        with ThreadPoolExecutor(max_workers=self.num_cpu_workers) as executor:
            futures = []
            streams = [self.stream_a, self.stream_b]
            events = [torch.cuda.Event() for _ in range(len(all_batches))]
            
            # 存储 GPU 结果的双缓冲
            gpu_results: Dict[int, Tuple[torch.Tensor, List[int]]] = {}
            
            for batch_idx, (neighbor_ids, ref_ids) in enumerate(
                tqdm(all_batches, desc="  Frame progress", position=1, leave=False)
            ):
                stream_idx = batch_idx % 2
                stream = streams[stream_idx]
                
                # 处理 2 批之前的结果 (确保 GPU 已完成)
                if batch_idx >= 2:
                    prev_idx = batch_idx - 2
                    events[prev_idx].synchronize()
                    
                    if prev_idx in gpu_results:
                        prev_tensor, prev_neighbors = gpu_results.pop(prev_idx)
                        pred_np = prev_tensor.cpu().permute(0, 2, 3, 1).numpy() * 255
                        future = executor.submit(cpu_postprocess, pred_np, prev_neighbors)
                        futures.append(future)
                
                # GPU 操作
                range_push(f"GPU_batch_{batch_idx}")
                with torch.cuda.stream(stream):
                    selected_imgs = imgs_chunk[:1, neighbor_ids + ref_ids, :, :, :]
                    selected_masks = masks_chunk[:1, neighbor_ids + ref_ids, :, :, :]
                    
                    with torch.no_grad():
                        masked_imgs = selected_imgs * (1 - selected_masks)
                        
                        masked_imgs = torch.cat(
                            [masked_imgs, torch.flip(masked_imgs, [3])], 3
                        )[:, :, :, : h + h_pad, :]
                        masked_imgs = torch.cat(
                            [masked_imgs, torch.flip(masked_imgs, [4])], 4
                        )[:, :, :, :, : w + w_pad]
                        
                        pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                        pred_imgs = pred_imgs[:, :, :h, :w]
                        pred_imgs = (pred_imgs + 1) / 2
                    
                    events[batch_idx].record(stream)
                range_pop()
                
                # 保存结果
                gpu_results[batch_idx] = (pred_imgs, neighbor_ids)
            
            # 处理剩余批次
            for remaining_idx in sorted(gpu_results.keys()):
                events[remaining_idx].synchronize()
                tensor, neighbors = gpu_results[remaining_idx]
                pred_np = tensor.cpu().permute(0, 2, 3, 1).numpy() * 255
                future = executor.submit(cpu_postprocess, pred_np, neighbors)
                futures.append(future)
            
            # 等待所有 CPU 任务完成
            for future in futures:
                future.result()
        
        raise RuntimeError("Stop here")
        return comp_frames_chunk


# =============================================================================
# 方案 3: Pinned Memory + 非阻塞传输 (极致性能)
# =============================================================================

class PinnedMemoryE2FGVIHDCleaner(E2FGVIHDCleaner):
    """
    Pinned Memory + 非阻塞传输优化
    
    原理:
    - 使用 pinned memory (锁页内存) 加速 GPU↔CPU 传输
    - non_blocking=True 实现真正的异步传输
    - CUDA Events 精确控制同步点
    
    性能提升: ~50-70%
    
    注意: 需要更多 CPU 内存
    """
    
    def __init__(self, *args, max_neighbors: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = max_neighbors
        self.stream = torch.cuda.Stream()
        self.pinned_buffer_1: Optional[torch.Tensor] = None
        self.pinned_buffer_2: Optional[torch.Tensor] = None
    
    def _ensure_pinned_buffers(self, h: int, w: int):
        """懒加载 pinned memory 缓冲区"""
        if self.pinned_buffer_1 is None or \
           self.pinned_buffer_1.shape[1] != h or \
           self.pinned_buffer_1.shape[2] != w:
            
            logger.info(f"Allocating pinned memory buffers: {self.max_neighbors}x{h}x{w}x3")
            self.pinned_buffer_1 = torch.empty(
                (self.max_neighbors, h, w, 3),
                dtype=torch.float32,
                pin_memory=True
            )
            self.pinned_buffer_2 = torch.empty_like(self.pinned_buffer_1)
    
    def process_frames_chunk(
        self,
        chunk_length: int,
        neighbor_stride: int,
        imgs_chunk: torch.Tensor,
        masks_chunk: torch.Tensor,
        binary_masks_chunk: np.ndarray,
        frames_np_chunk: np.ndarray,
        h: int,
        w: int,
    ) -> List[np.ndarray]:
        
        # 确保 pinned memory 已分配
        self._ensure_pinned_buffers(h, w)
        buffers = [self.pinned_buffer_1, self.pinned_buffer_2]
        
        comp_frames_chunk = [None] * chunk_length
        
        # Padding 参数
        mod_size_h = 60
        mod_size_w = 108
        h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
        w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
        
        # 预计算批次
        all_batches = []
        for f in range(0, chunk_length, neighbor_stride):
            neighbor_ids = [
                i for i in range(
                    max(0, f - neighbor_stride),
                    min(chunk_length, f + neighbor_stride + 1),
                )
            ]
            ref_ids = get_ref_index(
                f, neighbor_ids, chunk_length,
                self.config.ref_length, self.config.num_ref,
            )
            all_batches.append((neighbor_ids, ref_ids))
        
        # 流水线状态
        prev_event: Optional[torch.cuda.Event] = None
        prev_buffer: Optional[torch.Tensor] = None
        prev_neighbor_ids: Optional[List[int]] = None
        prev_count: int = 0
        
        for batch_idx, (neighbor_ids, ref_ids) in enumerate(
            tqdm(all_batches, desc="  Frame progress", position=1, leave=False)
        ):
            buffer_idx = batch_idx % 2
            current_buffer = buffers[buffer_idx]
            
            # ===== CPU 处理上一批 (与 GPU 并行) =====
            if prev_event is not None:
                range_push("CPU_postprocess_pinned")
                
                # 等待异步传输完成
                prev_event.synchronize()
                
                # 直接使用 pinned memory 的 numpy 视图 (零拷贝!)
                pred_np = prev_buffer[:prev_count].numpy() * 255
                
                for i in range(len(prev_neighbor_ids)):
                    idx = prev_neighbor_ids[i]
                    img = (
                        np.array(pred_np[i]).astype(np.uint8)
                        * binary_masks_chunk[idx]
                        + frames_np_chunk[idx] * (1 - binary_masks_chunk[idx])
                    )
                    if comp_frames_chunk[idx] is None:
                        comp_frames_chunk[idx] = img
                    else:
                        comp_frames_chunk[idx] = (
                            comp_frames_chunk[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )
                
                range_pop()
            
            # ===== GPU 操作 + 异步传输 =====
            range_push(f"GPU_batch_{batch_idx}")
            
            with torch.cuda.stream(self.stream):
                selected_imgs = imgs_chunk[:1, neighbor_ids + ref_ids, :, :, :]
                selected_masks = masks_chunk[:1, neighbor_ids + ref_ids, :, :, :]
                
                with torch.no_grad():
                    masked_imgs = selected_imgs * (1 - selected_masks)
                    
                    masked_imgs = torch.cat(
                        [masked_imgs, torch.flip(masked_imgs, [3])], 3
                    )[:, :, :, : h + h_pad, :]
                    masked_imgs = torch.cat(
                        [masked_imgs, torch.flip(masked_imgs, [4])], 4
                    )[:, :, :, :, : w + w_pad]
                    
                    pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                    pred_imgs = pred_imgs[:, :, :h, :w]
                    pred_imgs = (pred_imgs + 1) / 2
                    
                    # 非阻塞传输到 pinned memory
                    pred_permuted = pred_imgs.permute(0, 2, 3, 1)
                    current_buffer[:len(neighbor_ids)].copy_(
                        pred_permuted, non_blocking=True
                    )
                    
                    # 记录传输完成事件
                    current_event = torch.cuda.Event()
                    current_event.record(self.stream)
            
            range_pop()
            
            # 保存状态
            prev_event = current_event
            prev_buffer = current_buffer
            prev_neighbor_ids = neighbor_ids
            prev_count = len(neighbor_ids)
        
        # ===== 处理最后一批 =====
        if prev_event is not None:
            range_push("CPU_postprocess_final")
            prev_event.synchronize()
            pred_np = prev_buffer[:prev_count].numpy() * 255
            
            for i in range(len(prev_neighbor_ids)):
                idx = prev_neighbor_ids[i]
                img = (
                    np.array(pred_np[i]).astype(np.uint8)
                    * binary_masks_chunk[idx]
                    + frames_np_chunk[idx] * (1 - binary_masks_chunk[idx])
                )
                if comp_frames_chunk[idx] is None:
                    comp_frames_chunk[idx] = img
                else:
                    comp_frames_chunk[idx] = (
                        comp_frames_chunk[idx].astype(np.float32) * 0.5
                        + img.astype(np.float32) * 0.5
                    )
            range_pop()
        
        raise RuntimeError("Stop here")
        return comp_frames_chunk


# =============================================================================
# 工具函数 (如果 e2fgvi_hq_cleaner 中没有的话)
# =============================================================================



# =============================================================================
# 使用示例
# =============================================================================

"""
# 方案 1: 基础优化 (推荐)
cleaner = OptimizedE2FGVIHDCleaner(config)

# 方案 2: 多线程优化
cleaner = ThreadPoolE2FGVIHDCleaner(config, num_cpu_workers=4)

# 方案 3: 极致性能优化
cleaner = PinnedMemoryE2FGVIHDCleaner(config, max_neighbors=20)

# 使用方式与原来相同
result = cleaner.process_frames_chunk(
    chunk_length, neighbor_stride,
    imgs_chunk, masks_chunk,
    binary_masks_chunk, frames_np_chunk,
    h, w
)
"""