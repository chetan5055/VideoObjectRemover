"""
简化版性能测试脚本
只运行一个 chunk 来快速对比性能
"""

from pathlib import Path
from time import perf_counter
from dataclasses import dataclass
from typing import List, Optional
import gc

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from sorawm.core import SoraWM
from sorawm.cleaner.e2fgvi_hq_cleaner import E2FGVIHDCleaner, get_ref_index
from sorawm.schemas import CleanerType
from tqdm import tqdm

console = Console()


class StopAfterOneChunk(Exception):
    """用于在一个 chunk 完成后停止"""
    def __init__(self, elapsed_time: float, result: List[np.ndarray]):
        self.elapsed_time = elapsed_time
        self.result = result
        super().__init__("Benchmark completed for one chunk")


# =============================================================================
# 基准测试版本的 Cleaners
# =============================================================================

class BaselineBenchmarkCleaner(E2FGVIHDCleaner):
    """原始版本 - 用于基准测试"""
    
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
        
        torch.cuda.synchronize()
        start_time = perf_counter()
        
        # 原始实现
        comp_frames_chunk = [None] * chunk_length
        for f in tqdm(
            range(0, chunk_length, neighbor_stride),
            desc=f"  Frame progress",
            position=1,
            leave=False,
        ):
            neighbor_ids = [
                i
                for i in range(
                    max(0, f - neighbor_stride),
                    min(chunk_length, f + neighbor_stride + 1),
                )
            ]
            ref_ids = get_ref_index(
                f,
                neighbor_ids,
                chunk_length,
                self.config.ref_length,
                self.config.num_ref,
            )
            selected_imgs = imgs_chunk[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_chunk[:1, neighbor_ids + ref_ids, :, :, :]
            with torch.no_grad():
                # GPU OPS
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[
                    :, :, :, : h + h_pad, :
                ]
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[
                    :, :, :, :, : w + w_pad
                ]
                pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                # CPU OPS
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_imgs[i]).astype(np.uint8) * binary_masks_chunk[
                        idx
                    ] + frames_np_chunk[idx] * (1 - binary_masks_chunk[idx])
                    if comp_frames_chunk[idx] is None:
                        comp_frames_chunk[idx] = img
                    else:
                        comp_frames_chunk[idx] = (
                            comp_frames_chunk[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )
        
        torch.cuda.synchronize()
        elapsed = perf_counter() - start_time
        
        # 抛出异常来停止，同时携带结果
        raise StopAfterOneChunk(elapsed, comp_frames_chunk)


class OptimizedBenchmarkCleaner(E2FGVIHDCleaner):
    """方案1: 双缓冲 + CUDA Streams"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        
        torch.cuda.synchronize()
        start_time = perf_counter()
        
        comp_frames_chunk = [None] * chunk_length
        
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
        prev_pred_imgs = None
        prev_neighbor_ids = None
        prev_stream = None
        streams = [self.stream_a, self.stream_b]
        
        for batch_idx, (neighbor_ids, ref_ids) in enumerate(
            tqdm(all_batches, desc="  Frame progress", position=1, leave=False)
        ):
            current_stream = streams[batch_idx % 2]
            
            # CPU 处理上一批
            if prev_pred_imgs is not None:
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
            
            # GPU 处理当前批
            with torch.cuda.stream(current_stream):
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
            
            prev_pred_imgs = pred_imgs
            prev_neighbor_ids = neighbor_ids
            prev_stream = current_stream
        
        # 处理最后一批
        if prev_pred_imgs is not None:
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
        
        torch.cuda.synchronize()
        elapsed = perf_counter() - start_time
        raise StopAfterOneChunk(elapsed, comp_frames_chunk)


class ThreadPoolBenchmarkCleaner(E2FGVIHDCleaner):
    """方案2: 多线程 + Streams"""
    
    def __init__(self, *args, num_cpu_workers: int = 4, **kwargs):
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
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock
        
        torch.cuda.synchronize()
        start_time = perf_counter()
        
        comp_frames_chunk = [None] * chunk_length
        results_lock = Lock()
        
        mod_size_h = 60
        mod_size_w = 108
        h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
        w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
        
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
        
        def cpu_postprocess(pred_np: np.ndarray, neighbor_ids: List[int]):
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
        
        with ThreadPoolExecutor(max_workers=self.num_cpu_workers) as executor:
            futures = []
            streams = [self.stream_a, self.stream_b]
            events = [torch.cuda.Event() for _ in range(len(all_batches))]
            gpu_results = {}
            
            for batch_idx, (neighbor_ids, ref_ids) in enumerate(
                tqdm(all_batches, desc="  Frame progress", position=1, leave=False)
            ):
                stream_idx = batch_idx % 2
                stream = streams[stream_idx]
                
                if batch_idx >= 2:
                    prev_idx = batch_idx - 2
                    events[prev_idx].synchronize()
                    
                    if prev_idx in gpu_results:
                        prev_tensor, prev_neighbors = gpu_results.pop(prev_idx)
                        pred_np = prev_tensor.cpu().permute(0, 2, 3, 1).numpy() * 255
                        future = executor.submit(cpu_postprocess, pred_np, prev_neighbors)
                        futures.append(future)
                
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
                
                gpu_results[batch_idx] = (pred_imgs, neighbor_ids)
            
            for remaining_idx in sorted(gpu_results.keys()):
                events[remaining_idx].synchronize()
                tensor, neighbors = gpu_results[remaining_idx]
                pred_np = tensor.cpu().permute(0, 2, 3, 1).numpy() * 255
                future = executor.submit(cpu_postprocess, pred_np, neighbors)
                futures.append(future)
            
            for future in futures:
                future.result()
        
        torch.cuda.synchronize()
        elapsed = perf_counter() - start_time
        raise StopAfterOneChunk(elapsed, comp_frames_chunk)


class PinnedMemoryBenchmarkCleaner(E2FGVIHDCleaner):
    """方案3: Pinned Memory + 异步传输"""
    
    def __init__(self, *args, max_neighbors: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_neighbors = max_neighbors
        self.stream = torch.cuda.Stream()
        self.pinned_buffer_1: Optional[torch.Tensor] = None
        self.pinned_buffer_2: Optional[torch.Tensor] = None
    
    def _ensure_pinned_buffers(self, h: int, w: int):
        if self.pinned_buffer_1 is None:
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
        
        torch.cuda.synchronize()
        start_time = perf_counter()
        
        self._ensure_pinned_buffers(h, w)
        buffers = [self.pinned_buffer_1, self.pinned_buffer_2]
        
        comp_frames_chunk = [None] * chunk_length
        
        mod_size_h = 60
        mod_size_w = 108
        h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
        w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
        
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
        
        prev_event = None
        prev_buffer = None
        prev_neighbor_ids = None
        prev_count = 0
        
        for batch_idx, (neighbor_ids, ref_ids) in enumerate(
            tqdm(all_batches, desc="  Frame progress", position=1, leave=False)
        ):
            buffer_idx = batch_idx % 2
            current_buffer = buffers[buffer_idx]
            
            # CPU 处理上一批
            if prev_event is not None:
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
            
            # GPU 操作
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
                    
                    # 非阻塞传输
                    pred_permuted = pred_imgs.permute(0, 2, 3, 1)
                    current_buffer[:len(neighbor_ids)].copy_(pred_permuted, non_blocking=True)
                    
                    current_event = torch.cuda.Event()
                    current_event.record(self.stream)
            
            prev_event = current_event
            prev_buffer = current_buffer
            prev_neighbor_ids = neighbor_ids
            prev_count = len(neighbor_ids)
        
        # 处理最后一批
        if prev_event is not None:
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
        
        torch.cuda.synchronize()
        elapsed = perf_counter() - start_time
        raise StopAfterOneChunk(elapsed, comp_frames_chunk)


# =============================================================================
# 测试框架
# =============================================================================

@dataclass
class Result:
    name: str
    time: float
    gpu_mb: float
    success: bool = True


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def benchmark(name: str, sora_wm: SoraWM, cleaner, input_path: Path, output_path: Path) -> Result:
    """运行单个基准测试"""
    console.print(f"[cyan]▶ 测试: {name}[/cyan]")
    
    clear_gpu()
    
    sora_wm.cleaner = cleaner
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    try:
        sora_wm.run(input_path, output_path)
        # 正常完成不应该到这里
        console.print(f"[yellow]  ⚠ 未能捕获到计时[/yellow]")
        return Result(name, 0, 0, False)
        
    except StopAfterOneChunk as e:
        # 成功捕获计时
        gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        console.print(f"[green]  ✓ {e.elapsed_time:.2f}s, GPU: {gpu_mb:.0f}MB[/green]")
        return Result(name, e.elapsed_time, gpu_mb, True)
        
    except Exception as e:
        console.print(f"[red]  ✗ 错误: {e}[/red]")
        import traceback
        traceback.print_exc()
        return Result(name, 0, 0, False)


def print_results(results: List[Result], baseline_time: float):
    """打印结果表格"""
    
    table = Table(
        title="🚀 性能基准测试结果 (单 Chunk)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    
    table.add_column("方案", style="cyan", width=38)
    table.add_column("耗时", justify="right", width=12)
    table.add_column("加速比", justify="right", width=10)
    table.add_column("节省", justify="right", width=12)
    table.add_column("GPU峰值", justify="right", width=12)
    table.add_column("状态", justify="center", width=6)
    
    for r in results:
        if not r.success:
            table.add_row(r.name, "-", "-", "-", "-", "[red]✗[/red]")
            continue
        
        if baseline_time > 0:
            speedup = baseline_time / r.time if r.time > 0 else 0
            saved = baseline_time - r.time
        else:
            speedup = 1.0
            saved = 0
        
        # 加速比颜色
        if speedup >= 1.5:
            sp_str = f"[bold green]{speedup:.2f}x[/bold green]"
        elif speedup >= 1.2:
            sp_str = f"[green]{speedup:.2f}x[/green]"
        elif speedup >= 1.0:
            sp_str = f"[yellow]{speedup:.2f}x[/yellow]"
        else:
            sp_str = f"[red]{speedup:.2f}x[/red]"
        
        # 节省时间颜色
        if saved > 0:
            saved_str = f"[green]-{saved:.2f}s[/green]"
        elif saved < 0:
            saved_str = f"[red]+{abs(saved):.2f}s[/red]"
        else:
            saved_str = "-"
        
        table.add_row(
            r.name,
            f"{r.time:.2f}s",
            sp_str,
            saved_str,
            f"{r.gpu_mb:.0f}MB",
            "[green]✓[/green]",
        )
    
    console.print()
    console.print(table)
    
    # 找最佳
    successful = [r for r in results if r.success]
    if len(successful) >= 2 and baseline_time > 0:
        best = min(successful, key=lambda x: x.time)
        improvement = (baseline_time - best.time) / baseline_time * 100
        
        console.print()
        console.print(Panel(
            f"🏆 最佳方案: [bold green]{best.name}[/bold green]\n"
            f"⏱️  耗时: [cyan]{best.time:.2f}s[/cyan]\n"
            f"📈 提升: [bold yellow]{improvement:.1f}%[/bold yellow]\n"
            f"💾 节省: [green]{baseline_time - best.time:.2f}s[/green]",
            title="📊 结论",
            border_style="green",
        ))


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    input_video_path = Path("resources/dog_vs_sam.mp4")
    output_dir = Path("outputs/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        f"[bold]E2FGVI HD Cleaner 性能测试[/bold]\n"
        f"输入: {input_video_path}\n"
        f"[dim]仅测试单个 chunk 的处理时间[/dim]",
        title="🧪 Benchmark",
        border_style="cyan",
    ))
    
    # 初始化
    console.print("\n[yellow]初始化 SoraWM...[/yellow]")
    sora_wm = SoraWM(cleaner_type=CleanerType.E2FGVI_HQ)
    
    results = []
    
    # ========== 测试 1: 原始版本 (基准) ==========
    r = benchmark(
        "① Original (Baseline)",
        sora_wm,
        BaselineBenchmarkCleaner(),
        input_video_path,
        output_dir / "baseline.mp4",
    )
    results.append(r)
    baseline_time = r.time if r.success else 0
    
    # ========== 测试 2: 双缓冲优化 ==========
    r = benchmark(
        "② Optimized (双缓冲 + CUDA Streams)",
        sora_wm,
        OptimizedBenchmarkCleaner(),
        input_video_path,
        output_dir / "optimized.mp4",
    )
    results.append(r)
    
    # ========== 测试 3: 线程池优化 ==========
    r = benchmark(
        "③ ThreadPool (多线程 + Streams)",
        sora_wm,
        ThreadPoolBenchmarkCleaner(num_cpu_workers=4),
        input_video_path,
        output_dir / "threadpool.mp4",
    )
    results.append(r)
    
    # ========== 测试 4: Pinned Memory 优化 ==========
    r = benchmark(
        "④ PinnedMemory (锁页内存 + 异步传输)",
        sora_wm,
        PinnedMemoryBenchmarkCleaner(max_neighbors=20),
        input_video_path,
        output_dir / "pinned.mp4",
    )
    results.append(r)
    
    # 输出结果
    print_results(results, baseline_time)