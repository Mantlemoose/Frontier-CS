import torch
import math
import triton
from typing import Optional
import torch.nn.functional as F

# Ensure CUDA is available and properly initialize device
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)

def alloc_fn(size: int, align: int, stream: Optional[int]):
    assert align == 128
    assert stream == 0
    return torch.empty(size, dtype=torch.int8, device=DEVICE)

triton.set_allocator(alloc_fn)
torch.manual_seed(0)
try:
    torch.cuda.manual_seed_all(0)
except Exception:
    pass
assert triton.runtime.driver.active.get_current_target().backend == "cuda", "This benchmark only supports CUDA backend."

def _bench_ms(fn):
    out = triton.testing.do_bench(fn, quantiles=[0.5])
    if isinstance(out, (tuple, list)):
        return float(out[0])
    return float(out)

def _is_close(x: torch.Tensor, y: torch.Tensor, rtol=1e-2, atol=5e-3):
    return torch.allclose(x, y, rtol=rtol, atol=atol)

def _pt_linear_gelu(X, W, B):
    return F.gelu((X @ W) + B).to(torch.float16)

def _cpu_linear_gelu(X, W, B):
    # CPU baseline: move to CPU, compute, move back
    X_cpu = X.cpu().float()
    W_cpu = W.cpu().float()
    B_cpu = B.cpu().float()
    result_cpu = _pt_linear_gelu(X_cpu, W_cpu, B_cpu)
    return result_cpu.to(DEVICE)

def _bench_pair(M, N, K, answer_linear_gelu, baseline_linear_gelu=_pt_linear_gelu):
    X = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    W = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    B = torch.randn(N, device=DEVICE, dtype=torch.float32)
    X32 = X.float()
    W32 = W.float()
    B32 = B.float()
    
    # CPU baseline timing (synchronize before timing)
    torch.cuda.synchronize()
    import time
    cpu_times = []
    for _ in range(10):
        start = time.perf_counter()
        _cpu_linear_gelu(X, W, B)
        torch.cuda.synchronize()  # Wait for CPU->GPU transfer
        cpu_times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    cpu_baseline_ms = sorted(cpu_times)[len(cpu_times)//2]  # Median
    
    # GPU baseline timing (using float32 for baseline)
    gpu_baseline_ms = _bench_ms(lambda: baseline_linear_gelu(X32, W32, B32))
    
    # Answer timing uses float16
    answer_ms = _bench_ms(lambda: answer_linear_gelu(X, W, B))
    
    # Correctness check against GPU baseline (using float32 for reference)
    ref = baseline_linear_gelu(X32, W32, B32)
    out = answer_linear_gelu(X, W, B)
    passed = _is_close(out, ref, rtol=1e-2, atol=5e-3)
    
    flops = 2.0 * M * N * K
    to_tflops = lambda ms: flops * 1e-12 / (ms * 1e-3) if ms is not None else None
    
    return {
        "M": M, "N": N, "K": K,
        "cpu_baseline_ms": cpu_baseline_ms,
        "gpu_baseline_ms": gpu_baseline_ms,
        "answer_ms": answer_ms,
        "baseline_ms": cpu_baseline_ms,  # Keep for compatibility
        "baseline_tflops": to_tflops(gpu_baseline_ms),
        "answer_tflops": to_tflops(answer_ms),
        "close_passed": passed,
        "rtol": 1e-2, "atol": 5e-3, "passed": passed,
    }

def _warmup_gpu(iters: int = 10):
    try:
        M, N, K = 512, 4096, 4096
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        W = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        B = torch.randn(N, device=DEVICE, dtype=torch.float32)
        for _ in range(max(1, int(iters))):
            _ = _pt_linear_gelu(X.float(), W.float(), B.float())
        torch.cuda.synchronize()
    except Exception:
        pass

def summarize_speedup(answer_linear_gelu, baseline_linear_gelu=None, print_output=False, metadata=None):
    # baseline_linear_gelu parameter kept for compatibility
    # Scoring: 0 points = 1x GPU baseline, 100 points = 3x GPU baseline
    # Warm up GPU to stabilize clocks and caches
    _warmup_gpu(10)
    
    # Get shapes from metadata or use defaults
    if metadata is None:
        metadata = {}
    shapes = metadata.get("shapes", None)
    if shapes is None:
        M_list = metadata.get("M_list", [512, 1024])
        N = metadata.get("N", 4096)
        K = metadata.get("K", 4096)
        shapes = [(M, N, K) for M in M_list]
    
    rows = []
    for (M, N, K) in shapes:
        r = _bench_pair(M, N, K, answer_linear_gelu, _pt_linear_gelu)
        rows.append(r)
    
    if print_output:
        print("\n=== Answer vs Baseline: Speedup for each shape (based on median time) ===")
    
    speedups_cpu = []
    speedups_gpu = []
    for r in rows:
        answer_time = r["answer_ms"]
        cpu_time = r.get("cpu_baseline_ms")
        gpu_time = r.get("gpu_baseline_ms")
        
        if cpu_time is not None and answer_time is not None:
            sp_cpu = cpu_time / answer_time
            speedups_cpu.append(sp_cpu)
        
        if gpu_time is not None and answer_time is not None:
            sp_gpu = gpu_time / answer_time
            speedups_gpu.append(sp_gpu)
        
        status = "OK" if r["close_passed"] else "FAIL"
        if print_output:
            print(
                f"M={r['M']:5d} N={r['N']:5d} K={r['K']:5d}  "
                f"CPU={cpu_time:7.3f} ms  GPU={gpu_time:7.3f} ms  answer={answer_time:7.3f} ms  "
                f"[Passed: {status}  "
                f"rtol={r['rtol']:.1e} atol={r['atol']:.1e}]"
            )
    
    if speedups_cpu:
        geo_mean_cpu = math.exp(sum(math.log(s) for s in speedups_cpu) / len(speedups_cpu))
    else:
        geo_mean_cpu = 0.0
    
    if speedups_gpu:
        geo_mean_gpu = math.exp(sum(math.log(s) for s in speedups_gpu) / len(speedups_gpu))
    else:
        geo_mean_gpu = 0.0
    
    if print_output:
        print("\n--- Summary ---")
        print(f"Geometric mean speedup vs CPU: {geo_mean_cpu:.3f}x")
        print(f"Geometric mean speedup vs GPU: {geo_mean_gpu:.3f}x")
    
    return rows, geo_mean_cpu, geo_mean_gpu, geo_mean_gpu  # Last param kept for compatibility

def run_benchmark(answer_linear_gelu, baseline_linear_gelu=None, print_output=False, metadata=None):
    # baseline_linear_gelu parameter kept for compatibility
    # Scoring: 0 points = 1x GPU baseline, 100 points = 3x GPU baseline
    rows, geo_mean_cpu, geo_mean_gpu, _ = summarize_speedup(answer_linear_gelu, baseline_linear_gelu, print_output=print_output, metadata=metadata)
    
    # Compute geometric mean CPU and GPU baseline times
    cpu_times = [r["cpu_baseline_ms"] for r in rows if r.get("cpu_baseline_ms") is not None]
    gpu_times = [r["gpu_baseline_ms"] for r in rows if r.get("gpu_baseline_ms") is not None]
    answer_times = [r["answer_ms"] for r in rows if r.get("answer_ms") is not None]
    
    geo_mean_cpu_time = math.exp(sum(math.log(t) for t in cpu_times) / len(cpu_times)) if cpu_times else 0.0
    geo_mean_gpu_time = math.exp(sum(math.log(t) for t in gpu_times) / len(gpu_times)) if gpu_times else 0.0
    geo_mean_answer_time = math.exp(sum(math.log(t) for t in answer_times) / len(answer_times)) if answer_times else 0.0
    
    return {
        "rows": rows,
        "geometric_mean_speedup_cpu": geo_mean_cpu,
        "geometric_mean_speedup_gpu": geo_mean_gpu,
        "geometric_mean_speedup": geo_mean_gpu,  # Keep for compatibility
        "arithmetic_mean_speedup": geo_mean_gpu,  # Keep for compatibility
        "median_speedup": geo_mean_gpu,  # Keep for compatibility
        "geo_mean_cpu_time": geo_mean_cpu_time,
        "geo_mean_gpu_time": geo_mean_gpu_time,
        "geo_mean_answer_time": geo_mean_answer_time,
        "pass_all": all(r["close_passed"] for r in rows),
    }

