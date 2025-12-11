import torch
import math
import triton
from typing import Optional

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

def _pt_scan(X, A, B):
    # y_t = a_t * y_{t-1} + b_t * x_t
    L, D = X.shape
    y = torch.zeros(D, device=X.device, dtype=torch.float32)
    out = torch.empty(L, D, device=X.device, dtype=torch.float32)
    for t in range(L):
        y = A[t].float() * y + B[t].float() * X[t].float()
        out[t] = y
    return out.to(torch.float16)

def _cpu_scan(X, A, B):
    # CPU baseline: move to CPU, compute, move back
    X_cpu = X.cpu().float()
    A_cpu = A.cpu().float()
    B_cpu = B.cpu().float()
    result_cpu = _pt_scan(X_cpu, A_cpu, B_cpu)
    return result_cpu.to(DEVICE)

def _bench_pair(L, D, chunk, BD, answer_chunk_scan, baseline_chunk_scan=_pt_scan):
    X = torch.randn(L, D, device=DEVICE, dtype=torch.float16)
    A = torch.randn(L, D, device=DEVICE, dtype=torch.float16).abs() * 0.5
    B = torch.randn(L, D, device=DEVICE, dtype=torch.float16)
    
    # CPU baseline timing (synchronize before timing)
    torch.cuda.synchronize()
    import time
    cpu_times = []
    for _ in range(10):
        start = time.perf_counter()
        _cpu_scan(X, A, B)
        torch.cuda.synchronize()  # Wait for CPU->GPU transfer
        cpu_times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    cpu_baseline_ms = sorted(cpu_times)[len(cpu_times)//2]  # Median
    
    # GPU baseline timing (using float16)
    gpu_baseline_ms = _bench_ms(lambda: baseline_chunk_scan(X, A, B))
    
    # Answer timing uses float16 with chunk and BD parameters
    answer_ms = _bench_ms(lambda: answer_chunk_scan(X, A, B, chunk=chunk, BD=BD))
    
    # Correctness check against GPU baseline (using float16)
    ref = baseline_chunk_scan(X, A, B)
    out = answer_chunk_scan(X, A, B, chunk=chunk, BD=BD)
    passed = _is_close(out, ref, rtol=1e-2, atol=5e-3)
    
    return {
        "L": L, "D": D, "chunk": chunk, "BD": BD,
        "cpu_baseline_ms": cpu_baseline_ms,
        "gpu_baseline_ms": gpu_baseline_ms,
        "answer_ms": answer_ms,
        "baseline_ms": cpu_baseline_ms,  # Keep for compatibility
        "close_passed": passed,
        "rtol": 1e-2, "atol": 5e-3, "passed": passed,
    }

def _warmup_gpu(iters: int = 10):
    try:
        L, D = 2048, 512
        X = torch.randn(L, D, device=DEVICE, dtype=torch.float16)
        A = torch.randn(L, D, device=DEVICE, dtype=torch.float16).abs() * 0.5
        B = torch.randn(L, D, device=DEVICE, dtype=torch.float16)
        for _ in range(max(1, int(iters))):
            _ = _pt_scan(X, A, B)
        torch.cuda.synchronize()
    except Exception:
        pass

def summarize_speedup(answer_chunk_scan, baseline_chunk_scan=None, print_output=False, metadata=None):
    # baseline_chunk_scan parameter kept for compatibility
    # Scoring: 0 points = 1x GPU baseline, 100 points = 3x GPU baseline
    # Warm up GPU to stabilize clocks and caches
    _warmup_gpu(10)
    
    # Get shapes from metadata or use defaults
    if metadata is None:
        metadata = {}
    shapes = metadata.get("shapes", None)
    if shapes is None:
        L_list = metadata.get("L_list", [2048, 4096])
        D = metadata.get("D", 512)
        chunk = metadata.get("chunk", 128)
        BD = metadata.get("BD", 128)
        shapes = [(L, D, chunk, BD) for L in L_list]
    
    rows = []
    for (L, D, chunk, BD) in shapes:
        r = _bench_pair(L, D, chunk, BD, answer_chunk_scan, _pt_scan)
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
                f"L={r['L']:5d} D={r['D']:4d} chunk={r['chunk']:4d} BD={r['BD']:4d}  "
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

def run_benchmark(answer_chunk_scan, baseline_chunk_scan=None, print_output=False, metadata=None):
    # baseline_chunk_scan parameter kept for compatibility
    # Scoring: 0 points = 1x GPU baseline, 100 points = 3x GPU baseline
    rows, geo_mean_cpu, geo_mean_gpu, _ = summarize_speedup(answer_chunk_scan, baseline_chunk_scan, print_output=print_output, metadata=metadata)
    
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

