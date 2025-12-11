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

def _is_close(x: torch.Tensor, y: torch.Tensor, rtol=1e-2, atol=0.5):
    return torch.allclose(x, y, rtol=rtol, atol=atol)

def _pt_fused_linear_jsd(X, W1, B1, W2, B2):
    logits1 = (X.float() @ W1.float()) + B1.float()
    logits2 = (X.float() @ W2.float()) + B2.float()
    P = torch.softmax(logits1, dim=-1)
    Q = torch.softmax(logits2, dim=-1)
    Mmid = 0.5 * (P + Q)
    eps = 1e-12
    jsd = 0.5 * (torch.sum(P * (torch.log(P + eps) - torch.log(Mmid + eps)), dim=-1) +
                 torch.sum(Q * (torch.log(Q + eps) - torch.log(Mmid + eps)), dim=-1))
    return jsd

def _cpu_fused_linear_jsd(X, W1, B1, W2, B2):
    # CPU baseline: move to CPU, compute, move back
    X_cpu = X.cpu().float()
    W1_cpu = W1.cpu().float()
    B1_cpu = B1.cpu().float()
    W2_cpu = W2.cpu().float()
    B2_cpu = B2.cpu().float()
    logits1_cpu = (X_cpu @ W1_cpu) + B1_cpu
    logits2_cpu = (X_cpu @ W2_cpu) + B2_cpu
    P_cpu = torch.softmax(logits1_cpu, dim=-1)
    Q_cpu = torch.softmax(logits2_cpu, dim=-1)
    Mmid_cpu = 0.5 * (P_cpu + Q_cpu)
    eps = 1e-12
    jsd_cpu = 0.5 * (torch.sum(P_cpu * (torch.log(P_cpu + eps) - torch.log(Mmid_cpu + eps)), dim=-1) +
                     torch.sum(Q_cpu * (torch.log(Q_cpu + eps) - torch.log(Mmid_cpu + eps)), dim=-1))
    return jsd_cpu.to(DEVICE)

def _bench_pair(M, N, K, answer_fused_linear_jsd, baseline_fused_linear_jsd=_pt_fused_linear_jsd):
    X = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    W1 = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    B1 = torch.randn(N, device=DEVICE, dtype=torch.float32)
    W2 = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    B2 = torch.randn(N, device=DEVICE, dtype=torch.float32)
    
    # CPU baseline timing (synchronize before timing)
    torch.cuda.synchronize()
    import time
    cpu_times = []
    for _ in range(10):
        start = time.perf_counter()
        _cpu_fused_linear_jsd(X, W1, B1, W2, B2)
        torch.cuda.synchronize()  # Wait for CPU->GPU transfer
        cpu_times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    cpu_baseline_ms = sorted(cpu_times)[len(cpu_times)//2]  # Median
    
    # GPU baseline timing
    gpu_baseline_ms = _bench_ms(lambda: baseline_fused_linear_jsd(X, W1, B1, W2, B2))
    
    # Answer timing
    answer_ms = _bench_ms(lambda: answer_fused_linear_jsd(X, W1, B1, W2, B2))
    
    # Correctness check against GPU baseline
    ref = baseline_fused_linear_jsd(X, W1, B1, W2, B2)
    out = answer_fused_linear_jsd(X, W1, B1, W2, B2)
    passed = _is_close(out, ref)  # Uses default rtol=1e-2, atol=0.5
    
    # Debug output for correctness failures
    if not passed:
        print(f"\n[DEBUG] Correctness failure for M={M}, N={N}, K={K}")
        print(f"[DEBUG] Reference shape: {ref.shape}, Output shape: {out.shape}")
        print(f"[DEBUG] Reference dtype: {ref.dtype}, Output dtype: {out.dtype}")
        print(f"[DEBUG] Reference min/max/mean: {ref.min().item():.6f} / {ref.max().item():.6f} / {ref.mean().item():.6f}")
        print(f"[DEBUG] Output min/max/mean: {out.min().item():.6f} / {out.max().item():.6f} / {out.mean().item():.6f}")
        diff = torch.abs(out - ref)
        max_diff_idx = torch.argmax(diff)
        print(f"[DEBUG] Max absolute difference: {diff.max().item():.6f} at index {max_diff_idx.item()}")
        print(f"[DEBUG] Reference value at max diff: {ref[max_diff_idx].item():.6f}")
        print(f"[DEBUG] Output value at max diff: {out[max_diff_idx].item():.6f}")
        print(f"[DEBUG] Relative error at max diff: {(diff[max_diff_idx] / (torch.abs(ref[max_diff_idx]) + 1e-8)).item():.6f}")
        # Check if any values are NaN or Inf
        ref_nan = torch.isnan(ref).sum().item()
        ref_inf = torch.isinf(ref).sum().item()
        out_nan = torch.isnan(out).sum().item()
        out_inf = torch.isinf(out).sum().item()
        print(f"[DEBUG] Reference NaN count: {ref_nan}, Inf count: {ref_inf}")
        print(f"[DEBUG] Output NaN count: {out_nan}, Inf count: {out_inf}")
        # Print first few values for comparison
        print(f"[DEBUG] First 5 reference values: {ref[:5].cpu().tolist()}")
        print(f"[DEBUG] First 5 output values: {out[:5].cpu().tolist()}")
    
    return {
        "M": M, "N": N, "K": K,
        "cpu_baseline_ms": cpu_baseline_ms,
        "gpu_baseline_ms": gpu_baseline_ms,
        "answer_ms": answer_ms,
        "baseline_ms": cpu_baseline_ms,  # Keep for compatibility
        "close_passed": passed,
        "rtol": 1e-2, "atol": 0.5, "passed": passed,
    }

def _warmup_gpu(iters: int = 10):
    try:
        M, N, K = 256, 4096, 2048
        X = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        W1 = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        B1 = torch.randn(N, device=DEVICE, dtype=torch.float32)
        W2 = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        B2 = torch.randn(N, device=DEVICE, dtype=torch.float32)
        for _ in range(max(1, int(iters))):
            _ = _pt_fused_linear_jsd(X, W1, B1, W2, B2)
        torch.cuda.synchronize()
    except Exception:
        pass

def summarize_speedup(answer_fused_linear_jsd, baseline_fused_linear_jsd=None, print_output=False, metadata=None):
    # baseline_fused_linear_jsd parameter kept for compatibility
    # Scoring: 0 points = 3x CPU baseline, 100 points = 7x GPU baseline
    # Warm up GPU to stabilize clocks and caches
    _warmup_gpu(10)
    
    # Get shapes from metadata or use defaults
    if metadata is None:
        metadata = {}
    shapes = metadata.get("shapes", None)
    if shapes is None:
        M_list = metadata.get("M_list", [128, 256])
        N = metadata.get("N", 4096)
        K = metadata.get("K", 2048)
        shapes = [(M, N, K) for M in M_list]
    
    rows = []
    for (M, N, K) in shapes:
        r = _bench_pair(M, N, K, answer_fused_linear_jsd, _pt_fused_linear_jsd)
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
                f"M={r['M']:4d} N={r['N']:4d} K={r['K']:4d}  "
                f"CPU={cpu_time:7.3f} ms  GPU={gpu_time:7.3f} ms  answer={answer_time:7.3f} ms  "
                f"[Passed: {status}  "
                f"rtol={r['rtol']:.1e} atol={r['atol']:.1f}]"
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

def run_benchmark(answer_fused_linear_jsd, baseline_fused_linear_jsd=None, print_output=False, metadata=None):
    # baseline_fused_linear_jsd parameter kept for compatibility
    # Scoring: 0 points = 3x CPU baseline, 100 points = 7x GPU baseline
    rows, geo_mean_cpu, geo_mean_gpu, _ = summarize_speedup(answer_fused_linear_jsd, baseline_fused_linear_jsd, print_output=print_output, metadata=metadata)
    
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

