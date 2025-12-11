import torch
import math
import triton
from typing import Optional
from itertools import product


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

def _reference_qknorm(q: torch.Tensor, k: torch.Tensor, norm_weight: torch.Tensor):
    """Reference implementation using default_qknorm approach (reshaping to 2D first)."""
    import flashinfer
    q_2d = q.contiguous().view(-1, q.shape[-1])
    k_2d = k.contiguous().view(-1, k.shape[-1])
    q_o = torch.empty_like(q_2d)
    k_o = torch.empty_like(k_2d)
    flashinfer.norm.rmsnorm(q_2d, norm_weight, out=q_o)
    flashinfer.norm.rmsnorm(k_2d, norm_weight, out=k_o)
    return q_o.view(q.shape), k_o.view(k.shape)

def _is_close(x: torch.Tensor, y: torch.Tensor, rtol=1e-2, atol=5e-3):
    return torch.allclose(x, y, rtol=rtol, atol=atol)
    
def _bench_pair(batch_size, num_kv_heads, num_qo_heads, head_dim, answer_qknorm, baseline_qknorm):
    # Generate qkv tensor and extract q and k
    qkv = torch.randn(batch_size, num_qo_heads + num_kv_heads * 2, head_dim, device=DEVICE, dtype=torch.float16)
    q = qkv[:, :num_qo_heads, :]
    k = qkv[:, num_qo_heads: num_qo_heads + num_kv_heads, :]
    norm_weight = torch.randn((head_dim,), device=DEVICE, dtype=torch.float16)
    
    # Benchmark baseline (functions now return values)
    baseline_ms = _bench_ms(lambda: baseline_qknorm(q, k, norm_weight))
    
    # Benchmark answer (functions now return values)
    answer_ms = _bench_ms(lambda: answer_qknorm(q, k, norm_weight))
    
    # Get reference result
    q_ref, k_ref = _reference_qknorm(q, k, norm_weight)
    
    # Get answer result for correctness check
    q_answer, k_answer = answer_qknorm(q, k, norm_weight)
    
    # Check correctness
    q_passed = _is_close(q_answer, q_ref, rtol=1e-2, atol=5e-3)
    k_passed = _is_close(k_answer, k_ref, rtol=1e-2, atol=5e-3)
    passed = q_passed and k_passed
    
    return {
        "batch_size": batch_size, "num_kv_heads": num_kv_heads, "num_qo_heads": num_qo_heads, "head_dim": head_dim,
        "baseline_ms": baseline_ms, "answer_ms": answer_ms,
        "close_passed": passed, 
        "rtol": 1e-2, "atol": 5e-3, "passed": passed,
    }

def _warmup_gpu(iters: int = 10):
    try:
        batch_size, num_kv_heads, num_qo_heads, head_dim = 1, 8, 64, 128
        qkv = torch.randn(batch_size, num_qo_heads + num_kv_heads * 2, head_dim, device=DEVICE, dtype=torch.float16)
        q = qkv[:, :num_qo_heads, :]
        k = qkv[:, num_qo_heads: num_qo_heads + num_kv_heads, :]
        norm_weight = torch.randn((head_dim,), device=DEVICE, dtype=torch.float16)
        import flashinfer
        for _ in range(max(1, int(iters))):
            q_o = torch.empty_like(q)
            k_o = torch.empty_like(k)
            flashinfer.norm.rmsnorm(q, norm_weight, out=q_o)
            flashinfer.norm.rmsnorm(k, norm_weight, out=k_o)
        torch.cuda.synchronize()
    except Exception:
        pass

def summarize_speedup(answer_qknorm, baseline_qknorm, print_output=False):
    # Warm up GPU to stabilize clocks and caches
    _warmup_gpu(10)
    # Configuration: num_kv_heads, num_qo_heads, head_dim, batch_sizes
    configs = {
        "num_kv_heads": [8, 32],
        "num_qo_heads": [32],
        "head_dim": [64, 128],
        "batch_size": [1, 16, 64, 128, 256],
    }
    rows = []
    for num_kv_heads in configs["num_kv_heads"]:
        for num_qo_heads in configs["num_qo_heads"]:
            for head_dim in configs["head_dim"]:
                for batch_size in configs["batch_size"]:
                    r = _bench_pair(batch_size, num_kv_heads, num_qo_heads, head_dim, answer_qknorm, baseline_qknorm)
                    rows.append(r)
    print("\n=== Answer vs Baseline: Speedup for each shape (based on median time) ===")
    speedups = []
    for r in rows:
        tm, cm = r["answer_ms"], r["baseline_ms"]
        sp = cm / tm
        speedups.append(sp)
        status = "OK" if r["close_passed"] else "FAIL"
        if print_output:
            print(
                f"batch={r['batch_size']:3d} num_kv_heads={r['num_kv_heads']:2d} num_qo_heads={r['num_qo_heads']:3d} head_dim={r['head_dim']:3d}  "
                f"baseline={cm:7.3f} ms  answer={tm:7.3f} ms  speedup={sp:5.2f}x  "
                f"[Passed: {status}  "
                f"rtol={r['rtol']:.1e} atol={r['atol']:.1e}]"
            )
    if speedups:
        arith_mean = sum(speedups) / len(speedups)
        geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        median = sorted(speedups)[len(speedups)//2]
        if print_output:
            print("\n--- Summary ---")
            print(f"Sample size: {len(speedups)}")
            print(f"Arithmetic mean speedup: {arith_mean:.3f}x")
            print(f"Geometric mean speedup: {geo_mean:.3f}x")
            print(f"Median speedup: {median:.3f}x")
    return rows, arith_mean, geo_mean, median

def run_benchmark(answer_qknorm, baseline_qknorm, print_output=False):
    rows, arith_mean, geo_mean, median = summarize_speedup(answer_qknorm, baseline_qknorm, print_output=print_output)
    return {
        "rows": rows,
        "arithmetic_mean_speedup": arith_mean,
        "geometric_mean_speedup": geo_mean,
        "median_speedup": median,
        "pass_all": all(r["close_passed"] for r in rows),
    }
