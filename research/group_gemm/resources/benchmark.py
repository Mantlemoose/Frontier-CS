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

def _pt_bmm(A, B):
    # A:[B,M,K], B:[B,K,N] -> [B,M,N]
    return torch.bmm(A.float(), B.float()).to(torch.float16)

def _bench_pair(B, M, N, K, answer_bmm, baseline_bmm=_pt_bmm):
    A = torch.randn(B, M, K, device=DEVICE, dtype=torch.float16)
    Bm = torch.randn(B, K, N, device=DEVICE, dtype=torch.float16)
    
    baseline_ms = _bench_ms(lambda: baseline_bmm(A, Bm))
    answer_ms = _bench_ms(lambda: answer_bmm(A, Bm))
    
    flops = 2.0 * B * M * N * K
    to_tflops = lambda ms: flops * 1e-12 / (ms * 1e-3) if ms is not None else None
    
    ref = baseline_bmm(A, Bm)
    out = answer_bmm(A, Bm)
    passed = _is_close(out, ref, rtol=1e-2, atol=5e-3)
    
    return {
        "B": B, "M": M, "N": N, "K": K,
        "baseline_ms": baseline_ms, "answer_ms": answer_ms,
        "baseline_tflops": to_tflops(baseline_ms),
        "answer_tflops": to_tflops(answer_ms),
        "close_passed": passed,
        "rtol": 1e-2, "atol": 5e-3, "passed": passed,
    }

def _warmup_gpu(iters: int = 10):
    try:
        B, M, K, N = 64, 64, 64, 64
        A = torch.randn(B, M, K, device=DEVICE, dtype=torch.float16)
        Bm = torch.randn(B, K, N, device=DEVICE, dtype=torch.float16)
        for _ in range(max(1, int(iters))):
            _ = torch.bmm(A, Bm)
        torch.cuda.synchronize()
    except Exception:
        pass

def summarize_speedup(answer_bmm, baseline_bmm=_pt_bmm, print_output=False, metadata=None):
    # Warm up GPU to stabilize clocks and caches
    _warmup_gpu(10)
    
    # Get shapes from metadata or use defaults
    if metadata is None:
        metadata = {}
    shapes = metadata.get("shapes", None)
    if shapes is None:
        B_list = metadata.get("B_list", [64, 256, 1024])
        M = metadata.get("M", 64)
        N = metadata.get("N", 64)
        K = metadata.get("K", 64)
        shapes = [(B, M, N, K) for B in B_list]
    
    rows = []
    for (B, M, N, K) in shapes:
        r = _bench_pair(B, M, N, K, answer_bmm, baseline_bmm)
        rows.append(r)
    
    if print_output:
        print("\n=== Answer vs Baseline: Speedup for each shape (based on median time) ===")
    
    speedups = []
    for r in rows:
        tm, cm = r["answer_ms"], r["baseline_ms"]
        if cm is None or tm is None:
            continue
        sp = cm / tm
        speedups.append(sp)
        status = "OK" if r["close_passed"] else "FAIL"
        if print_output:
            print(
                f"B={r['B']:4d} M={r['M']:4d} N={r['N']:4d} K={r['K']:4d}  "
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
    else:
        arith_mean = geo_mean = median = 0.0
    
    return rows, arith_mean, geo_mean, median

def run_benchmark(answer_bmm, baseline_bmm=_pt_bmm, print_output=False, metadata=None):
    rows, arith_mean, geo_mean, median = summarize_speedup(answer_bmm, baseline_bmm, print_output=print_output, metadata=metadata)
    return {
        "rows": rows,
        "arithmetic_mean_speedup": arith_mean,
        "geometric_mean_speedup": geo_mean,
        "median_speedup": median,
        "pass_all": all(r["close_passed"] for r in rows),
    }


