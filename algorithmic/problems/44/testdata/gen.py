# Generate 20 test cases total: keep original 10 base cases, add 10 "valuable" variants focusing on methods 3/5/8/9.
#   - New variants cover different parameters (start points, step sizes, radii, golden angle rotation, spiral params, shuffle seeds, sine wave cycles/amplitude/range/parity flip) to ensure diverse boundary and shape coverage.
# - Each test data generates a corresponding empty .ans file (e.g. 1.ans, 2.ans, ..., 20.ans).
# - Maintain sorted point set output (consistent with original script) to ensure reproducibility.
# - All new variant parameters are within limits and use clamp to ensure no overflow.

import math
import random

LIMIT = 10**9

def clamp(v, lo=-LIMIT, hi=LIMIT):
    return max(lo, min(hi, v))

def dump_case(fname, pts):
    with open(fname, "w") as f:
        f.write(str(len(pts)) + "\n")
        for x, y in pts:
            f.write(f"{x} {y}\n")

def shuffle_list(lst, inplace=True, seed=None):
    """
    Shuffle a list using Fisher-Yates algorithm.

    Parameters:
    - lst: the input list to shuffle. Must be a sequence (list or tuple). If not a list and inplace=True, a TypeError is raised.
    - inplace: if True, shuffle the list in-place and return None. If False, return a new shuffled list leaving the input unchanged.
    - seed: optional seed for a reproducible shuffle. If None, uses system randomness.

    Raises:
    - TypeError if lst is not a list-like sequence when appropriate.

    Returns:
    - None when inplace=True (the list is modified), or a new list when inplace=False.
    """
    # Basic type checks
    if inplace:
        if not isinstance(lst, list):
            raise TypeError("shuffle_list: inplace=True requires a list as input")
        rng = random.Random(seed) if seed is not None else random
        # Fisher-Yates in-place
        n = len(lst)
        for i in range(n - 1, 0, -1):
            j = rng.randrange(i + 1)
            lst[i], lst[j] = lst[j], lst[i]
        return None
    else:
        # Return a new shuffled list
        if isinstance(lst, (list, tuple)):
            out = list(lst)
            rng = random.Random(seed) if seed is not None else random
            n = len(out)
            for i in range(n - 1, 0, -1):
                j = rng.randrange(i + 1)
                out[i], out[j] = out[j], out[i]
            return out
        else:
            raise TypeError("shuffle_list: expected list or tuple when inplace=False")

# ----------------------
# Original 10 generators
# ----------------------

# Case 1: N=10, small ring + one far point, angle scramble
def gen_case1():
    N = 10
    pts = [(0, 0)] * N
    R = 900_000_000
    ring_n = 8
    order = [ (k * 3) % ring_n for k in range(ring_n) ]  # angle-scrambled
    for i in range(1, 1 + ring_n):
        j = order[i - 1]
        theta = 2.0 * math.pi * j / ring_n
        x = int(round(R * math.cos(theta)))
        y = int(round(R * math.sin(theta)))
        pts[i] = (clamp(x), clamp(y))
    pts[9] = (900_000_000, 900_000_000)
    return pts

# Case 2: N=97, 3 far-separated clusters, IDs interleaved across clusters
def gen_case2():
    N = 97
    rng = random.Random(2)
    pts = [(0, 0)] * N
    centers = [(-700_000_000, -700_000_000),
               ( 700_000_000, -700_000_000),
               ( 0,            700_000_000)]
    spread = 2_000_000
    for i in range(1, N):
        c = (i - 1) % 3
        cx, cy = centers[c]
        dx = rng.randrange(-spread, spread + 1)
        dy = rng.randrange(-spread, spread + 1)
        pts[i] = (clamp(cx + dx), clamp(cy + dy))
    return pts

# Case 3: N=250, two long parallel lines far apart, IDs alternate lines
def gen_case3():
    N = 250
    pts = [(0, 0)] * N
    yA = -700_000_000
    yB =  700_000_000
    start_x = -600_000_000
    step_x = 10_000_000
    cnt = N + 1
    cntA = (cnt + 1) // 2  # even IDs
    cntB = cnt // 2        # odd IDs
    xsA = [start_x + k * step_x for k in range(cntA)]
    xsB = [start_x + k * step_x for k in range(cntB)]
    ia = 0
    ib = 0
    for i in range(1, N):
        if i % 2 == 0:
            x = xsA[ia]; ia += 1
            pts[i] = (clamp(x), yA)
        else:
            x = xsB[ib]; ib += 1
            pts[i] = (clamp(x), yB)
    return pts

# Case 4: N=1000, grid with LCG scrambling of indices
def gen_case4():
    N = 1000
    pts = [(0, 0)] * N
    W, H = 40, 25  # 40*25 = 1000
    assert W * H == N
    a, b, m = 137, 61, N  # LCG parameters; a coprime with m
    s = 50_000_000
    rng = random.Random(4)
    jitter = 2_000_000
    for i in range(N):
        p = (i * a + b) % m
        r = p // W
        c = p % W
        x = (c - (W // 2)) * s
        y = (r - (H // 2)) * s
        # small jitter to break perfect grid symmetry
        x += rng.randrange(-jitter, jitter + 1)
        y += rng.randrange(-jitter, jitter + 1)
        pts[i] = (clamp(x), clamp(y))
    return pts

# Case 5: N=5000, two concentric circles interleaved (golden-angle stepping)
def gen_case5():
    N = 5000
    pts = [(0, 0)] * N
    R_in = 200_000_000
    R_out = 900_000_000
    phi = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    kin = 0
    kout = 0
    for i in range(1, N):
        if i % 2 == 0:
            # outer ring
            kout += 1
            theta = 2.0 * math.pi * (kout * phi % 1.0)
            x = int(round(R_out * math.cos(theta)))
            y = int(round(R_out * math.sin(theta)))
        else:
            # inner ring
            kin += 1
            theta = 2.0 * math.pi * (kin * phi % 1.0)
            x = int(round(R_in * math.cos(theta)))
            y = int(round(R_in * math.sin(theta)))
        pts[i] = (clamp(x), clamp(y))
    return pts

# Case 6: N=15000, Lissajous (co-prime frequencies) with co-prime wrap
def gen_case6():
    N = 15000
    pts = [(0, 0)] * N
    A = 900_000_000
    B = 900_000_000
    a = 123
    b = 97
    step = 7919  # co-prime with N (15000), ensures wrap
    for i in range(N):
        t = ((i * step) % N) / N
        x = int(round(A * math.sin(2.0 * math.pi * a * t)))
        y = int(round(B * math.sin(2.0 * math.pi * b * t)))
        pts[i] = (clamp(x), clamp(y))
    return pts

# Case 7: N=40000, hierarchical clusters: 10 superclusters × 4 subclusters
def gen_case7():
    N = 40000
    pts = [(0, 0)] * N
    rng = random.Random(7)
    S = 10  # superclusters
    R = 800_000_000
    super_centers = []
    for j in range(S):
        th = 2.0 * math.pi * j / S
        cx = int(round(R * math.cos(th)))
        cy = int(round(R * math.sin(th)))
        super_centers.append((cx, cy))
    # 4 subclusters per supercluster with small offset
    K = 4
    sub_offsets = []
    for j in range(S):
        offs = []
        for _ in range(K):
            ox = rng.randrange(-30_000_000, 30_000_000 + 1)
            oy = rng.randrange(-30_000_000, 30_000_000 + 1)
            offs.append((ox, oy))
        sub_offsets.append(offs)
    jitter = 1_000_000
    for i in range(1, N):
        k = i - 1
        sc = k % S
        g = k // S
        sub = g % K
        cx, cy = super_centers[sc]
        ox, oy = sub_offsets[sc][sub]
        dx = rng.randrange(-jitter, jitter + 1)
        dy = rng.randrange(-jitter, jitter + 1)
        x = cx + ox + dx
        y = cy + oy + dy
        pts[i] = (clamp(x), clamp(y))
    return pts

# Case 8: N=80000, Archimedean spiral, theta indices scrambled
def gen_case8():
    N = 80000
    pts = [(0, 0)] * N
    a = 10_000_000
    b = 30_000_000
    theta_max = 29.0  # radians
    step = 7  # co-prime with N
    for i in range(N):
        j = (i * step) % N
        theta = theta_max * (j / (N - 1 if N > 1 else 1))
        r = a + b * theta
        x = int(round(r * math.cos(theta)))
        y = int(round(r * math.sin(theta)))
        pts[i] = (clamp(x), clamp(y))
    shuffle_list(pts)
    return pts

# Case 9: N=120000, two opposite-phase sine waves; IDs alternate waves
def gen_case9():
    N = 120000
    pts = [(0, 0)] * N
    rng = random.Random(9)
    total = N - 1
    cntA = (total + 1) // 2  # even IDs
    cntB = total // 2        # odd IDs
    x_min = -900_000_000
    x_max =  900_000_000
    if cntA > 1:
        stepA = (x_max - x_min) / (cntA - 1)
    else:
        stepA = 0
    if cntB > 1:
        stepB = (x_max - x_min) / (cntB - 1)
    else:
        stepB = 0
    amp = 700_000_000
    cycles = 5.0
    w = 2.0 * math.pi * cycles / (x_max - x_min)
    ia = 0
    ib = 0
    for i in range(1, N):
        if i % 2 == 0:
            x = x_min + ia * stepA
            y = amp * math.sin(w * x)
            ia += 1
        else:
            x = x_min + ib * stepB
            y = -amp * math.sin(w * x)  # opposite phase
            ib += 1
        xi = int(round(x))
        yi = int(round(y))
        pts[i] = (clamp(xi), clamp(yi))
    return pts

# Case 10: N=200000, two interleaved offset grids, odd IDs reversed order
def gen_case10():
    N = 200000
    pts = [(0, 0)] * N
    # Two grids (layers), each 400 x 250 = 100000
    W, H = 400, 250
    count_layer = W * H  # 100000
    assert count_layer * 2 == N
    s = 4_000_000  # even, to allow (s//2) integer offset
    half = s // 2
    def idx_to_xy(k, offset):
        # k in [0, count_layer)
        r = k // W
        c = k % W
        x = (c - (W // 2)) * s + offset
        y = (r - (H // 2)) * s + offset
        return (clamp(x), clamp(y))
    for i in range(N):
        if i % 2 == 0:
            # layer A, forward order
            k = i // 2
            pts[i] = idx_to_xy(k, 0)
        else:
            # layer B, reverse order to induce large hops
            k = (count_layer - 1) - (i // 2)
            pts[i] = idx_to_xy(k, half)
    return pts

# ----------------------
# Valuable variants (3/5/8/9) for more coverage
# ----------------------

def gen_case3_variant(N=250, yA=-700_000_000, yB=700_000_000, start_x=-600_000_000, step_x=10_000_000):
    pts = [(0, 0)] * N
    cnt = N + 1
    cntA = (cnt + 1) // 2
    cntB = cnt // 2
    xsA = [start_x + k * step_x for k in range(cntA)]
    xsB = [start_x + k * step_x for k in range(cntB)]
    ia = 0
    ib = 0
    for i in range(1, N):
        if i % 2 == 0:
            x = xsA[ia]; ia += 1
            pts[i] = (clamp(x), clamp(yA))
        else:
            x = xsB[ib]; ib += 1
            pts[i] = (clamp(x), clamp(yB))
    return pts

def gen_case5_variant(N=5000, R_in=200_000_000, R_out=900_000_000, phi=None, base_rot=0.0):
    # base_rot: add a global rotation offset to diversify angular positions
    if phi is None:
        phi = (math.sqrt(5.0) - 1.0) / 2.0
    pts = [(0, 0)] * N
    kin = 0
    kout = 0
    for i in range(1, N):
        if i % 2 == 0:
            kout += 1
            theta = base_rot + 2.0 * math.pi * (kout * phi % 1.0)
            x = int(round(R_out * math.cos(theta)))
            y = int(round(R_out * math.sin(theta)))
        else:
            kin += 1
            theta = base_rot + 2.0 * math.pi * (kin * phi % 1.0)
            x = int(round(R_in * math.cos(theta)))
            y = int(round(R_in * math.sin(theta)))
        pts[i] = (clamp(x), clamp(y))
    return pts

def gen_case8_variant(N=80000, a=10_000_000, b=30_000_000, theta_max=29.0, step=7, shuffle_seed=None):
    # Archimedean spiral with controllable scramble and deterministic shuffle
    pts = [(0, 0)] * N
    for i in range(N):
        j = (i * step) % N
        theta = theta_max * (j / (N - 1 if N > 1 else 1))
        r = a + b * theta
        x = int(round(r * math.cos(theta)))
        y = int(round(r * math.sin(theta)))
        pts[i] = (clamp(x), clamp(y))
    shuffle_list(pts, inplace=True, seed=shuffle_seed)
    return pts

def gen_case9_variant(N=120000, cycles=5.0, amp=700_000_000, x_min=-900_000_000, x_max=900_000_000, flip=False):
    # flip=False matches original even/odd assignment; True swaps the phase assignment
    pts = [(0, 0)] * N
    total = N - 1
    cntA = (total + 1) // 2
    cntB = total // 2
    stepA = (x_max - x_min) / (cntA - 1) if cntA > 1 else 0
    stepB = (x_max - x_min) / (cntB - 1) if cntB > 1 else 0
    w = 2.0 * math.pi * cycles / (x_max - x_min) if (x_max - x_min) != 0 else 0.0
    ia = 0
    ib = 0
    for i in range(1, N):
        if (i % 2 == 0) ^ flip:
            x = x_min + ia * stepA
            y = amp * math.sin(w * x)
            ia += 1
        else:
            x = x_min + ib * stepB
            y = -amp * math.sin(w * x)
            ib += 1
        xi = int(round(x))
        yi = int(round(y))
        pts[i] = (clamp(xi), clamp(yi))
    return pts

def main():
    # Base 10 cases
    cases = [
        gen_case1,   # 1: N=10
        gen_case2,   # 2: N=97
        gen_case3,   # 3: N=250
        gen_case4,   # 4: N=1000
        gen_case5,   # 5: N=5000
        gen_case6,   # 6: N=15000
        gen_case7,   # 7: N=40000
        gen_case8,   # 8: N=80000
        gen_case9,   # 9: N=120000
        gen_case10,  # 10: N=200000
    ]
    # 10 additional valuable variants focusing on 3/5/8/9
    cases += [
        # 11-12: Variants of Case 3 (parallel lines)
        lambda: gen_case3_variant(start_x=-650_000_000, step_x=8_000_000),
        lambda: gen_case3_variant(start_x=-500_000_000, step_x=12_000_000, yA=-800_000_000, yB=800_000_000),
        # 13-14: Variants of Case 5 (concentric circles)
        lambda: gen_case5_variant(R_in=250_000_000, R_out=850_000_000, base_rot=0.30),
        lambda: gen_case5_variant(R_in=300_000_000, R_out=900_000_000, phi=(math.sqrt(2.0) - 1.0), base_rot=1.10),
        # 15-17: Variants of Case 8 (spirals with different scramble/params)
        lambda: gen_case8_variant(theta_max=31.5, step=11, shuffle_seed=15),
        lambda: gen_case8_variant(a=12_000_000, b=28_000_000, theta_max=27.5, step=13, shuffle_seed=16),
        lambda: gen_case8_variant(b=35_000_000, theta_max=33.0, step=17, shuffle_seed=17),
        # 18-20: Variants of Case 9 (sine waves with different cycles/amp/range/flip)
        lambda: gen_case9_variant(cycles=3.5, amp=800_000_000),
        lambda: gen_case9_variant(cycles=7.0, amp=650_000_000, flip=True),
        lambda: gen_case9_variant(cycles=6.0, amp=700_000_000, x_min=-800_000_000, x_max=800_000_000),
    ]

    # Generate 20 cases and empty .ans files
    for idx, fn in enumerate(cases, start=1):
        pts = fn()
        pts.sort()
        in_name = f"{idx}.in"
        dump_case(in_name, pts)
        # create empty .ans
        with open(f"{idx}.ans", "w"):
            pass
        print(f"Wrote {in_name} with N={len(pts)} and empty {idx}.ans")

if __name__ == "__main__":
    main()