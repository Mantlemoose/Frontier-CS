import sys
import random
import math
from typing import List, Tuple

Edge = Tuple[int, int]

def gen_path(n: int) -> List[Edge]:
    return [(i - 1, i) for i in range(2, n + 1)]

def gen_star(n: int, center: int = 1) -> List[Edge]:
    return [(center, i) for i in range(1, n + 1) if i != center]

def gen_balanced_binary(n: int) -> List[Edge]:
    return [(i // 2, i) for i in range(2, n + 1)]

def gen_prufer_random(n: int, rng: random.Random) -> List[Edge]:
    if n == 1:
        return []
    code = [rng.randint(1, n) for _ in range(n - 2)]
    deg = [1] * (n + 1)
    for x in code:
        deg[x] += 1
    import heapq
    pq = [i for i in range(1, n + 1) if deg[i] == 1]
    heapq.heapify(pq)
    edges = []
    for x in code:
        u = heapq.heappop(pq)
        edges.append((u, x))
        deg[u] -= 1
        deg[x] -= 1
        if deg[x] == 1:
            heapq.heappush(pq, x)
    u = heapq.heappop(pq)
    v = heapq.heappop(pq)
    edges.append((u, v))
    return edges

def gen_caterpillar(n: int, L: int, rng: random.Random) -> List[Edge]:
    L = max(2, min(L, n))
    edges = []
    for i in range(2, L + 1):
        edges.append((i - 1, i))
    for x in range(L + 1, n + 1):
        spine = rng.randint(1, L)
        edges.append((spine, x))
    return edges

def gen_broom(n: int, L: int) -> List[Edge]:
    L = max(2, min(L, n))
    edges = []
    for i in range(2, L + 1):
        edges.append((i - 1, i))
    for x in range(L + 1, n + 1):
        edges.append((L, x))
    return edges

def gen_two_star_bridge(n: int) -> List[Edge]:
    if n < 3:
        return gen_path(n)
    s1 = (n - 2) // 2
    s2 = n - 2 - s1
    edges = [(1, 2)]
    cur = 3
    for _ in range(s1):
        edges.append((1, cur)); cur += 1
    for _ in range(s2):
        edges.append((2, cur)); cur += 1
    return edges

def gen_preferential_attachment(n: int, rng: random.Random) -> List[Edge]:
    if n <= 1:
        return []
    edges = [(1, 2)]
    stubs = [1, 2]
    for i in range(3, n + 1):
        parent = rng.choice(stubs)
        edges.append((parent, i))
        stubs.append(parent)
        stubs.append(i)
    return edges

def gen_prufer_shuffled(n: int, rng: random.Random) -> List[Edge]:
    edges = gen_prufer_random(n, rng)
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    mp = lambda x: perm[x - 1]
    return [(mp(u), mp(v)) for (u, v) in edges]

def gen_chain_of_stars(n: int, block_size: int) -> List[Edge]:
    block_size = max(2, block_size)
    edges = []
    prev_center = None
    cur = 1
    while cur <= n:
        end = min(n, cur + block_size - 1)
        center = cur
        for v in range(cur + 1, end + 1):
            edges.append((center, v))
        if prev_center is not None:
            edges.append((prev_center, center))
        prev_center = center
        cur = end + 1
    return edges

def assign_weights(edges: List[Edge], rng: random.Random) -> List[Tuple[int, int, int]]:
    return [(u, v, rng.randint(1, 10000)) for (u, v) in edges]

def minimal_sum_distinct(k: int, start: int) -> int:
    # sum of start + (start+1) + ... + (start+k-1)
    return k * start + k * (k - 1) // 2

def gen_increasing_sizes(k: int, budget: int, rng: random.Random, start_min: int = 2) -> List[int]:
    # Generate k strictly increasing positive integers >= start_min with total sum <= budget.
    # Greedy random with feasibility guarantee.
    sizes = []
    prev = start_min - 1
    rem = budget
    for i in range(1, k + 1):
        remain_cnt = k - i + 1
        # Choose current t in [low, high], where high ensures feasibility for the rest
        low = prev + 1
        # Minimal sum for the rest if we pick current as t: remain_cnt * t + (remain_cnt-1)*remain_cnt/2 <= rem
        # => t <= (rem - (remain_cnt-1)*remain_cnt/2) // remain_cnt
        high = (rem - (remain_cnt - 1) * remain_cnt // 2) // remain_cnt
        if high < low:
            # Fallback to determinism in the (unlikely) event of rounding issues
            t = low
        else:
            t = rng.randint(low, high)
        sizes.append(t)
        rem -= t
        prev = t
    return sizes

def pick_structure(n: int, is_big: bool, rng: random.Random) -> List[Edge]:
    if is_big:
        # For large scale, prefer Prufer random trees, occasionally shuffle labels
        if rng.random() < 0.7:
            return gen_prufer_random(n, rng)
        else:
            return gen_prufer_shuffled(n, rng)
    # Mix various structures for other scales
    choice = rng.random()
    if n <= 3:
        return gen_path(n)
    if choice < 0.12:
        return gen_path(n)
    elif choice < 0.24:
        center = rng.randint(1, n)
        return gen_star(n, center=center)
    elif choice < 0.36:
        return gen_balanced_binary(n)
    elif choice < 0.48:
        L = max(2, int(round(math.sqrt(n) * (0.6 + 0.8 * rng.random()))))
        return gen_caterpillar(n, L, rng)
    elif choice < 0.60:
        L = max(2, min(n, int(n * (0.4 + 0.4 * rng.random()))))
        return gen_broom(n, L)
    elif choice < 0.72:
        return gen_two_star_bridge(n)
    elif choice < 0.84:
        return gen_preferential_attachment(n, rng)
    elif choice < 0.92:
        return gen_chain_of_stars(n, block_size=rng.randint(3, 40))
    else:
        return gen_prufer_shuffled(n, rng)

def main():
    # 1) Read test point index i (from command line args or interactive input)
    if len(sys.argv) >= 2:
        try:
            tp_idx = int(sys.argv[1])
        except:
            print("Invalid test point index in argv[1].", file=sys.stderr)
            return
    else:
        try:
            s = input("Enter test point index i: ").strip()
            tp_idx = int(s)
        except:
            print("Failed to read test point index.", file=sys.stderr)
            return

    # Read maximum total node count (optional argv[2], default 100000)
    if len(sys.argv) >= 3:
        try:
            MAX_N_TOTAL = int(sys.argv[2])
        except:
            print("Invalid MAX_N_TOTAL in argv[2], fallback to 100000.", file=sys.stderr)
            MAX_N_TOTAL = 100000
    else:
        MAX_N_TOTAL = 100000

    # 2) Random seed: use i to ensure reproducibility
    rng = random.Random(20250924 * 911382323 + tp_idx)

    # 3) Randomly determine T (3..8) and generate n for each group satisfying constraints
    T = rng.randint(3, 8)
    k_small = T - 1
    # Ensure at least one group >= MAX_N_TOTAL//2, and total sum <= MAX_N_TOTAL
    lower_big = MAX_N_TOTAL // 2 + 1
    upper_big = MAX_N_TOTAL - minimal_sum_distinct(k_small, 2)
    if upper_big < lower_big:
        # In extreme cases reduce T, but won't happen when T<=8
        T = 2
        k_small = 1
        upper_big = MAX_N_TOTAL - minimal_sum_distinct(k_small, 2)
        lower_big = MAX_N_TOTAL // 2 + 1
    big_n = rng.randint(lower_big, upper_big)

    remaining_budget = MAX_N_TOTAL - big_n
    small_sizes = gen_increasing_sizes(k_small, remaining_budget, rng, start_min=2)
    sizes = small_sizes + [big_n]
    rng.shuffle(sizes)

    # 4) Generate trees for each group
    groups = []
    for n in sizes:
        is_big = (n == big_n)
        edges = pick_structure(n, is_big, rng)
        assert len(edges) == max(0, n - 1), f"edge count mismatch for n={n}"
        for (u, v) in edges:
            assert 1 <= u <= n and 1 <= v <= n and u != v
        seen = set()
        for (u, v) in edges:
            a, b = (u, v) if u < v else (v, u)
            assert (a, b) not in seen
            seen.add((a, b))
        groups.append((n, edges))

    # 5) Write i.in and i.ans
    in_path = f"{tp_idx}.in"
    ans_path = f"{tp_idx}.ans"

    with open(in_path, "w") as fin:
        fin.write(f"{T}\n")
        for (n, _) in groups:
            fin.write(f"{n}\n")

    with open(ans_path, "w") as fans:
        for (n, edges) in groups:
            weighted = assign_weights(edges, rng)
            for (u, v, w) in weighted:
                fans.write(f"{u} {v} {w}\n")

    # 6) Console summary
    total_n = sum(n for n, _ in groups)
    print(f"Generated test point {tp_idx}:")
    print(f"  Number of groups T = {T}")
    print(f"  Each group n = {[n for (n, _) in groups]}")
    print(f"  Total n = {total_n} (<= {MAX_N_TOTAL})")
    print(f"  At least one group n>={MAX_N_TOTAL//2}: {any(n >= MAX_N_TOTAL//2 for n, _ in groups)}")
    print(f"Output files: {in_path}, {ans_path}")

if __name__ == "__main__":
    main()