# Python script to generate 20 test cases with n=1000 tree data files (test01.txt ~ test20.txt).
# It covers multiple tree types: long chains, stars (high-degree nodes), chain+star combinations,
# balanced distributions, and other representative patterns. The script validates each tree
# (connectivity, edge count), computes diameter and max degree, and outputs a summary.

# Usage:
# - Dependencies: Python 3, no external libraries required.
# - Run: python3 gen_trees.py
# - Output: Creates 1.in~20.in and 1.ans~20.ans in current directory
#   - .in file: First line contains 1000
#   - .ans file: 999 lines with "u v" representing edges
# - Console prints each test case name, diameter, and max degree for verification.

# 20 patterns covered (one output file per pattern):
# 1. Path-1000: Pure chain (long path, diameter≈999)
# 2. Star-1000: Pure star (hub-and-spoke, center degree≈999)
# 3. Broom-L700: Broom tree, long handle + large star (chain + star)
# 4. Balanced-3ary: Nearly ternary balanced expansion (uniform)
# 5. Caterpillar-L700: Caterpillar (long spine + uniform leaves, long chain)
# 6. Double-Star: Two stars connected by a bridge (two large hubs)
# 7. Random-Prufer: Random Prufer sequence (uniformly random tree)
# 8. PrefAttach-m1: Preferential attachment m=1 (power-law, scale-free-like)
# 9. Comb-L≈750 gap=3: Long comb (long backbone + regular short teeth)
# 10. Snowflake-40arms: Center + multiple equal-length arms (snowflake-like)
# 11. Two-Level-10hubs: Two-level star (root-subhub-leaf hierarchy)
# 12. Grid-31x32: Grid spanning tree approximation (locally uniform, low diameter)
# 13. Cluster-Path-10: 10 star clusters connected in a path (community + backbone)
# 14. Balanced-Binary: Nearly complete binary tree (uniform, low diameter)
# 15. Deep+3Hubs: Long chain with three large hubs (long chain + multiple stars)
# 16. Geometric-NN: 2D nearest-neighbor spanning tree (geometric, uniform)
# 17. Triple-Brooms: Three brooms concatenated (long chain + multiple stars)
# 18. Backbone+MiniStars: Main backbone + periodic small stars (repeating structure)
# 19. Center+2LongArms: Center with two long arms + many leaves (long chain + star)
# 20. Rand-Deg<=3: Random tree with degree cap of 3 (uniform degree control)

import os
import random
import math
from collections import deque, defaultdict

# Whether to include n in the first line of .ans file
INCLUDE_N_IN_ANS = False

N = 1000

def validate_tree(n, edges):
    assert len(edges) == n - 1, f"Edge count != n-1: {len(edges)}"
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        assert 1 <= u <= n and 1 <= v <= n and u != v
        adj[u].append(v)
        adj[v].append(u)
    # connectivity
    vis = [False] * (n + 1)
    q = deque([1])
    vis[1] = True
    cnt = 1
    while q:
        u = q.popleft()
        for w in adj[u]:
            if not vis[w]:
                vis[w] = True
                cnt += 1
                q.append(w)
    assert cnt == n, f"Not connected: visited {cnt}/{n}"
    # diameter via 2 BFS
    def bfs_far(s):
        dist = [-1] * (n + 1)
        q = deque([s])
        dist[s] = 0
        far = s
        while q:
            u = q.popleft()
            if dist[u] > dist[far]:
                far = u
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return far, dist[far], dist
    a, _, _ = bfs_far(1)
    b, diam, _ = bfs_far(a)
    # max degree
    max_deg = max(len(adj[i]) for i in range(1, n + 1))
    return diam, max_deg

def write_in_ans(idx, n, edges):
    # Write 1.in ~ 20.in: contains only one number n
    with open(f"{idx}.in", "w") as f_in:
        f_in.write(f"{n}\n")
    # Write 1.ans ~ 20.ans: edge list (u v), optionally write n in first line
    with open(f"{idx}.ans", "w") as f_ans:
        if INCLUDE_N_IN_ANS:
            f_ans.write(f"{n}\n")
        for u, v in edges:
            f_ans.write(f"{u} {v}\n")

# Generators

def gen_path(n=N):
    return [(i, i + 1) for i in range(1, n)]

def gen_star(n=N, center=1):
    return [(center, i) for i in range(1, n + 1) if i != center]

def gen_broom(n=N, path_len=700, center=1):
    # path nodes: 1..L ; leaves: L+1..n attached to center
    L = min(path_len, n)
    edges = []
    for i in range(1, L):
        edges.append((i, i + 1))
    for x in range(L + 1, n + 1):
        edges.append((center, x))
    return edges

def gen_balanced_kary(n=N, k=3):
    edges = []
    q = deque([1])
    nxt = 2
    while nxt <= n and q:
        u = q.popleft()
        for _ in range(k):
            if nxt > n:
                break
            edges.append((u, nxt))
            q.append(nxt)
            nxt += 1
    return edges

def gen_caterpillar(n=N, spine_len=700):
    L = min(spine_len, n)
    edges = []
    for i in range(1, L):
        edges.append((i, i + 1))
    rem = n - L
    host = 1
    node = L + 1
    while rem > 0:
        edges.append((host, node))
        node += 1
        rem -= 1
        host += 1
        if host > L:
            host = 1
    return edges

def gen_double_star(n=N):
    edges = [(1, 2)]
    rem = n - 2
    left = rem // 2
    right = rem - left
    node = 3
    for _ in range(left):
        edges.append((1, node))
        node += 1
    for _ in range(right):
        edges.append((2, node))
        node += 1
    return edges

def gen_random_prufer(n=N, seed=202401):
    rnd = random.Random(seed)
    prufer = [rnd.randint(1, n) for _ in range(n - 2)]
    deg = [0] * (n + 1)
    for i in range(1, n + 1):
        deg[i] = 1
    for x in prufer:
        deg[x] += 1
    import heapq
    heap = []
    for i in range(1, n + 1):
        if deg[i] == 1:
            heapq.heappush(heap, i)
    edges = []
    for x in prufer:
        u = heapq.heappop(heap)
        edges.append((u, x))
        deg[u] -= 1
        deg[x] -= 1
        if deg[x] == 1:
            heapq.heappush(heap, x)
    u = heapq.heappop(heap)
    v = heapq.heappop(heap)
    edges.append((u, v))
    return edges

def gen_preferential_attachment(n=N, seed=202402):
    rnd = random.Random(seed)
    edges = [(1, 2)]
    deg = [0] * (n + 1)
    deg[1] = 1
    deg[2] = 1
    bag = [1, 2]  # node i appears deg[i] times
    for i in range(3, n + 1):
        p = rnd.choice(bag)
        edges.append((p, i))
        deg[p] += 1
        deg[i] = 1
        bag.append(p)
        bag.append(i)
    return edges

def gen_comb(n=N, tooth_every=3):
    # maximize L with L + floor(L/tooth_every) <= n
    L = n
    while L + (L // tooth_every) > n:
        L -= 1
    edges = []
    for i in range(1, L):
        edges.append((i, i + 1))
    node = L + 1
    pos = 1
    while node <= n and pos <= L:
        edges.append((pos, node))
        node += 1
        pos += tooth_every
    return edges

def gen_snowflake(n=N, arms=40):
    edges = []
    center = 1
    node = 2
    base = (n - 1) // arms
    rem = (n - 1) % arms
    for a in range(arms):
        length = base + (1 if a < rem else 0)
        prev = center
        for _ in range(length):
            edges.append((prev, node))
            prev = node
            node += 1
    return edges

def gen_two_level(n=N, hubs=10):
    edges = []
    # connect hubs to root
    for h in range(2, 2 + hubs):
        edges.append((1, h))
    node = 2 + hubs
    idx = 0
    hub_nodes = list(range(2, 2 + hubs))
    while node <= n:
        edges.append((hub_nodes[idx], node))
        idx = (idx + 1) % len(hub_nodes)
        node += 1
    return edges

def gen_grid_tree(n=N, R=31, C=32):
    RC = R * C
    RC = min(RC, n)
    edges = []
    # map (r,c) to id: 1..RC
    def id_rc(r, c):
        return (r - 1) * C + c
    # Ensure (1,1) = 1
    for r in range(1, R + 1):
        for c in range(1, C + 1):
            idx = id_rc(r, c)
            if idx > RC: break
            if r == 1 and c == 1:
                continue
            if r > 1:
                p = id_rc(r - 1, c)
            else:
                p = id_rc(r, c - 1)
            if p <= RC and idx <= RC:
                edges.append((p, idx))
    # leftover nodes
    node = RC + 1
    last_row = [id_rc(R, c) for c in range(1, C + 1) if id_rc(R, c) <= RC]
    idx = 0
    while node <= n:
        edges.append((last_row[idx], node))
        idx = (idx + 1) % len(last_row)
        node += 1
    return edges

def gen_cluster_path(n=N, clusters=10):
    edges = []
    centers = []
    for i in range(clusters):
        centers.append(i + 1)
    for i in range(clusters - 1):
        edges.append((centers[i], centers[i + 1]))
    leaves = n - clusters
    base = leaves // clusters
    rem = leaves % clusters
    node = clusters + 1
    for i in range(clusters):
        cnt = base + (1 if i < rem else 0)
        for _ in range(cnt):
            edges.append((centers[i], node))
            node += 1
    return edges

def gen_balanced_binary(n=N):
    return gen_balanced_kary(n, k=2)

def gen_deep_three_hubs(n=N, path_len=700):
    L = min(path_len, n)
    edges = []
    for i in range(1, L):
        edges.append((i, i + 1))
    hubs = [1, (L + 1) // 2, L]
    rem = n - L
    node = L + 1
    # distribute leaves 1:2:1
    weights = [1, 2, 1]
    total_w = sum(weights)
    targets = [rem * w // total_w for w in weights]
    while sum(targets) < rem:
        for i in range(len(targets)):
            if sum(targets) < rem:
                targets[i] += 1
    for idx, cnt in enumerate(targets):
        u = hubs[idx]
        for _ in range(cnt):
            edges.append((u, node))
            node += 1
    return edges

def gen_geometric_nn(n=N, seed=202403):
    rnd = random.Random(seed)
    pts = [(0.0, 0.0)] * (n + 1)
    for i in range(1, n + 1):
        pts[i] = (rnd.random(), rnd.random())
    edges = []
    for i in range(2, n + 1):
        xi, yi = pts[i]
        best = 1
        bestd = (xi - pts[1][0]) ** 2 + (yi - pts[1][1]) ** 2
        for j in range(2, i):
            dx = xi - pts[j][0]
            dy = yi - pts[j][1]
            d = dx * dx + dy * dy
            if d < bestd:
                bestd = d
                best = j
        edges.append((best, i))
    return edges

def gen_triple_brooms(n=N, Ls=(250, 250, 250), leaves=250):
    edges = [(1, 2), (2, 3)]
    node = 4
    centers = [1, 2, 3]
    for ci, L in zip(centers, Ls):
        prev = ci
        for _ in range(L):
            if node > n: break
            edges.append((prev, node))
            prev = node
            node += 1
    # attach remaining leaves to centers round-robin
    while len(edges) < n - 1 and node <= n:
        for ci in centers:
            if len(edges) >= n - 1 or node > n:
                break
            edges.append((ci, node))
            node += 1
    return edges

def gen_backbone_ministars(n=N, backbone=200, period=5, arm=10):
    edges = []
    L = min(backbone, n)
    for i in range(1, L):
        edges.append((i, i + 1))
    node = L + 1
    i = 1
    while node <= n and i <= L:
        if (i % period) == 0:
            for _ in range(arm):
                if node > n: break
                edges.append((i, node))
                node += 1
        i += 1
    while node <= n:
        edges.append((L, node))
        node += 1
    return edges

def gen_center_two_long_arms(n=N, L1=400, L2=400):
    edges = []
    center = 1
    node = 2
    prev = center
    for _ in range(L1):
        if node > n: break
        edges.append((prev, node))
        prev = node
        node += 1
    prev = center
    for _ in range(L2):
        if node > n: break
        edges.append((prev, node))
        prev = node
        node += 1
    while node <= n:
        edges.append((center, node))
        node += 1
    return edges

def gen_random_deg_cap(n=N, cap=3, seed=202404):
    rnd = random.Random(seed)
    cap_left = defaultdict(int)
    cap_left[1] = cap
    eligible = [1]  # nodes with cap_left > 0
    edges = []
    for node in range(2, n + 1):
        p = rnd.choice(eligible)
        edges.append((p, node))
        cap_left[p] -= 1
        if cap_left[p] == 0:
            eligible = [x for x in eligible if x != p]
        cap_left[node] = cap
        eligible.append(node)
    return edges

def call_gen(fn, n, kwargs):
    if 'n' in fn.__code__.co_varnames:
        return fn(n=n, **kwargs)
    else:
        return fn(**kwargs)

def main():
    random.seed(123456)

    patterns = [
        ("Path-1000", gen_path, {}),
        ("Star-1000", gen_star, {}),
        ("Broom-L700", gen_broom, {"path_len": 700}),
        ("Balanced-3ary", gen_balanced_kary, {"k": 3}),
        ("Caterpillar-L700", gen_caterpillar, {"spine_len": 700}),
        ("Double-Star", gen_double_star, {}),
        ("Random-Prufer", gen_random_prufer, {"seed": 202401}),
        ("PrefAttach-m1", gen_preferential_attachment, {"seed": 202402}),
        ("Comb-L~750-gap3", gen_comb, {"tooth_every": 3}),
        ("Snowflake-40arms", gen_snowflake, {"arms": 40}),
        ("Two-Level-10hubs", gen_two_level, {"hubs": 10}),
        ("Grid-31x32", gen_grid_tree, {"R": 31, "C": 32}),
        ("Cluster-Path-10", gen_cluster_path, {"clusters": 10}),
        ("Balanced-Binary", gen_balanced_binary, {}),
        ("Deep+3Hubs", gen_deep_three_hubs, {"path_len": 700}),
        ("Geometric-NN", gen_geometric_nn, {"seed": 202403}),
        ("Triple-Brooms", gen_triple_brooms, {"Ls": (250, 250, 250)}),
        ("Backbone+MiniStars", gen_backbone_ministars, {"backbone": 200, "period": 5, "arm": 10}),
        ("Center+2LongArms", gen_center_two_long_arms, {"L1": 400, "L2": 400}),
        ("Rand-Deg<=3", gen_random_deg_cap, {"cap": 3, "seed": 202404}),
    ]

    summary = []
    for idx, (name, fn, kwargs) in enumerate(patterns, start=1):
        edges = call_gen(fn, N, kwargs)
        diam, max_deg = validate_tree(N, edges)
        write_in_ans(idx, N, edges)
        summary.append((idx, name, diam, max_deg))

    print("Generated 1.in~20.in and 1.ans~20.ans in current directory")
    print("Index  Name                   Diameter  MaxDeg")
    for idx, name, diam, max_deg in summary:
        print(f"{idx:02d}     {name:20s}  {diam:8d}  {max_deg:6d}")

if __name__ == "__main__":
    main()


# Design points and structure guarantees
# - Long chains:
#   - test01 Path-1000 diameter≈999
#   - test03 Broom-L700 diameter≥700
#   - test05 Caterpillar-L700 diameter≥700
#   - test09 Comb-L~750-gap3 diameter≥750
#   - test15 Deep+3Hubs diameter≥700
#   - test19 Center+2LongArms two arms total≈800
# - Star structures (high degree):
#   - test02 Star-1000 max degree≈999
#   - test03 Broom: center with≈300 leaves
#   - test06 Double-Star: two large hubs
#   - test11 Two-Level-10hubs: sub-hub degree≈100
#   - test19 Center+2LongArms: center with two long arms + leaves
# - Uniform/balanced structures:
#   - test04 Balanced-3ary
#   - test07 Random-Prufer
#   - test12 Grid-31x32
#   - test14 Balanced-Binary
#   - test16 Geometric-NN
#   - test20 Rand-Deg<=3

# Adjustments as needed:
# - For deeper chains: increase path_len or arm length
# - For more extreme stars: increase Two-Level hubs or leaf count per hub
# - For more uniform structures: increase k in k-ary/binary or use multiple random Prufer seeds