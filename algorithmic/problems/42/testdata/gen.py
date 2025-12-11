import math
import sys
from functools import lru_cache
import random

def s_base(n: int) -> float:
    # Returns s(n) for 1..100
    if n == 1:
        return 1.0
    if 2 <= n <= 4:
        return 2.0
    if n == 5:
        return 2.0 + 1.0 / math.sqrt(2.0)
    if 6 <= n <= 9:
        return 3.0
    if n == 10:
        return 3.0 + 1.0 / math.sqrt(2.0)
    if n == 11:
        return 3.8771
    if 12 <= n <= 16:
        return 4.0
    if n == 17:
        return 4.6756
    if n == 18:
        return 3.5 + 0.5 * math.sqrt(7.0)
    if n == 19:
        return 3.0 + (4.0/3.0) * math.sqrt(2.0)
    if 20 <= n <= 25:
        return 5.0
    if n == 26:
        return 3.5 + 1.5 * math.sqrt(2.0)
    if n == 27:
        return 5.0 + 1.0 / math.sqrt(2.0)
    if n == 28:
        return 3.0 + 2.0 * math.sqrt(2.0)
    if n == 29:
        return 5.9344
    if 30 <= n <= 36:
        return 6.0
    if n == 37:
        return 6.5987
    if n == 38:
        return 6.0 + 1.0 / math.sqrt(2.0)
    if n == 39:
        return 6.8189
    if n == 40:
        return 4.0 + 2.0 * math.sqrt(2.0)
    if n == 41:
        return 6.9473
    if 42 <= n <= 49:
        return 7.0
    if n == 50:
        return 7.5987
    if n == 51:
        return 7.7044
    if n == 52:
        return 7.0 + 1.0 / math.sqrt(2.0)
    if n == 53:
        return 7.8231
    if n == 54:
        return 7.8488
    if n == 55:
        return 7.9871
    if 56 <= n <= 64:
        return 8.0
    if n == 65:
        return 5.0 + 5.0 / math.sqrt(2.0)
    if n == 66:
        return 3.0 + 4.0 * math.sqrt(2.0)
    if n == 67:
        return 8.0 + 1.0 / math.sqrt(2.0)
    if n == 68:
        return 7.5 + 0.5 * math.sqrt(7.0)
    if n == 69:
        return 8.8562
    if n == 70:
        return 8.9121
    if n == 71:
        return 8.9633
    if 72 <= n <= 81:
        return 9.0
    if n == 82:
        return 6.0 + 5.0 / math.sqrt(2.0)
    if n == 83:
        return 4.0 + 4.0 * math.sqrt(2.0)
    if n == 84:
        return 9.0 + 1.0 / math.sqrt(2.0)
    if n == 85:
        return 5.5 + 3.0 * math.sqrt(2.0)
    if n == 86:
        return 8.5 + 0.5 * math.sqrt(7.0)
    if n == 87:
        return 9.8520
    if n == 88:
        return 9.9018
    if n == 89:
        return 5.0 + 7.0 / math.sqrt(2.0)
    if 90 <= n <= 100:
        return 10.0
    raise ValueError("s_base only defined for 1..100")

@lru_cache(maxsize=None)
def S(n: int) -> float:
    if n <= 100:
        return s_base(n)
    # fractal extension
    return 2.0 * S((n + 3) // 4)  # ceil(n/4)

def is_perfect_square(x: int) -> bool:
    r = math.isqrt(x)
    return abs(r * r - x) <= 1e-9 or abs(S(x) - math.ceil(math.sqrt(x))) <= 1e-9

import math
import random
import sys

# Assumes the following functions are defined:
# def is_perfect_square(n): ...
# def S(n): ...

def sample_log_uniform_int(lo: int, hi: int) -> int:
    """Sample an integer in [lo, hi] with logarithmic distribution."""
    if lo < 1:
        lo = 1
    if lo == hi:
        return lo
    u = random.random()
    x = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    n = int(round(x))
    return min(max(n, lo), hi)

def pick_in_bucket(lo: int, hi: int, need: int, picked_set: set) -> list:
    """Sample 'need' values of n in range [lo, hi] that satisfy conditions."""
    found = []
    attempts = 0
    max_attempts = max(2000, (hi - lo + 1) * 5)

    while len(found) < need and attempts < max_attempts:
        attempts += 1
        n = sample_log_uniform_int(lo, hi)
        if n in picked_set or is_perfect_square(n):
            continue
        s = S(n)
        if s > math.sqrt(n) + 1e-12:
            found.append(n)
            picked_set.add(n)

    if len(found) < need:
        order = range(lo, hi + 1) if random.random() < 0.5 else range(hi, lo - 1, -1)
        for n in order:
            if len(found) >= need:
                break
            if n in picked_set or is_perfect_square(n):
                continue
            s = S(n)
            if s > math.sqrt(n) + 1e-12:
                found.append(n)
                picked_set.add(n)

    return found

def main():
    random.seed()
    picked = []
    picked_set = set()

    base_plan = [
        (5, 9, 0),
        (10, 100, 4),
        (100, 1000, 2),
        (1000, 10000, 2),
        (10000, 100000, 2),
    ]
    plan = base_plan + base_plan  # x2 = 20 total

    for lo, hi, cnt in plan:
        got = pick_in_bucket(lo, hi, cnt, picked_set)
        if len(got) < cnt:
            print(f"Failed to find {cnt} suitable n in [{lo}, {hi}]", file=sys.stderr)
            sys.exit(1)
        picked.extend(got)

    random.shuffle(picked)

    for idx, n in enumerate(picked, start=1):
        s_val = S(n)
        with open(f"{idx}.in", "w") as f_in:
            f_in.write(f"{n}\n")
        with open(f"{idx}.ans", "w") as f_ans:
            f_ans.write(f"{s_val}\n")

    print("Generated", len(picked), "pairs.")
    print("Chosen n:", " ".join(map(str, picked)))

if __name__ == "__main__":
    main()