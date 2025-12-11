import argparse
import random
import secrets
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate 10 .in/.ans pairs for G up to 10000.")
    parser.add_argument("--master-seed", type=int, default=None,
                        help="Optional master seed to make the whole set reproducible.")
    args = parser.parse_args()

    # RNG: either deterministic (random.Random) or OS entropy (secrets.SystemRandom)
    rng = random.Random(args.master_seed) if args.master_seed is not None else secrets.SystemRandom()

    # Different scales of G, all <= 10000
    # G_list = [100 for _ in range(10)]
    G_list = [1000, 1000, 1000, 3000, 3000, 5000, 5000, 10000, 10000, 10000]

    out_dir = Path(".")
    U64_MAX = (1 << 64) - 1

    for idx, G in enumerate(G_list, start=1):
        # Public input (.in): just G
        in_path = out_dir / f"{idx}.in"
        with in_path.open("w", encoding="utf-8", newline="\n") as f_in:
            f_in.write(f"{G}\n")

        # Hidden answer (.ans):
        # - seed_sampling: one 64-bit integer
        # - G_ans: integer, here set to G (ensures G_ans >= G)
        # - hand_seed[i]: G_ans lines of 64-bit integers
        seed_sampling = rng.getrandbits(63)  # 0..2^64-1

        G_ans = G  # keep equal to G; can be changed if needed to be > G

        hand_seeds = [rng.getrandbits(63) for _ in range(G_ans)]

        ans_path = out_dir / f"{idx}.ans"
        with ans_path.open("w", encoding="utf-8", newline="\n") as f_ans:
            f_ans.write(f"{seed_sampling}\n")
            f_ans.write(f"{G_ans}\n")
            for s in hand_seeds:
                # ensure within [0, 2^64-1] and in decimal text
                s &= U64_MAX
                f_ans.write(f"{s}\n")

        # Optional: print summary to console
        print(f"Wrote {in_path.name} (G={G}) and {ans_path.name} (seed_sampling={seed_sampling}, G_ans={G_ans})")

if __name__ == "__main__":
    main()