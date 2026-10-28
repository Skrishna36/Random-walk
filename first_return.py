#!/usr/bin/env python3
"""
First-return probability of a 1D symmetric random walk.
usage: python first_return.py N E filenum [walks_per_ensemble=1e7] [batch=2e6]
"""
import sys
import numpy as np


def first_return_hits(N, M, rng, batch=2_000_000):
    """Simulate M walks (max length N). hits[k] = # walks whose first return is at step 2k."""
    K = N // 2
    hits = np.zeros(K + 1, dtype=np.int64)
    remaining = M
    while remaining > 0:
        b = min(batch, remaining)
        remaining -= b
        pos = np.zeros(b, dtype=np.int32)          # positions of walks still out
        for k in range(1, K + 1):                  # two steps at a time
            r = rng.integers(0, 4, size=pos.size, dtype=np.int8)
            pos += 2 * ((r & 1) + (r >> 1)) - 2    # net move: -2, 0, or +2
            back = pos == 0
            n_back = np.count_nonzero(back)
            if n_back:
                hits[k] += n_back
                pos = pos[~back]                   # drop walks that returned
                if pos.size == 0:
                    break
    return hits


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    N, E, filenum = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    M = int(float(sys.argv[4])) if len(sys.argv) > 4 else 10_000_000
    batch = int(float(sys.argv[5])) if len(sys.argv) > 5 else 2_000_000
    K = N // 2

    rng = np.random.default_rng()                  # seeded once from OS entropy
    f = np.zeros(K + 1)
    for en in range(1, E + 1):
        f += first_return_hits(N, M, rng, batch) / (M * E)
        print(f"ensem {en} ok", flush=True)

    k = np.arange(1, K + 1)
    u = np.cumprod((2 * k - 1) / (2 * k))          # C(2k,k)/4^k
    exact = u / (2 * k - 1)

    with np.errstate(divide="ignore"):
        data = np.column_stack([k, np.log(k), np.log(f[1:]), f[1:], exact])

    fname = f"1streturn_N{N}_en{E}_fnum{filenum}.dat"
    np.savetxt(fname, data, fmt=["%d", "%.10g", "%.10g", "%.10g", "%.10g"], delimiter="  ")
    for row in data:
        print(f"{int(row[0])}  " + "  ".join(f"{x:.10g}" for x in row[1:]))


if __name__ == "__main__":
    main()
