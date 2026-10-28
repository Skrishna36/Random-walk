#!/usr/bin/env python3
"""
Final-position distribution of a 1D random walk, scaled by sqrt(N).
usage: python3 walk.py [particles=100000]
"""
import sys
import numpy as np


def endpoint_histogram(N, M, rng):
    """Simulate M walks of N steps. Returns hist with index = pos + N, pos in [-N, N]."""
    hist = np.zeros(2 * N + 1, dtype=np.int64)
    batch = max(1, 20_000_000 // N)            # keeps memory around 20 MB
    done = 0
    while done < M:
        b = min(batch, M - done)
        up = rng.integers(0, 2, size=(b, N), dtype=np.int8).sum(axis=1, dtype=np.int32)
        # pos = 2*up - N, so pos + N = 2*up
        hist += np.bincount(2 * up, minlength=2 * N + 1)
        done += b
    return hist


def main():
    M = int(float(sys.argv[1])) if len(sys.argv) > 1 else 100_000
    steps = [50, 100, 200]
    rng = np.random.default_rng()              # seeded once from OS entropy

    print("Final position vs number of particles:")
    with open("walk4.dat", "w") as fout:
        for N in steps:
            hist = endpoint_histogram(N, M, rng)
            sN = np.sqrt(N)

            x = np.arange(-N, N + 1, 2)
            scaled = hist[x + N] / M * sN
            theory = 2.0 / np.sqrt(2.0 * np.pi) * np.exp(-x**2 / (2.0 * N))
            data = np.column_stack([x, scaled, x / sN, theory])

            header = f"# N = {N}\n# x  P(x)*sqrt(N)  x/sqrt(N)  theory\n"
            print(header, end="")
            fout.write(header)
            for row in data:
                line = f"{int(row[0])} {row[1]:.6g} {row[2]:.6g} {row[3]:.6g}\n"
                print(line, end="")
                fout.write(line)
            print()
            fout.write("\n\n")                 # two blank lines = new gnuplot index


if __name__ == "__main__":
    main()
