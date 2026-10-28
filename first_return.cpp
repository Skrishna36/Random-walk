// Build: g++ -O3 -march=native -fopenmp first_return.cpp -o first_return
// Run:   ./first_return N E filenum [walks_per_ensemble=1e7]
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " N E filenum [walks_per_ensemble]\n";
        return 1;
    }
    const int N = std::atoi(argv[1]);
    const int E = std::atoi(argv[2]);
    const int filenum = std::atoi(argv[3]);
    const long long M = (argc > 4) ? std::atoll(argv[4]) : 10000000LL;
    const int K = N / 2;  // returns can only happen at even steps 2k, k = 1..K

    const std::string fname = "1streturn_N" + std::to_string(N) + "_en" + std::to_string(E) +
                              "_fnum" + std::to_string(filenum) + ".dat";

    // One independent, well-seeded generator per thread, seeded ONCE.
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::random_device rd;
    std::vector<std::mt19937_64> gens;
    gens.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        std::seed_seq ss{rd(), rd(), rd(), rd(), (unsigned)filenum, (unsigned)t};
        gens.emplace_back(ss);
    }

    std::vector<long double> f(K + 1, 0.0L);

    for (int en = 1; en <= E; ++en) {
        std::vector<long long> hits(K + 1, 0);

        #pragma omp parallel num_threads(nthreads)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            std::mt19937_64& g = gens[tid];
            std::vector<long long> local(K + 1, 0);
            std::uint64_t bits = 0;
            int left = 0;

            #pragma omp for schedule(static)
            for (long long w = 0; w < M; ++w) {
                int pos = 0;
                for (int k = 1; k <= K; ++k) {          // two steps per iteration
                    if (left < 2) { bits = g(); left = 64; }
                    int up = (int)(bits & 1u) + (int)((bits >> 1) & 1u);
                    pos += 2 * up - 2;                  // net move: -2, 0, or +2
                    bits >>= 2; left -= 2;
                    if (pos == 0) { ++local[k]; break; } // first return at step 2k
                }
            }

            #pragma omp critical
            for (int k = 1; k <= K; ++k) hits[k] += local[k];
        }

        for (int k = 1; k <= K; ++k)
            f[k] += (long double)hits[k] / ((long double)M * E);

        std::cout << "ensem " << en << " ok" << std::endl;
    }

    std::ofstream file(fname);
    long double u = 1.0L;  // u_k = C(2k,k) / 4^k
    for (int k = 1; k <= K; ++k) {
        u *= (long double)(2 * k - 1) / (2 * k);
        const long double exact = u / (2 * k - 1);
        auto write = [&](std::ostream& os) {
            os << std::setprecision(10) << k << "  " << std::log((double)k) << "  "
               << std::log((double)f[k]) << "  " << f[k] << "  " << exact << '\n';
        };
        write(std::cout);
        write(file);
    }
    return 0;
}
